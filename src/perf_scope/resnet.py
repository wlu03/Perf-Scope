import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return parsed


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m perf_scope.resnet")
    parser.add_argument("backend", choices=("eager", "inductor"))
    parser.add_argument("--shape", type=_positive, metavar="BATCH", required=True)
    parser.add_argument("--dtype", choices=("fp32",), required=True)
    parser.add_argument("--seed", type=_nonnegative, required=True)
    parser.add_argument("--warmups", type=_nonnegative, required=True)
    parser.add_argument("--iterations", type=_positive, required=True)
    parser.add_argument("--cold-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.seed >= 2**64:
        parser.error("--seed must fit in an unsigned 64-bit integer")
    if args.cold_cache and args.backend != "inductor":
        parser.error("--cold-cache requires the inductor backend")
    return args


def _benchmark(args: argparse.Namespace) -> dict:
    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ResNet-50 CUDA benchmark requires a CUDA-enabled PyTorch build and NVIDIA GPU"
        )

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    gpu_id = f"GPU-{str(properties.uuid).removeprefix('GPU-')}"
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.shape, 3, 224, 224, generator=rng, dtype=torch.float32
    ).to(device)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval().to(device)
    compile_model = args.backend == "inductor"
    invoke = (
        torch.compile(model, backend="inductor", fullgraph=True)
        if compile_model
        else model
    )

    with torch.inference_mode():
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        first = invoke(inputs)
        torch.cuda.synchronize(device)
        first_call_ms = 1000 * (time.perf_counter() - began)
        correct = bool(torch.isfinite(first).all().item())
        if compile_model:
            reference = model(inputs)
            correct = correct and torch.allclose(first, reference, rtol=1e-3, atol=1e-4)
            del reference
        del first

        for _ in range(args.warmups):
            invoke(inputs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            invoke(inputs)
        stop.record()
        stop.synchronize()
        latency = start.elapsed_time(stop) / args.iterations

    if latency <= 0:
        raise RuntimeError("CUDA events did not resolve a positive block duration")
    return {
        "device": gpu_id,
        "correct": correct,
        "metrics": {
            "latency_ms": latency,
            "items_s": args.shape * 1000 / latency,
            "first_call_ms": first_call_ms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
        "timing_scope": "cuda_event_warm",
        "backend": args.backend,
        "checkpoint": weights.url,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
        },
        "device_name": properties.name,
        "inductor_cache": ("isolated" if args.cold_cache else "shared")
        if compile_model
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    isolation = (
        tempfile.TemporaryDirectory(prefix="perf-scope-inductor-")
        if args.cold_cache
        else nullcontext()
    )
    try:
        with isolation as cache:
            if cache is not None:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cache) / "inductor")
                os.environ["TRITON_CACHE_DIR"] = str(Path(cache) / "triton")
            print(json.dumps(_benchmark(args), allow_nan=False))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
