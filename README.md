# Perf-Scope | GPU Performance Observatory

Perf-Scope will measure GPU workload performance, detect slowdowns that exceed normal run-to-run variation, and collect profiler evidence for confirmed regressions. The first release will run on one Linux machine with one NVIDIA GPU. macOS and CPU CI will exercise the runner, analysis, and reporting with labeled fixtures.

## Scope

| Milestone | Deliverable |
| --- | --- |
| **0.1 · CUDA regression pipeline** | Naïve and optimized FP32 transpose and GEMM, with cuBLAS as a GEMM reference. One command runs correctness checks, takes paired timings, compares variants, and generates a report. A seeded slowdown triggers targeted profiling. |
| **0.2 · Workload coverage** | ResNet-50 inference in PyTorch eager and TorchInductor, plus one NVIDIA STREAM case on a supported GPU. Store all three workload tiers in the same results format. |
| **0.3 · Adaptive testing experiment** | Compare a budgeted scheduler with fixed-sample testing on seeded regressions. Report detection recall, false positives, diagnosis accuracy, and GPU time. |

Start with **transpose on two shapes**, one aligned and one rectangular with partial tiles. Run a fixture through the full pipeline on the Mac, then run the CUDA pair on a GPU and check that the optimized implementation produces the same values. Reverse the comparison to exercise slowdown detection and the profiling trigger.

## Target workflow

The command-line interface is planned. The first benchmark configuration will be run with this command.

~~~bash
perf-scope run configs/transpose.toml
~~~

Each invocation will produce a run manifest, raw timing samples, comparison results, available profiler artifacts, and a local report under a unique run directory. The report will identify which results came from real GPU runs or fixtures and show the workload, device, precision, timing scope, and uncertainty.

See the [implementation plan](docs/roadmap.md) for the measurement protocol and the [review sequence](docs/changes.md) for dependencies and acceptance checks.
