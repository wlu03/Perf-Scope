# Perf-Scope

Perf-Scope benchmarks CUDA kernels, ResNet-50 inference, and NVIDIA STREAM. It detects paired slowdowns, profiles confirmed CUDA regressions, and replays screening policies under a GPU-time budget.

## Install

~~~bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
~~~

## Run artifacts

A suite declares cases, shapes, primary metrics, and baseline and candidate argument arrays. Run from the checkout root. The manifest records the resolved suite, Git revision, and executable hashes. Each run saves the JSONL samples, raw subprocess logs, and `report.html`. Use `perf-scope compare` with a run directory to regenerate decisions and the report.

Each pair runs both variants in seeded order. Given at least eight pairs, the comparison bootstraps whole pairs to estimate a 95% percentile interval around median relative slowdown. The default margin is 5%, controlled by `slowdown_margin`. Invalid runs exit with status 1. With `--fail-on-regression`, a confirmed slowdown exits with status 3.

## CUDA benchmarks

Build on Linux with the CUDA toolkit and CMake 3.24 or later. Set `-DCMAKE_CUDA_ARCHITECTURES` for the target GPU. The default is 80.

~~~bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel
ctest --test-dir build/cuda -L gpu --output-on-failure
perf-scope run configs/transpose.toml
perf-scope run configs/gemm.toml
perf-scope run configs/bandwidth.toml
perf-scope run configs/softmax.toml
~~~

The two FP32 variants share their inputs and launch geometry. The tiled variant uses a padded shared-memory tile for coalesced reads and writes. CUDA events measure the average launch latency for each block. The executable validates the transpose and reports effective bandwidth from one logical matrix read and write.

GEMM compares naïve and tiled FP32 kernels with a cuBLAS reference using pedantic FP32 compute. GFLOP/s counts $2MNK$ operations. The reference checks up to 512 entries against a double-precision CPU calculation with absolute tolerance $10^{-3}$ and relative tolerance $10^{-4}$. The vector triad reads two arrays and writes one. Use a triad working set larger than the device's recorded L2 cache for bandwidth calibration.

## Profiling and roofline

With Nsight Compute on PATH, confirmed CUDA regressions trigger a separate counter pass. The report links the raw exports and describes available evidence. Nsight counter permissions come from the GPU host.

Install the plotting extra and pass a hardware reference for the scalar FP32 peak to generate a GEMM roofline.

~~~bash
python -m pip install '.[plot]'
bandwidth_run=$(perf-scope run configs/bandwidth.toml --profile off)
perf-scope run configs/gemm.toml --profile all --bandwidth-run "$bandwidth_run" \
  --fp32-peak-gflops "$peak_gflops" --peak-source "$peak_reference"
~~~

The roofline uses useful work divided by Nsight DRAM bytes for intensity, and useful work divided by profiled duration for throughput. The bandwidth ceiling comes from the measured triad on the same GPU.

The softmax suite compares the fused kernel with a three-stage implementation. Both subtract the row maximum and validate against a double-precision CPU reference. If the gate confirms an unfusing slowdown, Nsight Systems traces a single invocation of each variant and the report shows recorded launch counts.

## ResNet-50 inference

On a Linux CUDA host, install matching PyTorch and torchvision wheels into the active environment.

~~~bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
perf-scope run configs/resnet50.toml --profile off
~~~

Eager and TorchInductor use the torchvision IMAGENET1K_V2 checkpoint and seeded GPU-resident FP32 inputs. The adapter checks Inductor outputs against eager, separately records the synchronized first call, and reports warmed device latency, images per second, and peak allocated memory. The first checkpoint load can download weights. Inductor shares the on-disk compiler cache by default. Add `--cold-cache` to the Inductor variant's arguments to isolate it per process.

## NVIDIA STREAM

Place the NVIDIA HPC Benchmarks 26.02.01 package directory on PATH on a supported Linux GPU host. The suite runs the FP32 TRIAD benchmark as an A/A comparison. Each trial checks process success, the reported GPU index and array size, and the bandwidth row. The report uses the median rate across repeated launcher calls and retains their original logs.

~~~bash
perf-scope run configs/stream.toml --profile off
~~~

[Package and hardware support](https://docs.nvidia.com/nvidia-hpc-benchmarks/index.html#system-support) · [NVIDIA STREAM arguments](https://docs.nvidia.com/nvidia-hpc-benchmarks/STREAM_Benchmark.html)

## Budgeted regression evaluation

Capture the seeded transpose and softmax changes and their A/A controls on an exclusive GPU. The campaign measures an observation run and a longer independent reference run, then replays the budgeted policies. Set the replay budget from pilot subprocess costs.

~~~bash
perf-scope campaign configs/regressions.toml --budget-s 60 \
  --expected-causes configs/regression-causes.json
~~~

The GPU experiment workflow runs the same command after GPU correctness tests on a self-hosted Linux runner labeled `nvidia-gpu`. It uploads run artifacts and can be dispatched from `main`.

Replay three policies using independently recorded reference and observation runs from the same suite and GPU. Use more pairs in the reference run for labels. The observation run provides disjoint screening and confirmation windows and can capture profile costs with `--profile all`.

~~~bash
perf-scope evaluate "$reference_run" "$observation_run" --budget-s "$budget_seconds" \
  --output runs/evaluation.json
~~~

Exhaustive testing confirms every case. The fixed-budget policy confirms cases in seeded random order. The adaptive policy spends up to 25% of its budget on screening and ranks cases by declared priority, observed slowdown, and screening variability. All three use the same fixed-size confirmation window and charge recorded profiler time after detections. The JSON reports recall, false positives, detection time, diagnosis accuracy, and replayed subprocess wall time.

## Local checks

~~~bash
python -m pip install '.[dev]'
ruff check .
python -m unittest discover -s tests -v
python -m build
~~~

CPU fixtures exercise the CLI, reports, and replay logic. GPU correctness, counter collection, and performance measurements require a CUDA runner.
