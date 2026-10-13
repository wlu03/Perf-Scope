# Perf-Scope

Perf-Scope measures paired benchmark variants and estimates whether performance changed beyond a configured slowdown margin. A run records the suite, subprocess output, raw timings, and a static report.

## Install

~~~bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
~~~

## Usage

~~~bash
perf-scope validate suite.toml
run_dir=$(perf-scope run suite.toml --fail-on-regression)
perf-scope compare "$run_dir"
~~~

A suite is a version 1 TOML file with cases that declare a shape, a metric, and baseline and candidate argument arrays. Each subprocess writes a JSON result containing the device identifier, correctness status, and metrics. Run the command from the directory the argument arrays use for relative paths.

Each pair runs both variants in seeded order. Given at least eight pairs, the comparison bootstraps whole pairs to estimate a 95% percentile interval around median relative slowdown. The default margin is 5%, controlled by `slowdown_margin`. Invalid runs exit with status 1. With `--fail-on-regression`, a confirmed slowdown exits with status 3.

## CUDA benchmarks

Build on Linux with the CUDA toolkit and CMake 3.24 or later. The default build targets compute capability 8.0. Set `CMAKE_CUDA_ARCHITECTURES` for the target GPU.

~~~bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel
ctest --test-dir build/cuda -L gpu --output-on-failure
perf-scope run configs/transpose.toml
perf-scope run configs/gemm.toml
perf-scope run configs/bandwidth.toml
~~~

The two FP32 variants share their inputs and launch geometry. The tiled variant uses a padded shared-memory tile for coalesced reads and writes. CUDA events measure the average launch latency for each block. The executable validates the transpose and reports effective bandwidth from one logical matrix read and write.

GEMM compares naïve and tiled FP32 kernels with a cuBLAS reference using pedantic FP32 compute. GFLOP/s counts $2MNK$ operations. The reference checks up to 512 entries against a double-precision CPU calculation with absolute tolerance $10^{-3}$ and relative tolerance $10^{-4}$. The vector triad reads two arrays and writes one. Use a triad working set larger than the device's recorded L2 cache for bandwidth calibration.
