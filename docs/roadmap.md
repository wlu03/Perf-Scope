# Perf-Scope implementation plan

## First release

Deliver a reproducible CUDA experiment on one NVIDIA GPU. A single command should validate two implementations, measure their performance under the same conditions, and report whether a change exceeds a practical slowdown threshold. Confirmed regressions should trigger a targeted profile of both variants.

| Case | Baseline and comparator | Primary measurement |
| --- | --- | --- |
| FP32 transpose | Naïve global-memory transpose and shared-memory tiled transpose with coalesced accesses and padded tiles | Device latency and derived effective GB/s |
| FP32 GEMM | Naïve CUDA, shared-memory tiled CUDA, and a separate cuBLAS reference using matching FP32 compute | Device latency and derived GFLOP/s |

Use an aligned shape and a rectangular shape with partial tiles for each operation. Select a transpose size with a working set larger than the target GPU's L2 cache for the streaming-bandwidth comparison. Check that transpose preserves every value. Verify GEMM against a higher-precision reference with predeclared tolerances. Match shapes, layouts, input seeds, and FP32 math settings across variants. Record the cuBLAS compute mode and disable TF32 for the scalar FP32 comparison. [cuBLAS compute modes](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)

For the first regression demonstration, use the tiled transpose as the baseline and the naïve implementation as the candidate. Choose the demonstration shape during the pilot and keep an identical-implementation A/A control. Fix the slowdown threshold before the final run.

## Build order

| Week | Deliverable | Acceptance criterion |
| --- | --- | --- |
| 1 | Transpose through the complete runner | The fixture produces a report on CPU. Both CUDA variants pass exact checks on aligned and partial-tile shapes. The GPU runner exports raw event timings and records whether counter collection works. |
| 2 | GEMM and profiling | All three GEMM variants pass the same numerical checks. The report shows a scalar FP32 roofline point when DRAM counters are available, with the compute and bandwidth ceilings documented. |
| 3 | **0.1** paired regression gate | A fixed-size paired run records the seeded slowdown and the A/A control. The slowdown triggers a targeted profile. CPU CI exercises passing, regressing, inconclusive, and unavailable-profile fixtures. |
| 4 | **0.2** one model and one HPC benchmark | ResNet-50 passes output parity and reports steady-state latency, throughput, and first-call time. NVIDIA STREAM contributes one validated GPU TRIAD result on a supported runner. Both use the same result format as CUDA. |
| 5 | A second regression mechanism and scheduler | Add stable multi-pass and fused softmax. Use the multi-pass version as a seeded regression on a preselected shape. Implement deterministic scheduling and replay it against a small pilot dataset. |
| 6 | **0.3** adaptive testing evaluation | Compare policies on held-out runs under a recorded GPU-time budget. Publish the run recipe, findings, and limitations with the underlying results. |

This is a proposed schedule assuming access to a single GPU. An early GPU pilot determines the repetition count, profiler availability, and experiment budget. Layer normalization, warp-level reduction, attention, Triton, TensorRT, GROMACS, HPL and HPCG, multi-GPU scaling, Slurm, and a dashboard are later extensions. Add a pinned container once the first GPU recipe works and the host driver and toolkit requirements are known.

## Repository layout

~~~text
README.md
docs/
  roadmap.md
configs/
  fixture.toml
  transpose.toml
  gemm.toml
cuda/
  CMakeLists.txt
  src/                     # benchmark executable, timing, correctness, kernels
src/perf_scope/
  cli.py
  runner.py
  schema.py
  adapters/                # fixture and CUDA first
  analysis/                # comparison, roofline, and evidence rules
  report.py
tests/
  fixtures/                # timing records and profiler exports
pyproject.toml
~~~

The tree is a target for implementation. Start with Python 3.12, TOML configuration, a Python CLI, and one CMake-built CUDA executable that emits structured measurements. Use a fixture adapter and a CUDA adapter behind the same runner. Add PyTorch and STREAM adapters in 0.2. Write per-run JSON and JSONL, then add Parquet export and DuckDB queries when analysis needs to join multiple runs.

~~~mermaid
flowchart LR
    A["Suite configuration"] --> B["Resolve cases and check capabilities"]
    B --> C["Validate outputs and measure paired blocks"]
    C --> D["Compare against baseline"]
    D --> E["Profile confirmed regressions"]
    D --> F["Write run artifacts and report"]
    E --> F
~~~

### Configuration and artifacts

- A **case** defines the workload, shape or batch, precision and math mode, input seed, timing scope, and primary metric. A **variant** defines the implementation and software environment. The comparison declares which dimension may change. Other comparison inputs, including the physical GPU for a gate, must match.
- Store resolved config, schema version, source commit and working-tree state, executable or image identity, seeds, GPU model and device identity, driver, toolkit, library versions, compiler flags, and benchmark settings in `manifest.json`. Record clock policy, power limits, and available telemetry for interpreting outliers.
- Write `measurements.jsonl` with one row per timed block and variant, including case ID, pair ID, order, value, unit, iteration count, correctness, and run status. Store check failures, OOM, and unsupported cases with explicit reasons. Keep raw profiler exports in `profiles/` and structured decisions in `comparisons.json`. Generate a static `report.html` from those artifacts.
- The fixture suite exercises configuration, normalization, comparison, profile parsing, and report generation on macOS and CPU CI. Label fixtures in artifacts and reports. Use a scheduled or manually triggered GPU job on a dedicated runner for actual regression checks. Re-run a pinned baseline alongside the candidate for each gate. Give confirmed regressions a nonzero exit status in CI mode and upload the report on every outcome.

## Measurement and comparison

1. **Check the GPU host early.** Run a CUDA smoke workload and attempt to collect at least one counter with `ncu` on the target Linux host. Counter access depends on host policy. Record unavailable profiles as such and continue producing correctness and timing results. [NVIDIA counter permission guidance](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
2. **Define the timing scope.** Preallocate inputs and outputs for CUDA kernels, warm up the workload, record CUDA events on the measured stream, and synchronize the stop event before reading it. For short kernels, time a batch of launches and store per-invocation latency. Record setup, input transfers, and compilation separately where applicable. [CUDA timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers)
3. **Collect paired blocks.** Run both variants in each block with the same input, on the same GPU, in seeded randomized order. Prefer exclusive device access and record telemetry when available. Start the pilot with 20 pairs per case and use repeated A/A runs to set a fixed count for the release gate. The block pair is the unit of analysis.
4. **Decide from effect and uncertainty.** Take the median of per-pair slowdown ratios minus one. Use candidate latency divided by baseline latency, or baseline throughput divided by candidate throughput for fixed work. Bootstrap whole pairs for a per-case 95% interval. Start with a 5% practical margin and fix it before collecting the final comparison. Call a regression when the interval lies above the margin, within threshold when the upper end is at or below it, and inconclusive when it crosses the margin. Surface mismatched or invalid cases explicitly.
5. **Profile separately.** Trigger targeted Nsight Compute collection for both variants of a confirmed regression, with an explicit option to profile selected cases for a roofline report. Preserve tool versions, metrics, and collection settings. Replay and counter collection can affect execution, so use the event-timing pass for the regression decision. [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

### Roofline and diagnosis

Use a scalar FP32 DRAM roofline for GEMM in 0.1. Compute useful work as $2MNK$ FLOPs with FMA counted as two operations. Use the measured DRAM read and write bytes for that kernel as the denominator for arithmetic intensity, and the profiled duration for its roofline throughput. Use a documented scalar FP32 compute peak and a measured device-memory streaming bandwidth ceiling. Calibrate bandwidth with a simple CUDA streaming workload larger than L2. Record how each ceiling was obtained. [Nsight Compute roofline model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)

Show transpose as effective GB/s using two logical matrix transfers, one read and one write. Report DRAM traffic separately when counters are available. Compare store transaction efficiency for the transpose regression and report the supporting observations. For multi-pass softmax, use an Nsight Systems CUDA trace to check the change in launch count and timing. Produce a suspected cause with evidence, or an undetermined diagnosis when evidence is missing or ambiguous. [Nsight Systems CUDA tracing](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## Workload expansion

- **ResNet-50.** Use one pinned checkpoint and seeded synthetic inputs, with FP32 at batches 1, 8, and 32 as the starting matrix. Compare eager and `torch.compile` using the same evaluation mode, input tensors, and precision policy. Validate logits against eager before timing. Record synchronized first-call latency with the cache state specified, then report warmed inference latency and images per second. [PyTorch benchmarking utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) · [PyTorch compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)
- **NVIDIA STREAM.** Pin the upstream release and select the GPU TRIAD test with a fixed working set and repetition settings. Preserve the source logs and validation result. Use the package's documented CPU, GPU, and OS support to decide whether the selected host can run it. [NVIDIA HPC Benchmarks support](https://docs.nvidia.com/nvidia-hpc-benchmarks/index.html#system-support) · [NVIDIA STREAM documentation](https://docs.nvidia.com/nvidia-hpc-benchmarks/STREAM_Benchmark.html)

## Adaptive testing experiment

**Research question.** How much GPU time can a budgeted scheduler save at the same regression recall, and how often does targeted profiling identify the seeded cause?

1. **Build a small reference matrix.** Include unchanged A/A controls, the transpose implementation swap, and the extra softmax passes. Predeclare affected cases and expected mechanisms. Use independent high-repetition runs to establish which injected slowdowns exceed the chosen practical margin. Keep the pilot used for tuning separate from held-out evaluation runs.
2. **Compare three policies.** Run exhaustive fixed-sample testing, a policy with seeded random ordering at the same budget as the adaptive policy, and an adaptive policy that ranks cases using prior noise and change relevance. Let cheap screening observations determine where to spend confirmation runs. Use fresh fixed-size paired blocks for confirmation and the same profiling trigger for all policies.
3. **Count the full measurement cost.** Log allocated GPU time for setup, warmup, measurements, and profiler reruns. Choose the campaign budget using measured pilot costs and report detection recall, false-positive rate, time to detection, and missed detections. Include diagnosis accuracy and the count of undetermined diagnoses. Show the actual counts for each scenario and report which cases the budget left unmeasured.
4. **Publish a reproducible report.** Include the hardware and software versions, configs, seeds, raw measurements, and the commands needed to regenerate figures. State the limits of the single-GPU and seeded-regression evaluation.

## First implementation change

Build the transpose path and the fixture that exercises it. Create the package entry point, validate `configs/fixture.toml`, and produce artifact files and a report using recorded measurement rows. Add the CUDA executable with naïve and padded tiled implementations and point `configs/transpose.toml` at it. Check exact outputs on aligned and partial tiles, then use `perf-scope run configs/transpose.toml` to produce the first GPU report.
