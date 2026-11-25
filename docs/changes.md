# Implementation changes

Each PR targets its dependency branch. The first PR targets the initial planning commit on `main`.

| Phase | CR | Branch | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| 1. Foundation | 01 | `cr/01-suite-config` | main | Install the package and validate a TOML suite through the CLI. CI checks invalid shapes, identifiers, variants, and measurement settings. |
| 2. Measurement | 02 | `cr/02-paired-runner` | 01 | Execute a real subprocess adapter in randomized pairs. Record provenance, every sample, and timed-out or invalid attempts in per-run artifacts. |
| 3. CUDA and regression analysis | 03 | `cr/03-regression-gate` | 02 | Compute a seeded paired confidence interval, classify a slowdown, and write a report with evidence and an actionable CI exit status. |
| 3. CUDA and regression analysis | 04 | `cr/04-transpose` | 03 | Compile naïve and shared-memory transpose. Validate aligned and partial-tile inputs on CUDA, with a build-only check on hosted CI. |
| 3. CUDA and regression analysis | 05 | `cr/05-gemm-bandwidth` | 04 | Compile scalar FP32 GEMM, tiled GEMM, and cuBLAS in the same harness. Verify rectangular outputs and provide a device-memory streaming calibration case. |
| 4. Profiling and workloads | 06 | `cr/06-nsight-roofline` | 05 | Profile both variants after a confirmed slowdown. Parse counter exports with units and generate a GEMM roofline with explicit ceiling provenance. |
| 4. Profiling and workloads | 07 | `cr/07-resnet` | 06 | Compare ResNet-50 eager and Inductor with matching weights and inputs. Record synchronized first-call and warm timing with output checks. |
| 4. Profiling and workloads | 08 | `cr/08-stream` | 07 | Import validated GPU TRIAD measurements through the common runner and preserve upstream logs. |
| 5. Seeded regressions | 09 | `cr/09-softmax-diagnosis` | 08 | Compile stable multi-pass and fused softmax. Validate partial rows and collect launch evidence for an injected unfusing regression. |
| 6. Evaluation | 10 | `cr/10-budgeted-scheduling` | 09 | Compare exhaustive, fixed-budget, and adaptive policies with a held-out confirmation window. Account for measurement and profiling costs. |
| 6. Evaluation | 11 | `cr/11-experiment-suites` | 10 | Provide configured A/A and seeded regression runs, a controlled GPU CI entry point, and a reproducible evaluation command. |
| 7. Reproduction | 12 | `cr/12-reproduction` | 11 | Verify clean installation and report generation from documented commands. Publish measured results when GPU runs are available. |

Run CPU tests, lint, and available build and GPU checks for each change. Verify each PR's base, head, and single commit before starting its dependent branch.
