# Subprocess measurement protocol

The runner executes each variant's argument array from the working directory. It appends `--shape` (comma-separated dimensions), `--dtype`, `--seed`, `--warmups`, and `--iterations`. Each invocation measures one block with one device and one interpretation of the case's primary metric.

An adapter that completed measurement writes one JSON object to standard output and exits successfully. The object contains `device` (the physical GPU identifier), `correct` (the output validation result), and `metrics` (a map that includes the configured primary metric). Use `latency_ms` for milliseconds per invocation, `bandwidth_gb_s` for decimal gigabytes per second, and `items_s` for items per second. Optional fields can describe the device, software, and timing scope. Write diagnostics to standard error.

Each run keeps the resolved suite and source revision in `manifest.json`, one row per attempt in `measurements.jsonl`, and raw output in `logs/`. Failed or timed-out invocations still have measurement rows and logs. A manifest left in `running` state identifies an interrupted run.
