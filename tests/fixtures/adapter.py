import argparse
import json
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--shape", required=True)
parser.add_argument("--dtype", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--warmups", type=int, required=True)
parser.add_argument("--iterations", type=int, required=True)
args = parser.parse_args()

if args.mode == "timeout":
    print("timeout fixture started", file=sys.stderr, flush=True)
    time.sleep(30)
elif args.mode == "garbled":
    print("missing JSON")
else:
    print(
        json.dumps(
            {
                "device": "fixture-device",
                "correct": args.mode != "incorrect",
                "metrics": {"latency_ms": 1.8 if args.mode == "slow" else 1.5},
                "args": vars(args),
            }
        )
    )
