"""
Launch large model training as detached Modal functions.
Results persist in volume regardless of local process lifetime.

Usage:
  .venv/bin/python run_large_modal.py launch
  .venv/bin/python run_large_modal.py check
"""

import sys
import subprocess
import time


def launch():
    """Deploy the app and trigger all runs as detached functions."""
    # First deploy so the app stays alive
    print("Deploying app...")
    result = subprocess.run(
        [".venv/bin/modal", "deploy", "modal_deploy.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return

    # Now trigger each run individually via modal shell
    conditions = ["j1", "j2", "j3", "j4", "j5"]
    seeds = [42, 43]

    for cond in conditions:
        for seed in seeds:
            print(f"Triggering {cond} seed={seed}...")
            cmd = [
                ".venv/bin/python", "-c",
                f"""
import modal
f = modal.Function.from_name("compression-truth-deploy", "train_one")
fc = f.spawn("{cond}", {seed}, "large", 5000)
print(f"Spawned: {{fc.object_id}}")
"""
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout.strip())
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr.strip()}")

    print(f"\nAll {len(conditions) * len(seeds)} runs spawned!")
    print("Results will be saved to Modal volume.")
    print("Check with: .venv/bin/python run_large_modal.py check")


def check():
    """Check results in volume."""
    result = subprocess.run(
        [".venv/bin/modal", "run", "modal_deploy.py::check"],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "launch":
        launch()
    elif sys.argv[1] == "check":
        check()
    else:
        print(f"Usage: {sys.argv[0]} [launch|check]")
