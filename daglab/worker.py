"""One invocation per algorithm run. Communication uses atomic JSON files."""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .algorithms import execute
from .models import Graph, RunSettings


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(path)


def main():
    folder = Path(sys.argv[1])
    request = json.loads((folder / "request.json").read_text())
    started = time.monotonic()
    trace = []
    last_write = -1.0
    last_event = {}

    def progress(event):
        nonlocal last_write, trace, last_event
        last_event = event
        elapsed = time.monotonic() - started
        trace.append({"iteration": event["iteration"], "score": event["score"], "best_score": event["best_score"]})
        if len(trace) > 2000:
            trace = trace[::2]
        if elapsed-last_write >= 0.25 or event["iteration"] == 0:
            atomic_json(folder / "progress.json", {**event, "phase": "Searching", "elapsed": elapsed, "trace": trace})
            last_write = elapsed

    try:
        atomic_json(folder / "progress.json", {"phase": "Preparing numerical kernels", "iteration": 0, "elapsed": 0, "trace": []})
        data = np.load(request["data_path"], allow_pickle=False)
        result, history = execute(request["algorithm"], data, request["columns"],
            Graph.model_validate(request["graph"]), RunSettings.model_validate(request["settings"]), progress)
        if history is not None:
            np.save(folder / "history.npy", history, allow_pickle=False)
            result["history_available"] = True
        result["elapsed"] = time.monotonic() - started
        result["trace"] = trace
        atomic_json(folder / "progress.json", {**last_event, "phase": "Completed", "elapsed": result["elapsed"], "trace": trace})
        atomic_json(folder / "result.json", result)
    except Exception as exc:
        atomic_json(folder / "error.json", {"error": str(exc), "type": type(exc).__name__})
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
