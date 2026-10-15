"""Process isolation, bounded concurrency, immutable run snapshots."""
import json
import os
import subprocess
import sys
import threading
import uuid

from .algorithms import REGISTRY, validate_settings
from .models import RunRequest
from .scoring import Scorer, graph_arrays
from .storage import now


class Jobs:
    def __init__(self, store):
        self.store = store
        self.processes = {}
        self.lock = threading.RLock()
        for record in self.store.list("run"):
            if record["status"] == "running":
                record.update(status="interrupted", error="The application closed before this run completed.", finished_at=now())
                self.store.put("run", record)

    def start(self, request: RunRequest):
        with self.lock:
            validate_settings(request.algorithm, request.settings)
            dataset = self.store.get("dataset", request.dataset_id)
            graph_arrays(request.graph, dataset["columns"])
            data = self.store.load_data(dataset["id"])
            if data.shape[0] <= data.shape[1]:
                raise ValueError("Algorithm runs currently require more observations than variables.")
            Scorer(data, dataset["columns"], request.settings.center).score(request.graph)
            if self.store.get("dataset", request.dataset_id).get("deleted_at"):
                raise ValueError("Restore this sample before starting a new run.")
            if request.experiment_id:
                self.store.get("workspace", request.experiment_id)
            self.list()
            if any(process.poll() is None for process in self.processes.values()):
                raise ValueError("One run is already active. Stop it or wait for it to finish.")
            key = uuid.uuid4().hex
            folder = self.store.root / "runs" / key
            folder.mkdir()
            snapshot = request.model_dump()
            if request.settings.fixed_partition:
                snapshot["settings"]["refine_partition"] = False
                snapshot["settings"]["color_probability"] = 0.0
                snapshot["settings"]["moves"] = [move for move in snapshot["settings"]["moves"] if move != "change_color"]
            if request.algorithm == "mcmc" and request.settings.mode == "map":
                snapshot["settings"]["refine_partition"] = False
            if request.algorithm == "greedy_dp":
                snapshot["settings"]["moves"] = [move for move in snapshot["settings"]["moves"] if move != "change_color"]
            root = os.path.dirname(os.path.dirname(__file__))
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip() or "unknown"
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip())
            record = {"id": key, "experiment_id": request.experiment_id, "name": REGISTRY[request.algorithm]["name"], "status": "running",
                      "created_at": now(), "request": snapshot,
                      "dataset": {field: dataset[field] for field in ("id", "name", "columns", "n", "p", "sha256")},
                      "provenance": {"dataset_sha256": dataset["sha256"], "code_revision": revision, "working_tree_modified": dirty, "app_version": "0.1.0"}}
            (folder / "request.json").write_text(json.dumps({**snapshot, "columns": dataset["columns"], "data_path": str(self.store.dataset_path(dataset["id"]))}))
            env = os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED=str(request.settings.seed))
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            with (folder / "worker.log").open("w") as output:
                process = subprocess.Popen([sys.executable, "-m", "daglab.worker", str(folder)], cwd=root, env=env, stdout=output, stderr=output)
            self.processes[key] = process
            return self.store.put("run", record)

    def get(self, key):
        with self.lock:
            record = self.store.get("run", key)
            folder = self.store.root / "runs" / key
            if (folder / "progress.json").exists():
                record["progress"] = json.loads((folder / "progress.json").read_text())
            if record["status"] == "running":
                process = self.processes.get(key)
                if process is not None and process.poll() is not None:
                    if process.returncode == 0 and (folder / "result.json").exists():
                        record.update(status="completed", result=json.loads((folder / "result.json").read_text()))
                    else:
                        error_path = folder / "error.json"
                        message = json.loads(error_path.read_text())["error"] if error_path.exists() else "The algorithm worker stopped unexpectedly."
                        record.update(status="failed", error=message)
                    record["finished_at"] = now()
                    self.store.put("run", record)
                    self.processes.pop(key, None)
            return record

    def list(self):
        return [self.get(record["id"]) for record in self.store.list("run")]

    def cancel(self, key):
        with self.lock:
            record = self.get(key)
            if record["status"] == "running":
                process = self.processes.pop(key, None)
                if process is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                record.update(status="cancelled", finished_at=now())
                self.store.put("run", record)
            return record

    def close(self):
        for key in list(self.processes):
            self.cancel(key)
