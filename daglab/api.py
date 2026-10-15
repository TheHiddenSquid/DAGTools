import csv
import hashlib
import io
import json
import os
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .algorithms import REGISTRY
from .generation import circular_layout, random_graph, random_partition
from .jobs import Jobs
from .models import Column, GenerateRequest, Graph, RunRequest, ScoreRequest, Workspace, DeleteRequest, RunAssignmentRequest, RandomGraphRequest, RandomPartitionRequest, demo_graph
from .scoring import Scorer, graph_arrays, simulate
from .storage import Store, now

ROOT = Path(__file__).resolve().parent.parent


def create_app(data_dir=None):
    store = Store(Path(data_dir or os.environ.get("DAGLAB_DATA_DIR", ROOT / ".daglab")))
    jobs = Jobs(store)
    scorers = OrderedDict()
    score_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title="DAG Lab", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.jobs = store, jobs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        # Localhost is not a substitute for an origin check on mutating APIs.
        from urllib.parse import urlparse
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
                return JSONResponse({"detail": "Requests must originate from the local application."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc.args[0])}, status_code=404)

    def get_scorer(key, center):
        if store.get("dataset", key).get("deleted_at"):
            raise ValueError("Restore this sample before scoring a model with it.")
        cache_key = key, center
        if cache_key not in scorers:
            dataset = store.get("dataset", key)
            scorers[cache_key] = Scorer(store.load_data(key), dataset["columns"], center)
            if len(scorers) > 4:
                scorers.popitem(last=False)
        scorers.move_to_end(cache_key)
        return scorers[cache_key]

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "local_only": True}

    @app.get("/api/algorithms")
    def algorithms():
        return list(REGISTRY.values())

    @app.get("/api/bootstrap")
    def bootstrap():
        datasets = store.list("dataset")
        if not datasets and not store.list("setting") and not store.list("dataset", include_deleted=True):
            graph = demo_graph()
            data, columns = simulate(graph, 1000, 42)
            dataset = store.add_dataset(data, columns, "Five-variable demo", {"type": "generated", "seed": 42, "graph": graph.model_dump()})
            datasets = [dataset]
        store.put("setting", {"id": "demo_initialized"})
        dataset = datasets[0] if datasets else None
        source = (dataset["provenance"].get("graph") if dataset else None) or demo_graph().model_dump()
        source_graph = Graph.model_validate(source)
        if dataset and dataset["provenance"].get("graph"):
            candidate = source_graph.model_copy(deep=True)
            # A useful starting point: a partial structure rather than the truth.
            candidate.edges = candidate.edges[:2]
        elif dataset:
            candidate = Graph(nodes=[{**c, "group": str(i+1), "x": 60 + (i % 4) * 200, "y": 80 + (i // 4) * 130}
                                     for i, c in enumerate(dataset["columns"])])
        else:
            candidate = source_graph.model_copy(deep=True)
            candidate.edges = []
        if dataset and not dataset["provenance"].get("graph"):
            candidate = circular_layout(candidate)
        return {"datasets": datasets, "workspace": Workspace(name="Structure discovery", dataset_id=dataset["id"] if dataset else None,
                    source_graph=source_graph, candidate_graph=candidate).model_dump(),
                "algorithms": algorithms(), "experiments": store.list("workspace"), "runs": jobs.list()}

    @app.get("/api/datasets")
    def datasets(include_deleted: bool = False):
        return store.list("dataset", include_deleted=include_deleted)

    @app.post("/api/datasets/{key}/delete")
    def delete_dataset(key: str):
        with jobs.lock:
            store.get("dataset", key)
            if any(run["status"] == "running" and run["request"]["dataset_id"] == key for run in jobs.list()):
                raise HTTPException(409, "Stop the active run before deleting its sample.")
            with score_lock:
                record = store.set_dataset_deleted(key, True)
                for cache_key in list(scorers):
                    if cache_key[0] == key:
                        scorers.pop(cache_key)
            return record

    @app.post("/api/datasets/{key}/restore")
    def restore_dataset(key: str):
        with jobs.lock:
            return store.set_dataset_deleted(key, False)

    @app.get("/api/datasets/{key}")
    def dataset(key: str):
        return store.get("dataset", key)

    @app.post("/api/datasets/generate")
    def generate(request: GenerateRequest):
        data, columns = simulate(request.graph, request.sample_size, request.seed)
        return store.add_dataset(data, columns, request.name, {"type": "generated", "seed": request.seed, "graph": request.graph.model_dump()})

    @app.post("/api/datasets/import")
    async def import_csv(file: UploadFile = File(), header: bool = Query(True)):
        raw = await file.read(50 * 1024 * 1024 + 1)
        return store.import_csv(raw, (file.filename or "Imported sample")[:120], header)

    @app.get("/api/datasets/{key}/csv")
    def dataset_csv(key: str):
        with jobs.lock:
            dataset = store.get("dataset", key)
            observations = store.load_data(key)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([column["label"] for column in dataset["columns"]])
        writer.writerows(observations)
        return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="daglab-sample.csv"'})

    @app.post("/api/graphs/random")
    def generate_graph(request: RandomGraphRequest):
        return random_graph(request).model_dump()

    @app.post("/api/graphs/random-partition")
    def generate_partition(request: RandomPartitionRequest):
        return random_partition(request).model_dump()

    @app.post("/api/score")
    def score(request: ScoreRequest):
        with score_lock:
            return get_scorer(request.dataset_id, request.center).score(request.graph, request.optimize_partition)

    @app.get("/api/runs")
    def runs(experiment_id: str | None = None, unassigned: bool = False):
        records = jobs.list()
        if unassigned:
            return [r for r in records if not r.get("experiment_id")]
        if experiment_id is not None:
            store.get("workspace", experiment_id)
            return [r for r in records if r.get("experiment_id") == experiment_id]
        return records

    @app.post("/api/runs", status_code=202)
    def run(request: RunRequest):
        return jobs.start(request)

    @app.get("/api/runs/{key}")
    def run_details(key: str):
        return jobs.get(key)

    @app.put("/api/runs/{key}/experiment")
    def assign_run(key: str, request: RunAssignmentRequest):
        with jobs.lock:
            store.get("workspace", request.experiment_id)
            record = jobs.get(key)
            if record.get("experiment_id"):
                raise HTTPException(409, "This run already belongs to an experiment.")
            if record["status"] == "running":
                raise HTTPException(409, "Wait for the run to finish before assigning it.")
            record["experiment_id"] = request.experiment_id
            return store.put("run", record)

    @app.post("/api/runs/{key}/cancel")
    def cancel(key: str):
        return jobs.cancel(key)

    @app.get("/api/runs/{key}/export")
    def run_export(key: str):
        return JSONResponse(jobs.get(key), headers={"Content-Disposition": 'attachment; filename="daglab-run.json"'})

    @app.get("/api/runs/{key}/history")
    def history(key: str):
        record = jobs.get(key)
        if not record.get("result", {}).get("history_available"):
            raise HTTPException(404, "No exact score history was recorded for this run.")
        return FileResponse(store.root / "runs" / key / "history.npy", filename="bic-history.npy", media_type="application/octet-stream")

    @app.get("/api/workspaces")
    def workspaces():
        return store.list("workspace")

    def save_workspace(workspace, key):
        with jobs.lock:
            if workspace.dataset_id:
                dataset = store.get("dataset", workspace.dataset_id)
                graph_arrays(workspace.candidate_graph, dataset["columns"])
            workspace = workspace.model_copy(update={"experiment_id": key})
            return store.put("workspace", {"id": key, "updated_at": now(), "workspace": workspace.model_dump()})

    @app.post("/api/workspaces")
    def save(workspace: Workspace):
        return save_workspace(workspace, uuid.uuid4().hex)

    @app.put("/api/workspaces/{key}")
    def update(key: str, workspace: Workspace):
        with jobs.lock:
            store.get("workspace", key)
            return save_workspace(workspace, key)

    def deletion_plan(kind, key):
        record = store.get(kind, key)
        runs = jobs.list()
        affected_runs = [r for r in runs if (r.get("experiment_id") == key if kind == "workspace" else r["request"]["dataset_id"] == key)]
        references = [w for w in store.list("workspace") if kind == "dataset" and w["workspace"]["dataset_id"] == key]
        plan = {"id": key, "kind": kind, "name": record["workspace"]["name"] if kind == "workspace" else record["name"],
                "run_ids": [r["id"] for r in affected_runs], "running": any(r["status"] == "running" for r in affected_runs),
                "experiments": [{"id": w["id"], "name": w["workspace"]["name"]} for w in references]}
        plan["token"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        return plan

    @app.get("/api/workspaces/{key}/deletion")
    def experiment_deletion(key: str):
        with jobs.lock:
            return deletion_plan("workspace", key)

    @app.get("/api/datasets/{key}/deletion")
    def sample_deletion(key: str):
        with jobs.lock:
            return deletion_plan("dataset", key)

    @app.delete("/api/workspaces/{key}")
    def purge_experiment(key: str, request: DeleteRequest):
        with jobs.lock:
            plan = deletion_plan("workspace", key)
            if request.preview_token != plan["token"]:
                raise HTTPException(409, "The affected records changed. Cancel and review the deletion again.")
            if plan["running"]:
                raise HTTPException(409, "Stop the experiment's active run before deleting it.")
            store.purge([("workspace", key)] + [("run", r) for r in plan["run_ids"]])
            return plan

    @app.delete("/api/datasets/{key}")
    def purge_sample(key: str, request: DeleteRequest):
        with jobs.lock, score_lock:
            plan = deletion_plan("dataset", key)
            if request.preview_token != plan["token"]:
                raise HTTPException(409, "The affected records changed. Cancel and review the deletion again.")
            if plan["running"]:
                raise HTTPException(409, "Stop the active run before permanently deleting its sample.")
            updates = [("setting", {"id": "demo_initialized"})]
            for reference in plan["experiments"]:
                record = store.get("workspace", reference["id"])
                record["workspace"]["dataset_id"] = None
                record["updated_at"] = now()
                updates.append(("workspace", record))
            store.purge([("dataset", key)] + [("run", r) for r in plan["run_ids"]], updates)
            for cache_key in list(scorers):
                if cache_key[0] == key:
                    scorers.pop(cache_key)
            return plan

    @app.post("/api/workspaces/export")
    def export_workspace(workspace: Workspace):
        payload = {"format": "daglab-experiment-v1", "workspace": workspace.model_dump()}
        if workspace.dataset_id:
            with jobs.lock:
                payload["dataset"] = {"metadata": store.get("dataset", workspace.dataset_id), "data": store.load_data(workspace.dataset_id).tolist()}
        return JSONResponse(payload, headers={"Content-Disposition": 'attachment; filename="daglab-experiment.json"'})

    @app.post("/api/workspaces/import")
    async def import_workspace(file: UploadFile = File()):
        raw = await file.read(50 * 1024 * 1024 + 1)
        if len(raw) > 50 * 1024 * 1024:
            raise ValueError("Experiment imports are limited to 50 MB.")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Choose a valid exported DAG Lab experiment JSON.")
        if not isinstance(payload, dict) or payload.get("format") != "daglab-experiment-v1":
            raise ValueError("Unsupported experiment format.")
        workspace = Workspace.model_validate(payload.get("workspace"))
        imported = payload.get("dataset")
        if imported:
            if not isinstance(imported, dict) or not isinstance(imported.get("metadata"), dict) or "data" not in imported:
                raise ValueError("The experiment contains an invalid dataset bundle.")
            metadata = imported["metadata"]
            if not isinstance(metadata.get("columns"), list) or not isinstance(metadata.get("provenance"), dict) or not isinstance(metadata.get("name"), str):
                raise ValueError("The experiment contains invalid dataset metadata.")
            array = np.asarray(imported["data"], dtype=float)
            columns = [Column.model_validate(column).model_dump() for column in metadata["columns"]]
            graph_arrays(workspace.candidate_graph, columns)
            dataset = store.add_dataset(array, columns, metadata["name"], metadata["provenance"])
            workspace.dataset_id = dataset["id"]
        return save_workspace(workspace, uuid.uuid4().hex)

    frontend = ROOT / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/")
        def not_built():
            return {"message": "Build the interface with npm run build in frontend/, or use the Vite development server."}
    return app
