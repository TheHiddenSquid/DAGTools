import csv
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
from .jobs import Jobs
from .models import Column, GenerateRequest, Graph, RunRequest, ScoreRequest, Workspace, demo_graph
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
        if not datasets:
            graph = demo_graph()
            data, columns = simulate(graph, 1000, 42)
            dataset = store.add_dataset(data, columns, "Five-variable demo", {"type": "generated", "seed": 42, "graph": graph.model_dump()})
            datasets = [dataset]
        dataset = datasets[0]
        source = dataset["provenance"].get("graph") or demo_graph().model_dump()
        source_graph = Graph.model_validate(source)
        if dataset["provenance"].get("graph"):
            candidate = source_graph.model_copy(deep=True)
            # A useful starting point: a partial structure rather than the truth.
            candidate.edges = candidate.edges[:2]
        else:
            candidate = Graph(nodes=[{**c, "group": str(i+1), "x": 60 + (i % 4) * 200, "y": 80 + (i // 4) * 130}
                                     for i, c in enumerate(dataset["columns"])])
        return {"datasets": datasets, "workspace": Workspace(name="Structure discovery", dataset_id=dataset["id"],
                    source_graph=source_graph, candidate_graph=candidate).model_dump(),
                "algorithms": algorithms(), "experiments": store.list("workspace"), "runs": jobs.list()}

    @app.get("/api/datasets")
    def datasets():
        return store.list("dataset")

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
        dataset = store.get("dataset", key)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([column["label"] for column in dataset["columns"]])
        writer.writerows(store.load_data(key))
        return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="daglab-sample.csv"'})

    @app.post("/api/score")
    def score(request: ScoreRequest):
        with score_lock:
            return get_scorer(request.dataset_id, request.center).score(request.graph, request.optimize_partition)

    @app.get("/api/runs")
    def runs():
        return jobs.list()

    @app.post("/api/runs", status_code=202)
    def run(request: RunRequest):
        return jobs.start(request)

    @app.get("/api/runs/{key}")
    def run_details(key: str):
        return jobs.get(key)

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
        if workspace.dataset_id:
            dataset = store.get("dataset", workspace.dataset_id)
            graph_arrays(workspace.candidate_graph, dataset["columns"])
        return store.put("workspace", {"id": key, "updated_at": now(), "workspace": workspace.model_dump()})

    @app.post("/api/workspaces")
    def save(workspace: Workspace):
        return save_workspace(workspace, uuid.uuid4().hex)

    @app.put("/api/workspaces/{key}")
    def update(key: str, workspace: Workspace):
        store.get("workspace", key)
        return save_workspace(workspace, key)

    @app.post("/api/workspaces/export")
    def export_workspace(workspace: Workspace):
        payload = {"format": "daglab-experiment-v1", "workspace": workspace.model_dump()}
        if workspace.dataset_id:
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
