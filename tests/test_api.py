import json
import io
import time

from fastapi.testclient import TestClient
import pytest

from daglab.api import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        yield client


def test_import_validation_and_no_header_support(client):
    bad = client.post("/api/datasets/import", files={"file": ("bad.csv", b"A,B\n1,2\n3,\n")})
    assert bad.status_code == 422
    assert "nonnumeric" in bad.json()["detail"]
    duplicate = client.post("/api/datasets/import", files={"file": ("bad.csv", b"A,A\n1,2\n3,4\n")})
    assert duplicate.status_code == 422
    good = client.post("/api/datasets/import?header=false", files={"file": ("sample.csv", b"1,2\n3,5\n6,4\n")})
    assert good.status_code == 200
    assert good.json()["n"] == 3
    assert good.json()["columns"][0]["label"] == "X1"


def test_experiment_roundtrip_contains_immutable_data(client):
    bootstrap = client.get("/api/bootstrap").json()
    workspace = bootstrap["workspace"]
    saved = client.post("/api/workspaces", json=workspace)
    assert saved.status_code == 200
    bundle = client.post("/api/workspaces/export", json=workspace).json()
    imported = client.post("/api/workspaces/import", files={"file": ("experiment.json", json.dumps(bundle))})
    assert imported.status_code == 200
    restored = imported.json()["workspace"]
    assert restored["candidate_graph"] == workspace["candidate_graph"]
    assert restored["dataset_id"] != workspace["dataset_id"]
    old_data = client.get(f"/api/datasets/{workspace['dataset_id']}/csv").text
    new_data = client.get(f"/api/datasets/{restored['dataset_id']}/csv").text
    assert old_data == new_data


def test_process_worker_persistence_snapshot_and_cancellation(client):
    workspace = client.get("/api/bootstrap").json()["workspace"]
    body = {"dataset_id": workspace["dataset_id"], "graph": workspace["candidate_graph"],
            "algorithm": "mcmc", "settings": {"num_iters": 60, "record_history": True, "fixed_partition": True}}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 202
    run = response.json()
    assert not run["request"]["settings"]["refine_partition"]
    assert run["request"]["settings"]["color_probability"] == 0
    body["graph"]["edges"] = []
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run['id']}").json()
        if run["status"] != "running":
            break
        time.sleep(0.1)
    assert run["status"] == "completed", run
    assert len(run["request"]["graph"]["edges"]) == 2
    assert client.get(f"/api/runs/{run['id']}/history").status_code == 200
    assert client.get(f"/api/runs/{run['id']}/export").json()["provenance"]["dataset_sha256"]
    body["settings"] = {"num_iters": 100_000_000}
    running = client.post("/api/runs", json=body).json()
    assert client.post("/api/runs", json=body).status_code == 422
    stopped = client.post(f"/api/runs/{running['id']}/cancel").json()
    assert stopped["status"] == "cancelled"


def test_cycle_and_cross_origin_writes_are_rejected(client):
    workspace = client.get("/api/bootstrap").json()["workspace"]
    graph = workspace["candidate_graph"]
    graph["edges"].append({"id": "cycle", "source": "x2", "target": "x1"})
    response = client.post("/api/score", json={"dataset_id": workspace["dataset_id"], "graph": graph})
    assert response.status_code == 422
    assert client.post("/api/workspaces", json=workspace, headers={"origin": "https://example.com"}).status_code == 403


def test_saved_workspace_survives_app_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as first:
        workspace = first.get("/api/bootstrap").json()["workspace"]
        record = first.post("/api/workspaces", json=workspace).json()
    with TestClient(create_app(tmp_path)) as second:
        assert second.get("/api/workspaces").json()[0]["id"] == record["id"]


def test_malformed_experiment_cannot_create_an_orphan_dataset(client):
    workspace = client.get("/api/bootstrap").json()["workspace"]
    bundle = client.post("/api/workspaces/export", json=workspace).json()
    bundle["dataset"]["metadata"]["columns"][0]["id"] = "unrelated"
    before = len(client.get("/api/datasets").json())
    response = client.post("/api/workspaces/import", files={"file": ("bad.json", json.dumps(bundle))})
    assert response.status_code == 422
    assert len(client.get("/api/datasets").json()) == before


def test_sample_deletion_is_reversible_and_preserves_saved_experiments(client):
    initial = client.get("/api/bootstrap").json()
    workspace = initial["workspace"]
    key = workspace["dataset_id"]
    saved = client.post("/api/workspaces", json=workspace).json()
    original_csv = client.get(f"/api/datasets/{key}/csv").text
    deleted = client.post(f"/api/datasets/{key}/delete")
    assert deleted.status_code == 200 and deleted.json()["deleted_at"]
    assert client.get("/api/datasets").json() == []
    assert len(client.get("/api/datasets?include_deleted=true").json()) == 1
    empty = client.get("/api/bootstrap").json()
    assert empty["datasets"] == [] and empty["workspace"]["dataset_id"] is None
    assert empty["experiments"][0]["id"] == saved["id"]
    assert client.post("/api/workspaces/export", json=workspace).status_code == 200
    score_request = {"dataset_id": key, "graph": workspace["candidate_graph"]}
    assert client.post("/api/score", json=score_request).status_code == 422
    assert client.post(f"/api/datasets/{key}/restore").status_code == 200
    assert len(client.get("/api/datasets").json()) == 1
    assert client.get(f"/api/datasets/{key}/csv").text == original_csv
    assert client.post("/api/score", json=score_request).status_code == 200


def test_cannot_delete_sample_used_by_active_worker(client):
    workspace = client.get("/api/bootstrap").json()["workspace"]
    key = workspace["dataset_id"]
    body = {"dataset_id": key, "graph": workspace["candidate_graph"], "algorithm": "mcmc",
            "settings": {"num_iters": 100_000_000}}
    run = client.post("/api/runs", json=body).json()
    assert client.post(f"/api/datasets/{key}/delete").status_code == 409
    client.post(f"/api/runs/{run['id']}/cancel")
    assert client.post(f"/api/datasets/{key}/delete").status_code == 200
    assert client.post("/api/runs", json=body).status_code == 422


def test_algorithm_output_can_seed_next_algorithm_without_changing_old_run(client):
    import numpy as np
    workspace = client.get("/api/bootstrap").json()["workspace"]
    def wait_for_result(run):
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            record = client.get(f"/api/runs/{run['id']}").json()
            if record["status"] != "running":
                assert record["status"] == "completed", record
                return record
            time.sleep(0.1)
        pytest.fail("Algorithm worker did not finish.")
    first = wait_for_result(client.post("/api/runs", json={"dataset_id": workspace["dataset_id"],
        "graph": workspace["candidate_graph"], "algorithm": "greedy_color", "settings": {"num_starts": 1}}).json())
    second = wait_for_result(client.post("/api/runs", json={"dataset_id": workspace["dataset_id"],
        "graph": first["result"]["graph"], "algorithm": "mcmc",
        "settings": {"num_iters": 1, "start_from_candidate": True, "fixed_partition": True, "record_history": True}}).json())
    assert second["request"]["graph"] == first["result"]["graph"]
    history = np.load(io.BytesIO(client.get(f"/api/runs/{second['id']}/history").content), allow_pickle=False)
    assert history[0] == pytest.approx(first["result"]["score"])
    assert client.get(f"/api/runs/{first['id']}").json()["result"] == first["result"]
