import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from daglab.api import create_app
from daglab.storage import Store


@pytest.fixture
def app_client(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        yield app, client


def save(client, workspace, name):
    return client.post('/api/workspaces', json={**workspace, 'name': name}).json()


def run(client, workspace, experiment_id=None, algorithm='ges', settings=None):
    result = client.post('/api/runs', json={'dataset_id': workspace['dataset_id'],
        'graph': workspace['candidate_graph'], 'algorithm':algorithm,
        'experiment_id':experiment_id, 'settings':settings or {}})
    assert result.status_code == 202, result.text
    record = result.json()
    if algorithm == 'mcmc': return record
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{record['id']}").json()
        if record['status'] != 'running':
            assert record['status'] == 'completed', record
            return record
        time.sleep(0.05)
    pytest.fail('GES worker did not finish')


def purge(client, kind, key, plan=None):
    plan = plan or client.get(f'/api/{kind}/{key}/deletion').json()
    return client.request('DELETE', f'/api/{kind}/{key}', json={'preview_token':plan['token']})


def test_separate_run_histories_and_legacy_assignment(app_client):
    app, client = app_client
    workspace = client.get('/api/bootstrap').json()['workspace']
    a, b = save(client, workspace, 'A'), save(client, workspace, 'B')
    first, second, legacy = run(client, workspace, a['id']), run(client, workspace, b['id']), run(client, workspace)
    assert [r['id'] for r in client.get('/api/runs', params={'experiment_id':a['id']}).json()] == [first['id']]
    assert [r['id'] for r in client.get('/api/runs', params={'experiment_id':b['id']}).json()] == [second['id']]
    assert [r['id'] for r in client.get('/api/runs?unassigned=true').json()] == [legacy['id']]
    assigned = client.put(f"/api/runs/{legacy['id']}/experiment", json={'experiment_id':a['id']})
    assert assigned.status_code == 200
    assert assigned.json()['result'] == legacy['result']
    assert len(client.get('/api/runs', params={'experiment_id':a['id']}).json()) == 2
    assert client.put(f"/api/runs/{second['id']}/experiment", json={'experiment_id':a['id']}).status_code == 409
    assert a['workspace']['experiment_id'] == a['id']
    assert run(client, workspace, a['id'])['request']['experiment_id'] == a['id']


def test_permanent_experiment_deletion_removes_only_owned_runs(app_client):
    app, client = app_client
    workspace = client.get('/api/bootstrap').json()['workspace']
    a, b = save(client, workspace, 'A'), save(client, workspace, 'B')
    first, second, legacy = run(client, workspace, a['id']), run(client, workspace, b['id']), run(client, workspace)
    folder = app.state.store.root/'runs'/first['id']
    assert folder.exists()
    response = purge(client, 'workspaces', a['id'])
    assert response.status_code == 200
    assert not folder.exists()
    assert client.get(f"/api/runs/{first['id']}").status_code == 404
    assert {r['id'] for r in client.get('/api/runs').json()} == {second['id'],legacy['id']}
    assert [w['id'] for w in client.get('/api/workspaces').json()] == [b['id']]
    assert client.get(f"/api/datasets/{workspace['dataset_id']}/csv").status_code == 200
    assert client.get('/api/runs', params={'experiment_id':a['id']}).status_code == 404
    assert client.post('/api/runs', json={'algorithm':'ges','experiment_id':a['id'],
        'dataset_id':workspace['dataset_id'],'graph':workspace['candidate_graph']}).status_code == 404


def test_sample_purge_detaches_experiments_and_removes_all_related_runs(app_client):
    app, client = app_client
    workspace = client.get('/api/bootstrap').json()['workspace']
    a, b = save(client, workspace, 'A'), save(client, workspace, 'B')
    first, second = run(client, workspace, a['id']), run(client, workspace, b['id'])
    sample_id = workspace['dataset_id']
    sample_file = app.state.store.dataset_path(sample_id)
    client.post(f'/api/datasets/{sample_id}/delete')
    plan = client.get(f'/api/datasets/{sample_id}/deletion').json()
    assert {w['id'] for w in plan['experiments']} == {a['id'],b['id']}
    assert len(plan['run_ids']) == 2
    assert purge(client, 'datasets', sample_id, plan).status_code == 200
    assert not sample_file.exists()
    assert client.get(f'/api/datasets/{sample_id}').status_code == 404
    assert client.get(f'/api/datasets/{sample_id}/csv').status_code == 404
    assert client.post(f'/api/datasets/{sample_id}/restore').status_code == 404
    assert client.get('/api/datasets?include_deleted=true').json() == []
    assert client.get('/api/runs').json() == []
    for record in client.get('/api/workspaces').json():
        assert record['workspace']['dataset_id'] is None
        assert record['workspace']['candidate_graph'] == workspace['candidate_graph']
        assert client.post('/api/workspaces/export', json=record['workspace']).status_code == 200
    assert client.get('/api/bootstrap').json()['datasets'] == []
    assert not (app.state.store.root/'runs'/first['id']).exists()
    assert not (app.state.store.root/'runs'/second['id']).exists()
    with TestClient(create_app(app.state.store.root)) as reopened:
        assert reopened.get('/api/bootstrap').json()['datasets'] == []


def test_active_run_protects_experiment_and_sample(app_client):
    app, client = app_client
    workspace = client.get('/api/bootstrap').json()['workspace']
    a = save(client, workspace, 'Active')
    record = run(client, workspace, a['id'], algorithm='mcmc', settings={'num_iters':100000000})
    assert purge(client, 'workspaces', a['id']).status_code == 409
    assert purge(client, 'datasets', workspace['dataset_id']).status_code == 409
    client.post(f"/api/runs/{record['id']}/cancel")
    assert purge(client, 'workspaces', a['id']).status_code == 200


def test_changed_deletion_dependencies_require_review(app_client):
    app, client = app_client
    workspace = client.get('/api/bootstrap').json()['workspace']
    a = save(client, workspace, 'Before')
    plan = client.get(f"/api/workspaces/{a['id']}/deletion").json()
    record = run(client, workspace, a['id'])
    assert purge(client, 'workspaces', a['id'], plan).status_code == 409
    assert client.get(f"/api/runs/{record['id']}").status_code == 200
    assert purge(client, 'workspaces', a['id']).status_code == 200


def test_deleting_unused_sample_keeps_unrelated_data_and_runs(app_client):
    app, client = app_client
    workspace = client.get('/api/bootstrap').json()['workspace']
    a = save(client, workspace, 'Keep')
    record = run(client, workspace, a['id'])
    other = client.post('/api/datasets/generate', json={'graph':workspace['source_graph'], 'sample_size':20}).json()
    assert purge(client,'datasets',other['id']).status_code == 200
    assert client.get(f"/api/runs/{record['id']}").status_code == 200
    assert client.get(f"/api/datasets/{workspace['dataset_id']}/csv").status_code == 200


def test_legacy_workspace_identity_and_purge_crash_recovery(tmp_path):
    store = Store(tmp_path)
    store.put('workspace', {'id':'old','workspace':{'name':'Old'}})
    assert store.get('workspace','old')['workspace']['experiment_id'] == 'old'
    data_id = 'a'*32
    store.put('dataset',{'id':data_id})
    original = tmp_path/'datasets'/f'{data_id}.npy'
    original.write_bytes(b'test observations')
    stage = tmp_path/'.purging'/'interrupted'
    stage.mkdir(parents=True)
    (stage/'manifest.json').write_text(json.dumps([['dataset',data_id]]))
    original.rename(stage/f'dataset-{data_id}')
    Store(tmp_path)
    assert original.read_bytes() == b'test observations'
    assert not stage.exists()
    stage.mkdir()
    (stage/'manifest.json').write_text(json.dumps([['dataset',data_id]]))
    original.rename(stage/f'dataset-{data_id}')
    with store.connect() as db:
        db.execute("DELETE FROM records WHERE kind='dataset' AND id=?",(data_id,))
    Store(tmp_path)
    assert not original.exists() and not stage.exists()


def test_failed_metadata_transaction_restores_staged_files(tmp_path):
    store = Store(tmp_path)
    key = 'b'*32
    store.put('dataset', {'id':key,'name':'Keep'})
    path = tmp_path/'datasets'/f'{key}.npy'
    path.write_bytes(b'keep this sample')
    with pytest.raises(ValueError):
        store.purge([('dataset',key)], [('setting',{'id':'invalid','value':float('nan')})])
    assert store.get('dataset',key)['name'] == 'Keep'
    assert path.read_bytes() == b'keep this sample'
    assert not list((tmp_path/'.purging').iterdir())
