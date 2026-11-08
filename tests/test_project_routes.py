"""Routes of the project workflow, with the heavy modules (torch models, preprocessing) replaced by
small fakes: process_folder copies its inputs, so the tests check where files go, not the inking."""
import io
import os
import shutil
import sys
import time
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fake_process_folder(input_folder, model_path, prompt, output_dir, progress_callback=None, **kw):
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    names = sorted(os.listdir(input_folder))
    for n in names:
        shutil.copy2(os.path.join(input_folder, n), os.path.join(output_dir, n))
        Image.new('L', (8, 8)).save(os.path.join(output_dir, 'comparisons', f'comparison_{n}'))
    open(os.path.join(output_dir, 'logs', 'processing_log_test.txt'), 'w').close()
    return {"successful": len(names), "failed": 0, "average_time": 0.1}


class _FakeAnalyzer:
    def __init__(self, distributions=None):
        self.distributions = distributions or {"mean": {"mean": 1, "std": 0, "min": 0, "max": 2,
                                                         "percentiles": [0, 0, 1, 0, 0]}}

    def analyze_dataset(self, folder):
        return self.distributions

    def save_analysis(self, path):
        open(path, 'wb').write(b'npy')

    @classmethod
    def load_analysis(cls, path):
        return cls()


def _install_stubs():
    ink = types.ModuleType("ink")
    ink.process_folder = _fake_process_folder
    ink.process_single_image = lambda *a, **k: None
    ink.run_diagnostics = lambda input_folder, output_dir, **k: [
        Image.new('L', (8, 8)).save(os.path.join(output_dir, f'patches_{i}.png'))
        for i, _ in enumerate(os.listdir(input_folder), 1)] is not None
    models = types.ModuleType("models")
    models._MODELS_CACHE_DIR = "."
    pre = types.ModuleType("preprocessing")
    pre.DatasetAnalyzer = _FakeAnalyzer
    # drawings darker than 100 "need" an adjustment; the fake adjustment inverts them
    pre.check_image_quality = lambda img, stats: {
        "recommendations": ["brighten"] if img.convert('L').getpixel((0, 0)) < 100 else []}
    pre.apply_recommended_adjustments = lambda img, stats, verbose=False: Image.eval(img, lambda v: 255 - v)
    sys.modules.update(ink=ink, models=models, preprocessing=pre)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    _install_stubs()
    os.environ["PYPOTTERY_DISABLE_AUTO_SHUTDOWN"] = "1"
    sys.modules.pop("app", None)
    monkeypatch.chdir(ROOT)
    import app as A
    monkeypatch.setattr(A, "project_manager", A.ProjectManager(projects_root=str(tmp_path / "projects")))
    monkeypatch.setattr(A, "RUNS_FOLDER", str(tmp_path / "runs"))
    monkeypatch.setattr(A, "_is_sdturbo_cached", lambda: True)
    monkeypatch.setattr(A, "MODELS", {"Test Model": {"filename": "test.pkl", "prompt": "p"}})
    monkeypatch.setattr(A, "MODELS_DIR", str(tmp_path))
    (tmp_path / "test.pkl").write_bytes(b"x")
    A.app.config['TESTING'] = True
    c = A.app.test_client()
    c.A = A
    c.tmp = tmp_path
    return c


def png(shade, size=(40, 30)):
    buf = io.BytesIO()
    Image.new('L', size, shade).save(buf, 'PNG')
    buf.seek(0)
    return buf


def new_project(c, *drawings):
    pid = c.post('/api/projects', json={'name': 'T'}).get_json()['project']['project_id']
    if drawings:
        files = [(png(shade), name) for name, shade in drawings]
        r = c.post(f'/api/projects/{pid}/drawings', data={'files': files}, content_type='multipart/form-data')
        assert r.status_code == 200
    return pid


def wait(c, session_id, timeout=10):
    q = c.A.progress_queues[session_id]
    end = time.time() + timeout
    while time.time() < end:
        msg = q.get(timeout=timeout)
        if msg.get('completed') or msg.get('error'):
            return msg
    raise AssertionError("run did not finish")


LOCAL = {'model_name': 'Test Model', 'patch_size': 512, 'overlap': 64, 'contrast_scale': 1.0, 'upscale': 1}


def test_runs_require_a_project_and_library_names(client):
    c = client
    for url in ('/api/process-images', '/api/diagnostics', '/api/preprocess-images', '/api/cloud/start'):
        r = c.post(url, json={**LOCAL, 'drawings': ['a.png']})
        assert r.status_code == 400 and 'project' in r.get_json()['error'].lower()
    pid = new_project(c, ('a.png', 200))
    for bad in (['../project.json'], ['missing.png'], [], 'a.png'):
        r = c.post('/api/process-images', json={**LOCAL, 'project_id': pid, 'drawings': bad})
        assert r.status_code == 400
    assert not list((c.tmp / "runs").glob('*/input/*')) if (c.tmp / "runs").exists() else True


def test_import_report_and_listing(client):
    c = client
    pid = new_project(c, ('a.png', 200))
    files = [(png(200), 'same-content.png'), (png(10), 'a.png'), (io.BytesIO(b'hi'), 'notes.txt')]
    report = c.post(f'/api/projects/{pid}/drawings', data={'files': files},
                    content_type='multipart/form-data').get_json()['report']
    assert [d['name'] for d in report['duplicate']] == ['a.png']
    assert [d['name'] for d in report['renamed']] == ['a_2.png']
    assert len(report['skipped']) == 1
    names = [d['name'] for d in c.get(f'/api/projects/{pid}/drawings').get_json()['drawings']]
    assert names == ['a.png', 'a_2.png']
    thumb = c.get(f'/api/projects/{pid}/thumb/original/a.png')
    assert thumb.status_code == 200 and thumb.mimetype == 'image/png'
    assert c.get(f'/api/projects/{pid}/thumb/original/../project.json').status_code == 404


def test_local_run_places_results_per_settings_and_version(client):
    c = client
    pid = new_project(c, ('a.png', 200), ('b.png', 30))
    base = {**LOCAL, 'project_id': pid, 'drawings': ['a.png', 'b.png']}

    msg = wait(c, c.post('/api/process-images', json=base).get_json()['session_id'])
    res = msg['results']
    assert res['successful'] == 2 and res['folders'] == ['local/Test-Model_p512_o64_c1_x1']
    out = c.A.project_manager.outputs_dir(pid, 'local', 'Test-Model_p512_o64_c1_x1')
    assert sorted(p.name for p in out.iterdir() if p.is_file()) == ['a.png', 'b.png']
    assert (out / 'comparisons' / 'comparison_a.png').is_file()
    assert (out.parent / 'logs' / 'processing_log_test.txt').is_file()
    for rel in res['comparison_images'] + res['processed_images']:
        assert c.get(f'/api/projects/{pid}/file/outputs/{rel}').status_code == 200
    assert not list((c.tmp / 'runs').iterdir())                     # scratch folder removed

    status = c.post(f'/api/projects/{pid}/status', json={**LOCAL, 'mode': 'local'}).get_json()['drawings']
    assert {n: s['state'] for n, s in status.items()} == {'a.png': 'same', 'b.png': 'same'}
    status = c.post(f'/api/projects/{pid}/status', json={**LOCAL, 'contrast_scale': 1.5, 'mode': 'local'}).get_json()
    assert {s['state'] for s in status['drawings'].values()} == {'other'}

    # preprocess b (dark -> adjusted), a stays original; the next run splits into two folders
    stats = c.post('/api/calculate-statistics', json={'project_id': pid, 'drawings': ['a.png'], 'name': 'mine'})
    assert stats.get_json()['summary']['statistics_file'] == 'mine.npy'
    pre = c.post('/api/preprocess-images', json={'project_id': pid, 'drawings': ['a.png', 'b.png'],
                                                 'stats_source': 'project', 'stats_name': 'mine.npy'})
    done = wait(c, pre.get_json()['session_id'])['results']
    assert done['adjusted_names'] == ['b.png'] and done['unchanged'] == ['a.png']

    status = c.post(f'/api/projects/{pid}/status', json={**LOCAL, 'mode': 'local'}).get_json()['drawings']
    # b was inked from its original: with the new preprocessed version it counts as "other settings"
    assert status['a.png']['state'] == 'same'
    assert (status['b.png']['state'], status['b.png']['version']) == ('other', 'preprocessed')
    res = wait(c, c.post('/api/process-images', json={**base, 'drawings': ['b.png']}).get_json()['session_id'])['results']
    assert res['folders'] == ['local/Test-Model_p512_o64_c1_x1_pre']
    # the original-input result of b is still there
    assert (out / 'b.png').is_file()
    with Image.open(c.A.project_manager.outputs_dir(pid, 'local', 'Test-Model_p512_o64_c1_x1_pre') / 'b.png') as im:
        assert im.convert('L').getpixel((0, 0)) == 255 - 30          # inked from the preprocessed version

    res = wait(c, c.post('/api/process-images', json={**base, 'drawings': ['b.png'],
                                                      'use_preprocessed': False}).get_json()['session_id'])['results']
    assert res['folders'] == ['local/Test-Model_p512_o64_c1_x1']


def test_file_route_stays_inside_the_project(client):
    c = client
    pid = new_project(c, ('a.png', 200))
    for bad in ('outputs/../../project.json', 'outputs/..%2F..%2Fproject.json', 'secrets/x', 'original/../../project.json'):
        area, rel = bad.split('/', 1)
        assert c.get(f'/api/projects/{pid}/file/{area}/{rel}').status_code == 404
    assert c.get(f'/api/projects/{pid}/file/original/a.png').status_code == 200


def test_diagnostics_limit_and_output(client):
    c = client
    pid = new_project(c, *[(f'{i}.png', 100 + i) for i in range(6)])
    r = c.post('/api/diagnostics', json={**LOCAL, 'project_id': pid, 'drawings': [f'{i}.png' for i in range(6)]})
    assert r.status_code == 400
    r = c.post('/api/diagnostics', json={**LOCAL, 'project_id': pid, 'drawings': ['0.png', '1.png']})
    res = wait(c, r.get_json()['session_id'])['results']
    assert len(res['diagnostic_files']) == 2
    assert c.get(f"/api/projects/{pid}/file/diagnostics/{res['diagnostic_files'][0]}").status_code == 200


def test_cloud_results_are_recorded_without_the_key(client, monkeypatch):
    c = client
    pid = new_project(c, ('a.png', 200), ('b.png', 30))

    def fake_run_batch(api_key, input_folder, output_dir, variants=1, **kw):
        os.makedirs(output_dir, exist_ok=True)
        items = []
        for f in sorted(os.listdir(input_folder)):
            if f == 'b.png':
                continue                                              # b fails
            for v in range(1, variants + 1):
                name = f"{Path(f).stem}_v{v}.png"
                shutil.copy2(os.path.join(input_folder, f), os.path.join(output_dir, name))
                items.append({"file": name, "source": f, "variant": v, "seconds": 2.0, "cost_usd": 0.05})
        return {"successful": len(items), "failed": 1, "total": 2 * variants, "items": items,
                "cost_usd": 0.05 * len(items), "prompt_hash": "abc", "stopped": None, "errors": [],
                "processed_images": [i['file'] for i in items], "output_dir": output_dir, "log_file": ""}

    monkeypatch.setattr(c.A.cloud, "run_batch", fake_run_batch)
    r = c.post('/api/cloud/start', json={'project_id': pid, 'drawings': ['a.png', 'b.png'], 'api_key': 'sk-secret',
                                         'consent': True, 'model': 'openai/gpt-image-2', 'style': 'fine',
                                         'variants': 2, 'reference': 'none'})
    res = wait(c, r.get_json()['session_id'])['results']
    assert res['folders'] == ['cloud/gpt-image-2_fine_v2']
    for item in res['items']:
        assert c.get(f"/api/projects/{pid}/file/outputs/{item['output_path']}").status_code == 200
        assert c.get(f"/api/projects/{pid}/file/{item['input_path']}").status_code == 200
    project = c.A.project_manager.get_project(pid)
    run = project['runs'][-1]
    assert [(i['source'], i['status'], len(i['outputs'])) for i in run['items']] == [('a.png', 'ok', 2), ('b.png', 'failed', 0)]
    assert 'sk-secret' not in (c.A.project_manager.get_project_path(pid) / 'project.json').read_text(encoding='utf-8')
