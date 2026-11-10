"""Tests for poser.py against the viewer's own export (examples/mouse_social_analysis/).

Run with `python tests/test_poser.py` (or pytest). Needs numpy and pandas.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'api'))
import poser  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / 'examples' / 'mouse_social_analysis'
POSE = json.loads((DATA / 'mouse_social_pose.json').read_text())
CLUSTERS = json.loads((DATA / 'mouse_social_clusters.json').read_text())
MODEL = poser.Model.load(CLUSTERS)
VIEWER_LABELS = np.array([a['labels'] for a in CLUSTERS['animals']])


def test_reproduces_viewer_export():
    r = MODEL.apply(POSE)
    emb = np.array([a['embedding'] for a in CLUSTERS['animals']])
    assert np.abs(r.embedding - emb).max() < 1e-5          # export rounds to 6 significant digits
    assert (r.labels == VIEWER_LABELS).all()


def test_model_file_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        MODEL.save(Path(d) / 'm.json')
        spec = json.loads((Path(d) / 'm.json').read_text())
        assert spec['format'] == 'poser-model' and 'animals' not in spec
        assert (poser.Model.load(Path(d) / 'm.json').apply(POSE).labels == VIEWER_LABELS).all()


def test_units_joint_order_and_extra_joints():
    pose = copy.deepcopy(POSE)
    pose['space']['units'] = 'mm'
    sk = pose['skeletons']['mouse']
    J = len(sk['joints'])
    order = list(range(J))[::-1]                              # reversed joint order + one extra joint
    sk['joints'] = [sk['joints'][j] for j in order] + [{'name': 'implant', 'parent': None}]
    sk['bones'] = []
    for t in pose['clips'][0]['tracks'].values():
        t['positions'] = [[[v * 1000 for v in fr[j]] for j in order] + [[0, 0, 0]] for fr in t['positions']]
    assert (MODEL.apply(pose).labels == VIEWER_LABELS).all()


def test_missing_samples_are_interpolated():
    pose = copy.deepcopy(POSE)
    pos = pose['clips'][0]['tracks']['mouse-1']['positions']
    for f in range(200, 203):
        pos[f] = None
    pos[400][4] = None
    r = MODEL.apply(pose)
    assert (r.labels != VIEWER_LABELS).mean() < 0.01


def test_new_clip_from_part_of_the_session():
    # a non-looping excerpt is new data the model never saw as such; away from its ends the states agree
    pose = copy.deepcopy(POSE)
    clip = pose['clips'][0]
    clip.update(frame_count=500, loop=False)
    for t in clip['tracks'].values():
        t['positions'] = t['positions'][200:700]
    r = MODEL.apply(pose)
    assert (r.labels[:, 5:-5] == VIEWER_LABELS[:, 205:695]).mean() > 0.95


def test_rejects_mismatched_data():
    one = copy.deepcopy(POSE)
    one['animals'] = one['animals'][:1]
    for bad, msg in [(one, 'two animals'), ({**POSE, 'space': {**POSE['space'], 'dims': 2}}, '3D')]:
        try:
            MODEL.apply(bad)
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError('expected ValueError')
    raw = copy.deepcopy(CLUSTERS)
    raw['reduction']['method'] = 'none'
    try:
        poser.Model(raw).apply(POSE)
    except ValueError as e:
        assert 'raw features' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_tables_and_clusters_export():
    r = MODEL.apply(POSE)
    tb = r.time_budget()
    assert np.allclose(tb.drop(index='all').sum(axis=1), 100, atol=0.2)
    assert r.bout_stats()['bouts'].sum() == len(r.bout_list()) == sum(len(a['bouts']) for a in CLUSTERS['animals'])
    assert np.allclose(r.transitions().sum(axis=1), 1)
    ss = r.state_summary()
    assert abs(ss.loc['S1', 'speed (cm/s)'] - 4.600) < 0.01  # same as the analysis notebook
    out = r.to_clusters()
    assert out['format'] == 'poser-clusters'
    assert [b['state'] for b in out['animals'][0]['bouts']] == [b['state'] for b in CLUSTERS['animals'][0]['bouts']]


def _other_block_weight(feat):
    """Per-row weight of the other-animal block: weighted features / unweighted features."""
    nodes = feat['nodes']
    _, P = poser._tracks(POSE, 0, nodes, 'm')
    X = poser._features(P, feat, 3)
    X1 = poser._features(P, dict(feat, proximity_kernel=None), 3)
    d = X.shape[1] // 2
    big = np.abs(X1[:, d:]).argmax(1)                  # the largest unweighted entry of each row
    rows = np.arange(len(X))
    return X[rows, d + big] / X1[rows, d + big], P


def test_elliptical_field():
    base = dict(CLUSTERS['features'], align_others=False)
    k = dict(base['proximity_kernel'])
    nodes = base['nodes']
    ant, post = nodes.index('neck'), nodes.index('pelvis')
    for side, dist_dims in [(0.5, 2), (0.3, 3), (1.0, 3)]:
        feat = dict(base, distance_dims=dist_dims, proximity_kernel=dict(k, lateral_scale=side, dims=dist_dims))
        w, P = _other_block_weight(feat)
        cen = P.mean(axis=2)
        th = np.arctan2(P[:, :, ant, 2] - P[:, :, post, 2], P[:, :, ant, 0] - P[:, :, post, 0])
        expect = []
        for a in range(2):
            off = cen[1 - a] - cen[a]
            fw = off[:, 0] * np.cos(th[a]) + off[:, 2] * np.sin(th[a])
            lat = -off[:, 0] * np.sin(th[a]) + off[:, 2] * np.cos(th[a])
            dh = off[:, 1] if dist_dims == 3 else 0
            expect.append(np.exp(-0.5 * ((fw ** 2 + dh ** 2) / k['d0'] ** 2 + lat ** 2 / (side * k['d0']) ** 2)))
        assert np.allclose(w, np.concatenate(expect), atol=1e-9), side
    # side reach 1 with 2D distance is the old isotropic kernel, and the default for older files
    old = dict(base, proximity_kernel={x: v for x, v in k.items() if x != 'lateral_scale'})
    new = dict(base, distance_dims=2, proximity_kernel=dict(k, lateral_scale=1.0, dims=2))
    _, P = poser._tracks(POSE, 0, nodes, 'm')
    assert np.array_equal(poser._features(P, old, 3), poser._features(P, new, 3))


def test_others_scale():
    feat = dict(CLUSTERS['features'], align_others=False)
    _, P = poser._tracks(POSE, 0, feat['nodes'], 'm')
    X1, X3 = poser._features(P, feat, 3), poser._features(P, dict(feat, others_scale=3.0), 3)
    d = X1.shape[1] // 2
    assert np.array_equal(X1[:, :d], X3[:, :d])            # own pose untouched
    assert np.allclose(X3[:, d:], 3 * X1[:, d:])


CLEAN = {'swaps': True, 'outliers': {'window': 7, 'threshold': 5}, 'fill': 'body', 'smooth': {'method': 'gaussian', 'size': 1}}


def test_model_cleans_new_data_like_its_training_data():
    model = poser.Model(dict(CLUSTERS, preprocessing=CLEAN))
    cleaned = poser.clean_pose(POSE, CLEAN)
    assert cleaned['meta']['cleaned'] == CLEAN
    assert all(q is not None for t in cleaned['clips'][0]['tracks'].values() for fr in t['positions'] for q in fr)
    r = model.apply(POSE)                              # raw file: cleaned first
    assert (r.labels == MODEL.apply(cleaned).labels).all()
    assert (model.apply(cleaned).labels == r.labels).all()   # already cleaned: not cleaned twice
    assert r.to_clusters()['preprocessing'] == CLEAN


def test_clean_up_brings_noisy_group_closer_to_truth():
    # the four-mouse example is generated deterministically: replay the generator for the noise-free scene
    root = ROOT / 'examples'
    sys.path.insert(0, str(root / 'generators'))
    import generate_mouse_group_dataset as G
    rng = np.random.default_rng(20261109)
    paths, face_t, face_w, beh, _ = G.choreograph()
    flat = [G.body(paths[m], G.heading(paths[m], face_t[m], face_w[m]), beh[m], rng) for m in range(4)]
    lifted = [G.lift(f)[0] for f in flat]
    lifted[2] = G.lift(flat[2], G.back_bump(lifted[3], G.W(11.2, 12.6, 0.1)))[0]
    truth = np.stack(lifted)
    raw = poser.load_pose(root / 'mouse_group_blocks.json')
    names = [j['name'] for j in raw['skeletons']['mouse']['joints']]

    def errors(pose):
        _, P = poser._tracks(pose, 0, names, 'm')              # gaps filled linearly, like the viewer
        P = P[..., [0, 1, 2]]                                  # already Y up
        pos = np.sqrt((np.linalg.norm(P - truth, axis=-1) ** 2).mean())
        vel = np.sqrt((np.linalg.norm(np.diff(P, axis=1) - np.diff(truth, axis=1), axis=-1) ** 2).mean())
        return pos, vel
    (p0, v0), (p1, v1) = errors(raw), errors(poser.clean_pose(raw, CLEAN))
    assert p1 < 0.85 * p0 and v1 < 0.6 * v0, (p0, p1, v0, v1)


def test_analysis_notebook_runs_on_applied_result():
    # the result file is a normal poser-clusters export: the viewer's analysis notebook reads it
    r = MODEL.apply(POSE)
    with tempfile.TemporaryDirectory() as d:
        r.save(Path(d) / 'c.json')
        env = dict(os.environ, POSE_PATH=str(DATA / 'mouse_social_pose.json'), CLUSTERS_PATH=str(Path(d) / 'c.json'), MPLBACKEND='Agg')
        p = subprocess.run([sys.executable, str(HERE.parent / 'viewer' / 'analysis_notebook.py')], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
    print(f'{len(tests)} passed')
