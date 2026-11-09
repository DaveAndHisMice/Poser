"""Tests for etho.py against the viewer's own export (mouse_social_analysis/).

Run with `python api/test_etho.py` (or pytest). Needs numpy and pandas.
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
sys.path.insert(0, str(HERE))
import etho  # noqa: E402

DATA = HERE.parent / 'mouse_social_analysis'
POSE = json.loads((DATA / 'mouse_social_pose.json').read_text())
CLUSTERS = json.loads((DATA / 'mouse_social_clusters.json').read_text())
MODEL = etho.Model.load(CLUSTERS)
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
        assert spec['format'] == 'etho-model' and 'animals' not in spec
        assert (etho.Model.load(Path(d) / 'm.json').apply(POSE).labels == VIEWER_LABELS).all()


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
        etho.Model(raw).apply(POSE)
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
    assert out['format'] == 'etho-clusters'
    assert [b['state'] for b in out['animals'][0]['bouts']] == [b['state'] for b in CLUSTERS['animals'][0]['bouts']]


def _other_block_weight(feat):
    """Per-row weight of the other-animal block: weighted features / unweighted features."""
    nodes = feat['nodes']
    _, P = etho._tracks(POSE, 0, nodes, 'm')
    X = etho._features(P, feat, 3)
    X1 = etho._features(P, dict(feat, proximity_kernel=None), 3)
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
    _, P = etho._tracks(POSE, 0, nodes, 'm')
    assert np.array_equal(etho._features(P, old, 3), etho._features(P, new, 3))


def test_others_scale():
    feat = dict(CLUSTERS['features'], align_others=False)
    _, P = etho._tracks(POSE, 0, feat['nodes'], 'm')
    X1, X3 = etho._features(P, feat, 3), etho._features(P, dict(feat, others_scale=3.0), 3)
    d = X1.shape[1] // 2
    assert np.array_equal(X1[:, :d], X3[:, :d])            # own pose untouched
    assert np.allclose(X3[:, d:], 3 * X1[:, d:])


def test_analysis_notebook_runs_on_applied_result():
    # the result file is a normal etho-clusters export: the viewer's analysis notebook reads it
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
