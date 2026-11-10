"""poser: apply a behaviour clustering made in the Poser viewer to new pose data.

The viewer's "Export script" saves a clustering run as a poser-model file: the feature settings
(ego or social, removed nodes, body axis, proximity field), the PCA (mean and loadings)
and the switching linear dynamics model (each state's flow field and noise, the transition matrix).
This module rebuilds the same features from any poser-pose dataset with the same skeleton, projects
them onto the same principal components and segments them with the same model (Viterbi), so a state
means the same behaviour in every dataset it is applied to.

    import poser
    model = poser.Model.load('mice_model.json')     # a poser-model or poser-clusters file
    result = model.apply('new_session.json')       # a poser-pose file (or the parsed dict)
    result.time_budget()                           # pandas tables, see Result
    result.save('new_session_clusters.json')       # poser-clusters 1.0, like the viewer's export

If the model was trained on data cleaned in the viewer's Clean tab, new data is cleaned the same way
first (clean_pose; skipped for files that are already cleaned). The model is applied, not refit: new data gets the trained states and transition probabilities.
Only models fit on PCA scores can be applied (the viewer's default); models fit on raw features do
not store what is needed.

Formats: poser-pose (viewer/POSE_FORMAT.md), poser-clusters and poser-model (viewer/CLUSTER_FORMAT.md).
Needs numpy; the tables need pandas.
"""
from __future__ import annotations

import datetime
import json
import math
import warnings
from pathlib import Path

import numpy as np

__all__ = ['Model', 'Result', 'load_pose', 'clean_pose']
__version__ = '1.0'

UNIT_SCALE = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}     # length units the model can be applied across
MEASURE_UNITS = {'speed': '{L}/s', 'accel': '{L}/s²', 'turn_rate': '°/s', 'centroid_height': '{L}',
                 'max_height': '{L}', 'body_length': '{L}', 'nearest_distance': '{L}', 'nose_nose': '{L}'}


# ---------- loading ----------
def _load_json(src):
    if isinstance(src, dict):
        return src
    with open(src, encoding='utf-8') as f:
        return json.load(f)


def load_pose(src):
    """A poser-pose dataset from a path or an already parsed dict.

    Legacy pose-testset files are not converted here: open them in the viewer, which converts them.
    """
    ds = _load_json(src)
    if ds.get('format') not in ('poser-pose', 'etho-pose'):           # etho-*: the project's earlier name
        raise ValueError(f"not a poser-pose dataset (format = {ds.get('format')!r})")
    return ds


# ---------- pose clean-up (same steps as viewer/pose_clean.js, the viewer's Clean tab) ----------
_ANT = ['neck', 'chest', 'spine_mid', 'head', 'snout', 'nose']
_POST = ['pelvis', 'tail_base', 'hips']


def _floor_axes(space):
    if int(space['dims']) == 2:
        return 0, 1
    up = space.get('up', '+Y')[1]
    return (0, 2) if up == 'Y' else (0, 1) if up == 'Z' else (1, 2)


def _axis_joints(names):
    ant = next((names.index(n) for n in _ANT if n in names), None)
    post = next((names.index(n) for n in _POST if n in names), None)
    return (ant, post) if ant is not None and post is not None and ant != post else (None, None)


def _fix_swaps(P, names, space):
    """Swap X_L / X_R back in frames where both sit on the wrong side of the body axis."""
    ant, post = _axis_joints(names)
    fixed = {}
    if ant is None:
        return fixed
    ax, az = _floor_axes(space)
    u = P[:, ant, [ax, az]] - P[:, post, [ax, az]]
    n = np.hypot(u[:, 0], u[:, 1])

    def lat(j):
        d = P[:, j, [ax, az]] - P[:, post, [ax, az]]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, (u[:, 0] * d[:, 1] - u[:, 1] * d[:, 0]) / n, np.nan)
    for i, nm in enumerate(names):
        if not nm.endswith('_L') or nm[:-2] + '_R' not in names:
            continue
        k = names.index(nm[:-2] + '_R')
        li, lk = lat(i), lat(k)
        ok = np.isfinite(li) & np.isfinite(lk)
        if ok.sum() < 10:
            continue
        si, sk = np.sign(np.median(li[ok])), np.sign(np.median(lk[ok]))
        if not si or si == sk:
            continue
        mi, mk = 0.25 * np.median(np.abs(li[ok])), 0.25 * np.median(np.abs(lk[ok]))
        with np.errstate(invalid='ignore'):
            bad = (li * si < -mi) & (lk * sk < -mk)
        if bad.any():
            P[bad, i], P[bad, k] = P[bad, k].copy(), P[bad, i].copy()
            fixed[f'{nm}/{names[k]}'] = int(bad.sum())
    return fixed


def _windows(v, h):
    """frames x (2h+1) windows of a series, NaN beyond the ends."""
    pad = np.concatenate([np.full(h, np.nan), v, np.full(h, np.nan)])
    return np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)


def _remove_outliers(P, window, threshold):
    nF, J, dims = P.shape
    h = max(1, window // 2)
    removed = np.zeros(J, int)
    for j in range(J):
        present = np.isfinite(P[:, j, 0])
        d2, ok = np.zeros(nF), present.copy()
        for c in range(dims):
            W = _windows(P[:, j, c], h)
            cnt = np.isfinite(W).sum(1)
            ok &= cnt >= 3
            with warnings.catch_warnings():                    # all-NaN windows: median NaN, masked by ok
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(W, axis=1)
            d2 += np.where(ok, (P[:, j, c] - med) ** 2, 0)
        r = np.where(ok, np.sqrt(d2), np.nan)
        scale = 1.4826 * np.median(r[np.isfinite(r)]) if np.isfinite(r).any() else 0
        if not scale > 0:
            continue
        with np.errstate(invalid='ignore'):
            bad = r > threshold * scale
        P[bad, j] = np.nan
        removed[j] = int(bad.sum())
    return removed


def _fill_series(v, cubic=False):
    known = np.flatnonzero(np.isfinite(v))
    if not len(known):
        return False
    v[:known[0]] = v[known[0]]
    v[known[-1] + 1:] = v[known[-1]]
    for q in range(len(known) - 1):
        i0, i1 = known[q], known[q + 1]
        h = i1 - i0
        if h < 2:
            continue
        p0, p1 = v[i0], v[i1]
        m0 = m1 = (p1 - p0) / h
        if cubic:
            if q > 0:
                m0 = (p1 - v[known[q - 1]]) / (i1 - known[q - 1])
            if q + 2 < len(known):
                m1 = (v[known[q + 2]] - p0) / (known[q + 2] - i0)
        t = (np.arange(i0 + 1, i1) - i0) / h
        if not cubic:
            v[i0 + 1:i1] = p0 + (p1 - p0) * t
        else:
            t2, t3 = t * t, t * t * t
            v[i0 + 1:i1] = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * h * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * h * m1
    return True


def _fill_body(P, names, space):
    """Fill the body-axis nodes linearly and every other joint via its offset in the body frame."""
    ant, post = _axis_joints(names)
    if ant is None:
        return None
    nF, J, dims = P.shape
    ax, az = _floor_axes(space)
    ups = [c for c in range(dims) if c not in (ax, az)]
    empty = set()
    for j in (post, ant):
        for c in range(dims):
            if not _fill_series(P[:, j, c]):
                empty.add(j)
    if empty:
        return None
    ux, uz = P[:, ant, ax] - P[:, post, ax], P[:, ant, az] - P[:, post, az]
    n = np.hypot(ux, uz)
    n[n == 0] = 1
    cs, sn = ux / n, uz / n
    for j in range(J):
        if j in (post, ant):
            continue
        miss = ~np.isfinite(P[:, j, 0])
        if not miss.any():
            continue
        dx, dz = P[:, j, ax] - P[:, post, ax], P[:, j, az] - P[:, post, az]
        fw, lat = dx * cs + dz * sn, -dx * sn + dz * cs
        up = [P[:, j, c] - P[:, post, c] for c in ups]
        if not _fill_series(fw):
            empty.add(j)
            continue
        _fill_series(lat)
        for u in up:
            _fill_series(u)
        P[miss, j, ax] = P[miss, post, ax] + fw[miss] * cs[miss] - lat[miss] * sn[miss]
        P[miss, j, az] = P[miss, post, az] + fw[miss] * sn[miss] + lat[miss] * cs[miss]
        for c, u in zip(ups, up):
            P[miss, j, c] = P[miss, post, c] + u[miss]
    return empty


def _smooth_series(v, method, size):
    if method == 'median':
        W = _windows(v, max(1, int(size) // 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            v[:] = np.nanmedian(W, axis=1)
        return
    R = max(1, math.ceil(3 * size))
    k = np.exp(-0.5 * (np.arange(-R, R + 1) / size) ** 2)
    v[:] = np.convolve(v, k, 'same') / np.convolve(np.ones_like(v), k, 'same')


def clean_pose(pose, settings):
    """A cleaned copy of a poser-pose dataset (the viewer's Clean tab; see viewer/pose_clean.js).

    settings, as stored in exports under `preprocessing`: {swaps: bool, outliers: {window, threshold}
    or None, fill: 'linear' | 'cubic' | 'body', smooth: {method: 'gaussian' | 'median', size} or None}.
    Steps: left/right swap fix, outlier removal (Hampel), gap filling, smoothing.
    """
    pose = json.loads(json.dumps(load_pose(pose)))
    space, dims = pose['space'], int(pose['space']['dims'])
    out_ = settings.get('outliers')
    sm = settings.get('smooth')
    for clip in pose['clips']:
        nF = int(clip['frame_count'])
        for an in pose['animals']:
            tr = clip['tracks'].get(an['id'])
            if tr is None:
                continue
            names = [j['name'] for j in pose['skeletons'][an['skeleton']]['joints']]
            J = len(names)
            P = np.full((nF, J, dims), np.nan)
            for f, fr in enumerate(tr['positions'][:nF]):
                if fr is None:
                    continue
                for j, q in enumerate(fr):
                    if q is not None:
                        P[f, j] = q
            if settings.get('swaps'):
                _fix_swaps(P, names, space)
            if out_:
                _remove_outliers(P, int(out_['window']), float(out_['threshold']))
            empty = _fill_body(P, names, space) if settings.get('fill') == 'body' else None
            body = empty is not None
            empty = empty or set()
            for j in range(J):
                if j in empty:
                    continue
                for c in range(dims):
                    v = P[:, j, c]
                    if not body and not _fill_series(v, settings.get('fill') == 'cubic'):
                        empty.add(j)
                        break
                    if sm and sm.get('method') in ('gaussian', 'median'):
                        _smooth_series(v, sm['method'], float(sm['size']))
            tr['positions'] = [[None if j in empty else P[f, j].tolist() for j in range(J)] for f in range(nF)]
    pose['meta'] = dict(pose.get('meta') or {}, cleaned=settings)
    return pose


def _clip_index(pose, clip):
    if isinstance(clip, str):
        names = [c.get('name') for c in pose['clips']]
        if clip not in names:
            raise KeyError(f'no clip named {clip!r}; clips: {names}')
        return names.index(clip)
    if not 0 <= clip < len(pose['clips']):
        raise IndexError(f"clip {clip} out of range ({len(pose['clips'])} clips)")
    return clip


def _to_y_up(p, space):
    """Viewer convention: x, height, z. 2D (x, y) lies on the floor."""
    if space['dims'] == 2:
        return np.stack([p[..., 0], np.zeros(p.shape[:-1]), p[..., 1]], -1)
    up = space.get('up', '+Y')
    s = -1.0 if up[0] == '-' else 1.0
    if up[1] == 'Y':
        return np.stack([p[..., 0], s * p[..., 1], s * p[..., 2]], -1)
    if up[1] == 'Z':
        return np.stack([p[..., 0], s * p[..., 2], -s * p[..., 1]], -1)
    return np.stack([s * p[..., 1], s * p[..., 0], -p[..., 2]], -1)


def _tracks(pose, ci, nodes, units):
    """Every tracked animal's joints as frames x nodes x 3 (Y up, in `units`), gaps interpolated.

    Joints are matched to the model's nodes by name, so the skeleton may list them in another order
    or carry extra joints (ignored).
    """
    clip, space = pose['clips'][ci], pose['space']
    nF, dims = int(clip['frame_count']), int(space['dims'])
    src_units = space.get('units', 'm')
    if src_units not in UNIT_SCALE or units not in UNIT_SCALE:
        if src_units != units:
            raise ValueError(f'cannot convert lengths from {src_units!r} to {units!r}')
        scale = 1.0
    else:
        scale = UNIT_SCALE[src_units] / UNIT_SCALE[units]
    animals, out = [], []
    for a in pose['animals']:
        if a['id'] not in clip['tracks']:
            continue
        names = [j['name'] for j in pose['skeletons'][a['skeleton']]['joints']]
        missing = [n for n in nodes if n not in names]
        if missing:
            raise ValueError(f"{a.get('name', a['id'])}: skeleton lacks the model's node(s) {missing}")
        idx = [names.index(n) for n in nodes]
        raw = clip['tracks'][a['id']]['positions']
        arr = np.full((nF, len(nodes), dims), np.nan)
        for f, fr in enumerate(raw[:nF]):
            if fr is None:
                continue
            for k, j in enumerate(idx):
                if fr[j] is not None:
                    arr[f, k] = fr[j]
        t = np.arange(nF)
        for k in range(len(nodes)):
            for c in range(dims):
                v = arr[:, k, c]
                ok = ~np.isnan(v)
                if not ok.any():
                    raise ValueError(f"{a.get('name', a['id'])}: node {nodes[k]!r} has no data in this clip")
                if not ok.all():
                    arr[:, k, c] = np.interp(t, t[ok], v[ok])      # linear, ends held flat
        out.append(_to_y_up(arr, space) * scale)
        animals.append({'id': a['id'], 'name': a.get('name') or a['id']})
    if not out:
        raise ValueError('the clip has no tracked animals')
    return animals, np.stack(out)                                   # animals x frames x nodes x 3


# ---------- features (same construction as the viewer's buildFeatures) ----------
def _features(P, feat, dims):
    """Feature matrix, one row per animal per frame (animal 1's frames first), as the model saw it."""
    A, nF, J, _ = P.shape
    nodes = feat['nodes']
    ant, post = nodes.index(feat['egocentric_axis']['anterior']), nodes.index(feat['egocentric_axis']['posterior'])
    cen = P.mean(axis=2)                                           # centroid of every node
    height = cen[..., 1].copy() if dims == 3 else np.zeros((A, nF))
    cen[..., 1] = 0.0                                              # ... on the floor
    th = np.arctan2(P[:, :, ant, 2] - P[:, :, post, 2], P[:, :, ant, 0] - P[:, :, post, 0])
    cs, sn = np.cos(th), np.sin(th)                                # animals x frames

    def in_frame(a, Q, rot=None):
        """Q (frames x nodes x 3) in animal a's egocentric frame, optionally turned by (cos, sin)."""
        dx, dz = Q[..., 0] - cen[a, :, None, 0], Q[..., 2] - cen[a, :, None, 2]
        fw = dx * cs[a, :, None] + dz * sn[a, :, None]
        lat = -dx * sn[a, :, None] + dz * cs[a, :, None]
        if rot is not None:
            ca, sa = rot[0][:, None], rot[1][:, None]
            fw, lat = fw * ca + lat * sa, -fw * sa + lat * ca
        cols = [fw, Q[..., 1], lat] if dims == 3 else [fw, lat]
        return np.stack(cols, -1).reshape(nF, -1)

    social = feat['mode'] == 'social'
    d = J * (3 if dims == 3 else 2)
    if social and A < 2:
        raise ValueError('this model uses social features, which need at least two animals')
    blocks = A if social else 1
    kernel = feat.get('proximity_kernel') if social else None
    # exports before these settings existed used floor distances and an isotropic kernel
    d3 = dims == 3 and int(feat.get('distance_dims', 2)) == 3
    side = float(kernel.get('lateral_scale', 1.0)) if kernel else 1.0
    beta = float(feat.get('others_scale') or 1.0) if social else 1.0   # others vs own (own pose stays at 1)
    X = np.zeros((A, nF, d * blocks))
    for a in range(A):
        X[a, :, :d] = in_frame(a, P[a])
        if not social:
            continue
        others = [b for b in range(A) if b != a]
        dh = np.stack([height[b] - height[a] if d3 else np.zeros(nF) for b in others])
        dist = np.stack([np.sqrt((cen[b, :, 0] - cen[a, :, 0]) ** 2 + (cen[b, :, 2] - cen[a, :, 2]) ** 2 + dh[i] ** 2)
                         for i, b in enumerate(others)])
        order = np.argsort(dist, axis=0, kind='stable')            # nearest other animal first, per frame
        for o in range(len(others)):
            pick = np.array(others)[order[o]]                       # which animal fills slot o in each frame
            Q = P[pick, np.arange(nF)]                              # frames x nodes x 3
            bx, bz = cen[pick, np.arange(nF), 0] - cen[a, :, 0], cen[pick, np.arange(nF), 2] - cen[a, :, 2]
            rot = None
            if feat.get('align_others'):   # older exports: other animal rotated onto the body axis
                phi = np.arctan2(-bx * sn[a] + bz * cs[a], bx * cs[a] + bz * sn[a])
                phi = np.where(phi > np.pi / 2, phi - np.pi, np.where(phi <= -np.pi / 2, phi + np.pi, phi))
                rot = (np.cos(phi), np.sin(phi))
            if kernel:   # Gaussian field in the focal frame: d0 ahead/behind and in height, side * d0 sideways
                fwc, latc = bx * cs[a] + bz * sn[a], -bx * sn[a] + bz * cs[a]
                hh = dh[order[o], np.arange(nF)]
                k0 = kernel['d0']
                w = np.exp(-0.5 * ((fwc ** 2 + hh ** 2) / k0 ** 2 + latc ** 2 / (side * k0) ** 2))
            else:
                w = np.ones(nF)
            X[a, :, d * (o + 1):d * (o + 2)] = in_frame(a, Q, rot) * (w * beta)[:, None]
    removed = set(feat.get('removed_nodes') or [])
    axes = 3 if dims == 3 else 2
    keep = np.array([nodes[(k % d) // axes] not in removed for k in range(d * blocks)], float)
    return (X * keep).reshape(A * nF, -1)


# ---------- the model ----------
class Model:
    """A clustering run from the viewer: feature settings, PCA and switching linear dynamics.

    Load it from a poser-model file (Export script) or a poser-clusters file (Clusters .json).
    """

    def __init__(self, spec):
        if spec.get('format') not in ('poser-model', 'poser-clusters', 'etho-model', 'etho-clusters'):
            raise ValueError(f"not a poser-model or poser-clusters file (format = {spec.get('format')!r})")
        self.spec = {k: v for k, v in spec.items() if k != 'animals'}
        self.features = spec['features']
        self.reduction = spec['reduction']
        self.model = spec['model']
        self.space = spec['space']
        self.preprocessing = spec.get('preprocessing')       # the Clean tab's settings, or None
        self.states = self.model['states']
        self.k = len(self.states)
        self.state_names = [f"S{s['id']}" for s in self.states]
        self.colors = [s['color'] for s in self.states]

    @classmethod
    def load(cls, src):
        return cls(_load_json(src))

    def save(self, path):
        """Write the poser-model file (settings, PCA and dynamics; no per-frame data)."""
        spec = dict(self.spec, format='poser-model', version='1.0')
        Path(path).write_text(json.dumps(spec))

    def __repr__(self):
        f, r = self.features, self.reduction
        return (f"<poser.Model: {self.k} states, {f['mode']} features on {len(f['nodes'])} nodes, "
                f"{r['n_components']} {'PCs' if r['method'] == 'pca' else 'raw features'}, "
                f"trained on {self.spec.get('source', {}).get('dataset', '?')}>")

    @property
    def applicable(self):
        return self.reduction['method'] == 'pca' and all(s.get('A') is not None for s in self.states)

    # -- the pipeline, step by step --
    def feature_matrix(self, pose, clip=0):
        """(animals, feature matrix with one row per animal per frame) for one clip of a poser-pose dataset."""
        pose = load_pose(pose)
        if int(pose['space']['dims']) != int(self.space['dims']):
            raise ValueError(f"the model is {self.space['dims']}D, the data {pose['space']['dims']}D")
        ci = _clip_index(pose, clip)
        animals, P = _tracks(pose, ci, self.features['nodes'], self.space.get('units', 'm'))
        return animals, _features(P, self.features, int(self.space['dims']))

    def embed(self, X):
        """Project feature rows onto the model's principal components."""
        mean, L = np.asarray(self.reduction['mean']), np.asarray(self.reduction['loadings'])
        if X.shape[1] != len(mean):
            hint = ' (social features: the data must have as many animals as the training data)' if self.features['mode'] == 'social' else ''
            raise ValueError(f'{X.shape[1]} features, the model expects {len(mean)}{hint}')
        return (X - mean) @ L.T

    def segment(self, Z, n_animals, loop=False):
        """Most likely state per frame (Viterbi), 0-based, shape animals x frames."""
        if not self.applicable:
            raise ValueError('this model was fit on raw features, so it cannot be applied to new data; '
                             'rerun the clustering with PCA reduction')
        m, k = int(self.model['dims']), self.k
        A = np.array([s['A'] for s in self.states])                 # k x m x m: next = A @ z + b
        b = np.array([s['b'] for s in self.states])
        Lc = np.linalg.cholesky(np.array([s['noise_cov'] for s in self.states]))
        logdet = 2 * np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)
        logP = np.log(np.asarray(self.model['transition_matrix']))
        logpi = np.log(np.asarray(self.model['initial_distribution']))
        nF = len(Z) // n_animals
        if nF < 2:
            raise ValueError('need at least two frames')
        labels = np.zeros((n_animals, nF), int)
        for a in range(n_animals):
            z = Z[a * nF:(a + 1) * nF, :m]
            T = nF if loop else nF - 1                              # transitions: frame t -> t + 1
            x, y = z[:T], z[(np.arange(T) + 1) % nF]
            res = y[None] - (np.einsum('cqa,ta->ctq', A, x) + b[:, None])          # k x T x m
            u = np.stack([np.linalg.solve(Lc[c], res[c].T) for c in range(k)])     # k x m x T
            logem = (-0.5 * (m * math.log(2 * math.pi) + logdet[:, None] + (u ** 2).sum(1))).T   # T x k
            dl, bp = logpi + logem[0], np.zeros((T, k), int)
            for t in range(1, T):
                cand = dl[:, None] + logP                           # from j (rows) to c (columns)
                bp[t] = cand.argmax(0)
                dl = cand.max(0) + logem[t]
            path = np.zeros(T, int)
            path[-1] = dl.argmax()
            for t in range(T - 1, 0, -1):
                path[t - 1] = bp[t, path[t]]
            labels[a, :T] = path
            if T < nF:
                labels[a, T:] = path[-1]                            # the last frame of an open clip
        return labels

    def prepare(self, pose):
        """The dataset as the model needs it: cleaned like the training data (unless already cleaned)."""
        pose = load_pose(pose)
        if self.preprocessing and not (pose.get('meta') or {}).get('cleaned'):
            pose = clean_pose(pose, self.preprocessing)
        return pose

    def apply(self, pose, clip=0):
        """Run the whole pipeline on one clip of a poser-pose dataset: clean-up, features, PCs, states."""
        pose = self.prepare(pose)
        ci = _clip_index(pose, clip)
        animals, X = self.feature_matrix(pose, ci)
        Z = self.embed(X)
        labels = self.segment(Z, len(animals), bool(pose['clips'][ci].get('loop')))
        nF = labels.shape[1]
        return Result(self, pose, ci, animals, labels + 1, Z.reshape(len(animals), nF, -1))

    def apply_all(self, pose):
        """apply() to every clip of a dataset; a list of Results."""
        pose = self.prepare(pose)
        return [self.apply(pose, i) for i in range(len(pose['clips']))]


# ---------- results ----------
class Result:
    """States of every animal in one clip. `labels` is animals x frames, 1-based (state 1 = S1)."""

    def __init__(self, model, pose, clip_index, animals, labels, embedding):
        self.model, self.pose, self.clip_index = model, pose, clip_index
        self.clip = pose['clips'][clip_index]
        self.fps = float(self.clip['fps'])
        self.animals, self.labels, self.embedding = animals, labels, embedding
        self.n_frames = labels.shape[1]

    def __repr__(self):
        return (f"<poser.Result: {self.pose.get('meta', {}).get('name', 'dataset')} · clip {self.clip.get('name', self.clip_index)}, "
                f"{len(self.animals)} animal(s) x {self.n_frames} frames, {self.model.k} states>")

    def _names(self):
        return self.model.state_names

    def bout_list(self):
        """Every bout as dicts: animal index, state (1-based), start and end frame, duration in s."""
        out = []
        for a, lab in enumerate(self.labels):
            change = np.flatnonzero(np.diff(lab)) + 1
            starts, ends = np.r_[0, change], np.r_[change, len(lab)] - 1
            out += [{'animal': a, 'state': int(lab[s]), 'start': int(s), 'end': int(e), 'duration_s': (e - s + 1) / self.fps}
                    for s, e in zip(starts, ends)]
        return out

    # -- tables (pandas) --
    def bouts(self):
        import pandas as pd
        df = pd.DataFrame(self.bout_list())
        df['animal'] = [self.animals[a]['name'] for a in df['animal']]
        df['state'] = [self._names()[s - 1] for s in df['state']]
        return df

    def time_budget(self):
        """% of frames in each state, per animal and over all animals."""
        import pandas as pd
        k = self.model.k
        rows = {an['name']: np.bincount(lab - 1, minlength=k) / len(lab) * 100 for an, lab in zip(self.animals, self.labels)}
        rows['all'] = np.bincount(self.labels.ravel() - 1, minlength=k) / self.labels.size * 100
        return pd.DataFrame(rows, index=self._names()).T.round(1)

    def bout_stats(self):
        """Per state: number of bouts and their mean, median, 90th percentile and total duration (s)."""
        import pandas as pd
        b = pd.DataFrame(self.bout_list())
        g = b.groupby('state')['duration_s']
        t = pd.DataFrame({'bouts': g.size(), 'mean_s': g.mean(), 'median_s': g.median(),
                          'p90_s': g.quantile(0.9), 'total_s': g.sum()}).reindex(range(1, self.model.k + 1))
        t.index = self._names()
        return t.fillna({'bouts': 0}).astype({'bouts': int})

    def transitions(self):
        """What follows each bout: rows = from, columns = to, each row sums to 1."""
        import pandas as pd
        k = self.model.k
        T = np.zeros((k, k))
        for lab in self.labels:
            ch = np.flatnonzero(np.diff(lab))
            np.add.at(T, (lab[ch] - 1, lab[ch + 1] - 1), 1)
        s = T.sum(1, keepdims=True)
        return pd.DataFrame(np.divide(T, s, out=np.zeros_like(T), where=s > 0), index=self._names(), columns=self._names())

    def measures(self):
        """One row per animal per frame: state and movement measures (lengths in cm for metre data).

        speed, accel (centroid, floor plane), turn_rate (body axis), centroid_height and max_height
        (3D), body_length (anterior-posterior) and, with several animals, nearest_distance (centroid to
        nearest centroid) and nose_nose. Derivatives use central differences.
        """
        import pandas as pd
        feat, units = self.model.features, self.model.space.get('units', 'm')
        nodes = feat['nodes']
        _, P = _tracks(self.pose, self.clip_index, nodes, units)
        P = P * (100.0 if units == 'm' else 1.0)
        ant, post = nodes.index(feat['egocentric_axis']['anterior']), nodes.index(feat['egocentric_axis']['posterior'])
        nose = next((nodes.index(n) for n in ('snout', 'nose', 'head') if n in nodes), ant)
        dims, A, fps = int(self.model.space['dims']), len(self.animals), self.fps
        rows = []
        for a, an in enumerate(self.animals):
            p, cen = P[a], P[a].mean(axis=1)
            vel = np.gradient(cen[:, [0, 2]], axis=0) * fps
            ax = p[:, ant, [0, 2]] - p[:, post, [0, 2]]
            rec = {'animal': an['name'], 'frame': np.arange(self.n_frames), 'time_s': np.arange(self.n_frames) / fps,
                   'state': self.labels[a], 'speed': np.linalg.norm(vel, axis=1),
                   'accel': np.linalg.norm(np.gradient(vel, axis=0) * fps, axis=1),
                   'turn_rate': np.abs(np.degrees(np.gradient(np.unwrap(np.arctan2(ax[:, 1], ax[:, 0]))) * fps)),
                   'body_length': np.linalg.norm(p[:, ant] - p[:, post], axis=1)}
            if dims == 3:
                rec['centroid_height'], rec['max_height'] = cen[:, 1], p[:, :, 1].max(axis=1)
            if A > 1:
                others = [b for b in range(A) if b != a]
                rec['nearest_distance'] = np.min([np.linalg.norm(P[b].mean(1)[:, [0, 2]] - cen[:, [0, 2]], axis=1) for b in others], 0)
                rec['nose_nose'] = np.min([np.linalg.norm(P[b][:, nose] - p[:, nose], axis=1) for b in others], 0)
            rows.append(pd.DataFrame(rec))
        return pd.concat(rows, ignore_index=True)

    def state_summary(self):
        """Per state: share of frames, median bout and the median of every measure."""
        m = self.measures()
        cols = [c for c in MEASURE_UNITS if c in m]
        L = 'cm' if self.model.space.get('units', 'm') == 'm' else self.model.space.get('units')
        t = m.groupby('state')[cols].median().reindex(range(1, self.model.k + 1))
        t.index = self._names()
        t.columns = [f"{c} ({MEASURE_UNITS[c].format(L=L)})" for c in cols]
        t.insert(0, 'median bout (s)', self.bout_stats()['median_s'])
        t.insert(0, '% of frames', self.time_budget().loc['all'])
        return t

    # -- export --
    def to_clusters(self):
        """This result as a poser-clusters 1.0 dict (same layout as the viewer's Clusters export)."""
        spec, k, nF = self.model.spec, self.model.k, self.n_frames
        bl, total = self.bout_list(), self.labels.size
        states = []
        for c, s in enumerate(self.model.states):
            frames = int((self.labels == c + 1).sum())
            nb = sum(1 for b in bl if b['state'] == c + 1)
            states.append(dict(s, frames=frames, share=frames / total, bouts=nb, mean_bout_s=frames / nb / self.fps if nb else 0))
        rnd = lambda v: float(f'{v:.6g}')
        model = dict(spec['model'], states=states)
        return {
            'format': 'poser-clusters', 'version': '1.0',
            'created': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'generator': f'poser.py {__version__} (model applied, not refit)',
            'source': {'dataset': self.pose.get('meta', {}).get('name'), 'file': None,
                       'clip': {'index': self.clip_index, 'name': self.clip.get('name'), 'fps': self.fps,
                                'frame_count': nF, 'loop': bool(self.clip.get('loop'))}},
            'trained_on': spec.get('source'),
            'preprocessing': spec.get('preprocessing'),
            'space': spec['space'], 'features': spec['features'], 'reduction': spec['reduction'], 'model': model,
            'animals': [{'id': an['id'], 'name': an['name'], 'labels': self.labels[a].tolist(),
                         'bouts': [{kk: b[kk] for kk in ('state', 'start', 'end', 'duration_s')} for b in bl if b['animal'] == a],
                         'embedding': [[rnd(v) for v in row] for row in self.embedding[a]]}
                        for a, an in enumerate(self.animals)],
        }

    def save(self, path):
        """Write to_clusters() as JSON (readable by the analysis notebook from the viewer)."""
        Path(path).write_text(json.dumps(self.to_clusters()))
