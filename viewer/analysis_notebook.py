# Analysis notebook template (Jupytext "percent" format).
#
# build_viewer.py splits this file into notebook cells at the "# %%" markers and embeds them in the
# viewer. When a user exports "Analysis notebook (.zip)", the viewer puts a title cell and a
# configuration cell (POSE_PATH, CLUSTERS_PATH) in front of these cells. To run this file directly
# for testing, set POSE_PATH and CLUSTERS_PATH in the environment.
#
# Only numpy, pandas and matplotlib are needed.

# %%
# (test shim: not exported)
import os
if 'POSE_PATH' not in globals():          # running as a plain script (testing)
    POSE_PATH = os.environ['POSE_PATH']
    CLUSTERS_PATH = os.environ['CLUSTERS_PATH']
    import matplotlib
    matplotlib.use('Agg')

# %% [markdown]
# ## 1. Setup
# Loads the pose data (etho-pose 1.x) and the clustering output (etho-clusters 1.x), rebuilds each
# animal's pose for the clustered clip, and computes the per-frame measures used below.

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'font.size': 9,
})
pd.set_option('display.precision', 3)
try:
    from IPython.display import display, Markdown
except ImportError:                       # plain Python: print instead
    display = print
    Markdown = str


def boxplot(ax, data, labels, **kw):
    """ax.boxplot across matplotlib versions (tick_labels since 3.9, labels before)."""
    data = [d if len(d) else [np.nan] for d in data]
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)

with open(POSE_PATH) as f:
    pose = json.load(f)
with open(CLUSTERS_PATH) as f:
    cl = json.load(f)
assert pose.get('format') == 'etho-pose', 'POSE_PATH is not an etho-pose file'
assert cl.get('format') == 'etho-clusters', 'CLUSTERS_PATH is not an etho-clusters file'

src = cl['source']
clip = pose['clips'][src['clip']['index']]
fps = float(clip['fps'])
nF = int(clip['frame_count'])
model = cl['model']
states = model['states']
K = len(states)
COLORS = [s['color'] for s in states]
STATE_NAMES = [f"S{s['id']}" for s in states]
CMAP = ListedColormap(COLORS)
animals = cl['animals']
A = len(animals)
units = pose['space'].get('units', 'm')
TO_DISPLAY = 100.0 if units == 'm' else 1.0          # metres are shown in cm
LEN = 'cm' if units == 'm' else units
dims = int(pose['space']['dims'])
skel_of = {a['id']: pose['skeletons'][a['skeleton']] for a in pose['animals']}


def to_y_up(p):
    """Viewer convention: x, height, z. 2D (x, y) lies on the floor."""
    p = np.asarray(p, float)
    if dims == 2:
        return np.stack([p[..., 0], np.zeros(p.shape[:-1]), p[..., 1]], -1)
    up = pose['space'].get('up', '+Y')
    s = -1.0 if up[0] == '-' else 1.0
    if up[1] == 'Y':
        return np.stack([p[..., 0], s * p[..., 1], s * p[..., 2]], -1)
    if up[1] == 'Z':
        return np.stack([p[..., 0], s * p[..., 2], -s * p[..., 1]], -1)
    return np.stack([s * p[..., 1], s * p[..., 0], -p[..., 2]], -1)


def track_array(animal_id):
    """frames x joints x 3 (Y up, display units), missing samples linearly interpolated."""
    raw = clip['tracks'][animal_id]['positions']
    J = len(skel_of[animal_id]['joints'])
    out = np.full((nF, J, dims), np.nan)
    for f, fr in enumerate(raw):
        if fr is None:
            continue
        for j, q in enumerate(fr):
            if q is not None:
                out[f, j] = q
    t = np.arange(nF)
    for j in range(J):
        for c in range(dims):
            v = out[:, j, c]
            ok = ~np.isnan(v)
            if ok.any() and not ok.all():
                out[:, j, c] = np.interp(t, t[ok], v[ok])
    return to_y_up(out) * TO_DISPLAY


P = {a['id']: track_array(a['id']) for a in animals}
LABELS = {a['id']: np.asarray(a['labels'], int) for a in animals}   # 1-based state ids
joint_names = [j['name'] for j in skel_of[animals[0]['id']]['joints']]
bones = skel_of[animals[0]['id']].get('bones', [])
J = len(joint_names)
axis_info = cl['features'].get('egocentric_axis', {})
ANT = joint_names.index(axis_info['anterior']) if axis_info.get('anterior') in joint_names else J - 1
POST = joint_names.index(axis_info['posterior']) if axis_info.get('posterior') in joint_names else 0
NOSE = next((joint_names.index(n) for n in ('snout', 'nose', 'head') if n in joint_names), ANT)
print(f"{pose.get('meta', {}).get('name', 'dataset')} · clip '{clip.get('name', src['clip']['index'])}': "
      f"{nF} frames at {fps:g} fps ({nF / fps:.1f} s), {A} animal(s), {J} nodes, {K} states")

# %% [markdown]
# ### Per-frame measures
# For every animal and frame: centroid speed and acceleration (floor plane), turning rate of the body
# axis, centroid and maximum height (3D data), body length (anterior–posterior distance) and, with
# several animals, distance to the nearest other animal and nose-to-nose distance. Derivatives use
# central differences.

# %%
def central_diff(x, axis=0):
    d = np.gradient(x, axis=axis)
    return d * fps


rows = []
for a in animals:
    aid = a['id']
    p = P[aid]
    cen = p.mean(axis=1)                                   # frames x 3
    vel = central_diff(cen[:, [0, 2]])
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(central_diff(vel), axis=1)
    axis_vec = p[:, ANT, [0, 2]] - p[:, POST, [0, 2]]
    heading = np.unwrap(np.arctan2(axis_vec[:, 1], axis_vec[:, 0]))
    turn = np.degrees(central_diff(heading))
    body_len = np.linalg.norm(p[:, ANT] - p[:, POST], axis=1)
    rec = dict(animal=a['name'], animal_id=aid, frame=np.arange(nF), time_s=np.arange(nF) / fps,
               state=LABELS[aid], speed=speed, accel=accel, turn_rate=np.abs(turn), body_length=body_len)
    if dims == 3:
        rec['centroid_height'] = cen[:, 1]
        rec['max_height'] = p[:, :, 1].max(axis=1)
    if A > 1:
        others = [P[b['id']] for b in animals if b['id'] != aid]
        dist = np.min([np.linalg.norm(o.mean(axis=1)[:, [0, 2]] - cen[:, [0, 2]], axis=1) for o in others], axis=0)
        nose = np.min([np.linalg.norm(o[:, NOSE] - p[:, NOSE], axis=1) for o in others], axis=0)
        rec['nearest_distance'] = dist
        rec['nose_nose'] = nose
    rows.append(pd.DataFrame(rec))
df = pd.concat(rows, ignore_index=True)
MEASURES = [c for c in ['speed', 'accel', 'turn_rate', 'centroid_height', 'max_height', 'body_length',
                        'nearest_distance', 'nose_nose'] if c in df]
UNITS = dict(speed=f'{LEN}/s', accel=f'{LEN}/s²', turn_rate='°/s', centroid_height=LEN, max_height=LEN,
             body_length=LEN, nearest_distance=LEN, nose_nose=LEN)
df.head()

# %% [markdown]
# ## 2. What was clustered
# The input features, the reduction and the model settings, exactly as exported by the viewer.

# %%
feat = cl['features']
red = cl['reduction']
info = {
    'feature set': feat['mode'],
    'features per sample': len(feat.get('feature_names', [])),
    'removed nodes': ', '.join(feat.get('removed_nodes', [])) or 'none',
    'egocentric axis': f"{axis_info.get('posterior')} → {axis_info.get('anterior')}",
}
if feat['mode'] == 'social':
    if feat.get('align_others'):                       # an option of older viewer versions
        info['other animals rotated onto the body axis'] = True
    k = feat.get('proximity_kernel')
    side = k.get('lateral_scale', 1) if k else 1
    info['proximity kernel'] = ('off' if not k else
                                f"elliptical Gaussian field: d0 = {k['d0_body_lengths']} body lengths ({k['d0'] * TO_DISPLAY:.1f} {LEN}) ahead and behind, "
                                f"{side * k['d0_body_lengths']:.2f} to the sides" if side < 1 else
                                f"{'isotropic %dD ' % k['dims'] if k.get('dims') else ''}Gaussian on centroid-to-centroid distance, d0 = {k['d0_body_lengths']} body lengths ({k['d0'] * TO_DISPLAY:.1f} {LEN})")
if feat['mode'] == 'social' and feat.get('others_scale', 1) != 1:
    info['others vs own'] = (f"balanced 50/50 in contact (others x{feat['others_scale']})" if feat.get('others_balance')
                             else f"other animals x{feat['others_scale']} relative to the own pose")
pp = cl.get('preprocessing')
if pp:                                                 # the viewer's Clean tab
    info['pose clean-up'] = ', '.join(x for x in [
        'left/right swaps fixed' if pp.get('swaps') else None,
        f"outliers removed (window {pp['outliers']['window']}, threshold {pp['outliers']['threshold']})" if pp.get('outliers') else None,
        f"{'body-relative' if pp['fill'] == 'body' else pp['fill']} gap filling",
        f"{pp['smooth']['method']} smoothing ({pp['smooth']['size']} frames)" if pp.get('smooth') else None] if x)
info['reduction'] = (f"PCA, {red['n_components']} PCs = {100 * red['cumulative_variance'][red['n_components'] - 1]:.1f}% of variance"
                     if red['method'] == 'pca' else f"none (model on {red['n_components']} raw features)")
info['model'] = f"switching linear dynamics, k = {model['k']} (best by BIC: k = {model['k_auto']})"
hp = model.get('hyperparameters', {})
info['k range searched'] = f"{hp.get('k_min')}–{hp.get('k_max')}"
info['stickiness'] = hp.get('stickiness')
info['log-likelihood per step'] = round(model['log_likelihood_per_step'], 3)
pd.Series(info, name='value').to_frame()

# %% [markdown]
# ### Model selection
# Every k in the searched range was fit; BIC (lower is better) trades fit against the number of
# parameters. The marked point is the k used for everything below.

# %%
ks = [r['k'] for r in model['k_selection']]
bic = [r['bic'] for r in model['k_selection']]
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot(ks, bic, '-o', color='#8fb5a8', ms=4)
ax.plot([model['k']], [bic[ks.index(model['k'])]], 'o', color='black', ms=8, mfc='none', mew=1.5)
ax.set_xlabel('k (number of states)'); ax.set_ylabel('BIC'); ax.set_xticks(ks)
ax.set_title('BIC by number of states')
plt.show()

# %% [markdown]
# ## 3. Ethogram
# State over time for each animal.

# %%
fig, ax = plt.subplots(figsize=(10, 0.55 * A + 0.9))
mat = np.array([LABELS[a['id']] - 1 for a in animals])
ax.imshow(mat, aspect='auto', cmap=CMAP, vmin=-0.5, vmax=K - 0.5, interpolation='nearest',
          extent=[0, nF / fps, A - 0.5, -0.5])
ax.set_yticks(range(A)); ax.set_yticklabels([a['name'] for a in animals])
ax.set_xlabel('time (s)')
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
ax.legend(handles, STATE_NAMES, ncol=min(K, 8), bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', frameon=False, fontsize=8)
plt.show()

# %% [markdown]
# ## 4. Time budget
# Share of frames in each state, overall and per animal.

# %%
share = df.groupby('state').size().reindex(range(1, K + 1), fill_value=0) / len(df)
per_animal = df.groupby(['animal', 'state']).size().unstack(fill_value=0).reindex(columns=range(1, K + 1), fill_value=0)
per_animal = per_animal.div(per_animal.sum(axis=1), axis=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
axes[0].bar(STATE_NAMES, share.values * 100, color=COLORS)
axes[0].set_ylabel('% of frames'); axes[0].set_title('Overall')
left = np.zeros(len(per_animal))
for c in range(K):
    axes[1].barh(per_animal.index, per_animal[c + 1] * 100, left=left, color=COLORS[c], label=STATE_NAMES[c])
    left += per_animal[c + 1].values * 100
axes[1].set_xlabel('% of frames'); axes[1].set_title('Per animal')
plt.tight_layout(); plt.show()
(per_animal * 100).round(1).rename(columns=dict(enumerate(STATE_NAMES, 1)))

# %% [markdown]
# ## 5. Bouts
# A bout is an unbroken run of one state within one animal's sequence.

# %%
bout_rows = []
for a in animals:
    for b in a['bouts']:
        bout_rows.append(dict(animal=a['name'], state=b['state'], start=b['start'], end=b['end'], duration_s=b['duration_s']))
bouts = pd.DataFrame(bout_rows)
fig, ax = plt.subplots(figsize=(8, 3))
data = [bouts.loc[bouts.state == c, 'duration_s'].values for c in range(1, K + 1)]
bp = boxplot(ax, data, STATE_NAMES, patch_artist=True, showfliers=True, medianprops=dict(color='black'), flierprops=dict(ms=3))
for patch, c in zip(bp['boxes'], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.set_yscale('log'); ax.set_ylabel('bout duration (s, log scale)'); ax.set_title('Bout durations by state')
plt.show()
bout_table = bouts.groupby('state')['duration_s'].agg(bouts='count', mean_s='mean', median_s='median',
                                                     p90_s=lambda x: np.percentile(x, 90), total_s='sum')
bout_table.index = [STATE_NAMES[i - 1] for i in bout_table.index]
bout_table

# %% [markdown]
# ## 6. Transitions
# Left: what follows each bout (rows = from, columns = to; each row sums to 1). Right: the model's
# frame-to-frame transition matrix, whose diagonal is the probability of staying in a state for one
# more frame (expected bout length ≈ 1 / (1 − p) frames).

# %%
T = np.zeros((K, K))
for a in animals:
    lab = LABELS[a['id']]
    change = np.flatnonzero(np.diff(lab) != 0)
    for i in change:
        T[lab[i] - 1, lab[i + 1] - 1] += 1
Tn = np.divide(T, T.sum(axis=1, keepdims=True), out=np.zeros_like(T), where=T.sum(axis=1, keepdims=True) > 0)
M = np.array(model['transition_matrix'])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, title in [(axes[0], Tn, 'Bout-to-bout transitions'), (axes[1], M, 'Model transition matrix (per frame)')]:
    im = ax.imshow(mat, cmap='Greys', vmin=0, vmax=max(mat.max(), 1e-9))
    ax.set_xticks(range(K)); ax.set_xticklabels(STATE_NAMES, rotation=90); ax.set_yticks(range(K)); ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel('to'); ax.set_ylabel('from'); ax.set_title(title)
    if K <= 10:
        for i in range(K):
            for j in range(K):
                if mat[i, j] >= 0.05:
                    ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if mat[i, j] > 0.5 * mat.max() else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()
self_p = np.diag(M)
pd.DataFrame({'stay probability': self_p, 'expected bout (s)': 1 / np.maximum(1 - self_p, 1e-9) / fps,
              'most common next state': [STATE_NAMES[int(np.argmax(Tn[i]))] if Tn[i].sum() else '–' for i in range(K)]},
             index=STATE_NAMES)

# %% [markdown]
# ## 7. The feature space
# How much variance each principal component carries, and which body parts drive them. Node
# importance sums each node's squared loadings over its axes and the kept PCs, weighted by each PC's
# share of the variance.

# %%
evr = np.array(red['explained_variance_ratio'])
npc = red['n_components'] if red['method'] == 'pca' else len(red.get('loadings', []))
names = feat.get('feature_names', [])
L = np.array(red.get('loadings', []))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
shown = min(len(evr), max(npc + 3, 8))
axes[0].bar(range(1, shown + 1), evr[:shown] * 100, color=['#8fb5a8' if i < npc else '#c8c8c8' for i in range(shown)])
axes[0].plot(range(1, shown + 1), np.cumsum(evr[:shown]) * 100, '-o', color='black', ms=3)
axes[0].set_xlabel('PC'); axes[0].set_ylabel('% variance'); axes[0].set_title('Variance explained (kept PCs in green)')
if len(L) and names:
    w = evr[:len(L)][:, None]
    contrib = (L ** 2 * w).sum(axis=0)
    node_imp = {}
    for nm, v in zip(names, contrib):
        who, node, _ = nm.split('/')
        key = node if who == 'own' else f'{node} (other)'
        node_imp[key] = node_imp.get(key, 0) + v
    s = pd.Series(node_imp).sort_values(ascending=True)
    s = s / s.sum() * 100
    s.tail(15).plot.barh(ax=axes[1], color='#8fb5a8')
    axes[1].set_xlabel('% of weighted loading'); axes[1].set_title('Node importance in the kept PCs')
plt.tight_layout(); plt.show()
if len(L) and names:
    top = {}
    for i in range(min(3, len(L))):
        order = np.argsort(-np.abs(L[i]))[:6]
        top[f'PC{i + 1} ({evr[i] * 100:.0f}%)'] = [f"{names[j]} ({L[i, j]:+.2f})" for j in order]
    display(pd.DataFrame(top))

# %% [markdown]
# ## 8. States in PC space
# Each frame in the first two PCs, colored by state, then one panel per state with its flow field
# (the step the state's linear dynamics predicts from each point, other PCs held at the state's mean)
# and its longest bouts. ⊗ marks the field's fixed point, where its trajectories converge or circle.

# %%
emb = {a['id']: np.asarray(a['embedding'], float) for a in animals}
allE = np.vstack(list(emb.values()))
allL = np.concatenate([LABELS[a['id']] for a in animals])
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(allE[:, 0], allE[:, 1] if allE.shape[1] > 1 else np.zeros(len(allE)), c=allL - 1, cmap=CMAP,
           vmin=-0.5, vmax=K - 0.5, s=4, alpha=0.7, linewidths=0)
ax.set_xlabel(f'PC1 ({evr[0] * 100:.0f}%)'); ax.set_ylabel(f'PC2 ({evr[1] * 100:.0f}%)' if len(evr) > 1 else '')
ax.set_title('Frames in PC space'); plt.show()

has_flow = all(s.get('A') is not None for s in states) and allE.shape[1] >= 2
cols = 3 if K > 4 else 2
rows_n = int(np.ceil(K / cols))
fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 2.9 * rows_n), squeeze=False)
x0, x1 = np.percentile(allE[:, 0], [0.5, 99.5]); y0, y1 = np.percentile(allE[:, 1], [0.5, 99.5])
px, py = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
gx, gy = np.meshgrid(np.linspace(x0, x1, 11), np.linspace(y0, y1, 9))
for c in range(K):
    ax = axes[c // cols][c % cols]
    ax.scatter(allE[:, 0], allE[:, 1], s=1, color='#bbbbbb', alpha=0.35, linewidths=0)
    members = allL == c + 1
    if has_flow and members.any():
        Am, bm = np.array(states[c]['A']), np.array(states[c]['b'])
        mu = allE[members].mean(axis=0)
        Z = np.tile(mu, (gx.size, 1)); Z[:, 0] = gx.ravel(); Z[:, 1] = gy.ravel()
        V = Z @ Am.T + bm - Z
        ax.quiver(gx.ravel(), gy.ravel(), V[:, 0], V[:, 1], color='#777777', angles='xy', width=0.004, alpha=0.8)
        fp = states[c].get('fixed_point')
        if fp is not None and x0 <= fp[0] <= x1 and y0 <= fp[1] <= y1:
            ax.plot(fp[0], fp[1], marker='$\\otimes$', ms=11, color='black')
    longest = sorted([b for a in animals for b in a['bouts'] if b['state'] == c + 1 and b['end'] > b['start']],
                     key=lambda b: b['end'] - b['start'], reverse=True)[:15]
    for a in animals:
        for b in a['bouts']:
            if b in longest:
                seg = emb[a['id']][b['start']:b['end'] + 1]
                ax.plot(seg[:, 0], seg[:, 1], color=COLORS[c], lw=1.2, alpha=0.9)
                ax.plot(seg[0, 0], seg[0, 1], 'o', color=COLORS[c], ms=2.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{STATE_NAMES[c]} · {share.iloc[c] * 100:.0f}% of time", color=COLORS[c], fontsize=10)
for i in range(K, rows_n * cols):
    axes[i // cols][i % cols].axis('off')
if not has_flow:
    fig.suptitle('Model fit on raw features: flow fields are not in PC space, bouts only', fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9. Behaviour in each state
# Distributions of the per-frame measures by state (boxes: quartiles; whiskers: 1.5 × IQR), then the
# median of each measure per state, and the same as a z-score against all frames so states can be
# compared at a glance.

# %%
nm = len(MEASURES)
cols = 3
rows_n = int(np.ceil(nm / cols))
fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.6 * rows_n), squeeze=False)
for i, mname in enumerate(MEASURES):
    ax = axes[i // cols][i % cols]
    data = [df.loc[df.state == c, mname].values for c in range(1, K + 1)]
    bp = boxplot(ax, data, STATE_NAMES, patch_artist=True, showfliers=False, medianprops=dict(color='black'))
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_title(f"{mname.replace('_', ' ')} ({UNITS[mname]})")
    ax.tick_params(axis='x', labelsize=7, rotation=90 if K > 8 else 0)
for i in range(nm, rows_n * cols):
    axes[i // cols][i % cols].axis('off')
plt.tight_layout(); plt.show()

med = df.groupby('state')[MEASURES].median().reindex(range(1, K + 1))
med.index = STATE_NAMES
z = (med - df[MEASURES].median()) / df[MEASURES].std().replace(0, np.nan)
fig, ax = plt.subplots(figsize=(1.0 + 0.9 * nm, 0.4 * K + 1.2))
im = ax.imshow(z.values, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
ax.set_xticks(range(nm)); ax.set_xticklabels([m.replace('_', ' ') for m in MEASURES], rotation=40, ha='right')
ax.set_yticks(range(K)); ax.set_yticklabels(STATE_NAMES)
for i in range(K):
    for j in range(nm):
        if np.isfinite(z.values[i, j]):
            ax.text(j, i, f'{z.values[i, j]:+.1f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, label='median vs all frames (z)')
ax.set_title('How each state differs (median, z-scored)')
plt.tight_layout(); plt.show()
med.round(2)

# %% [markdown]
# ## 10. Average pose in each state
# Every frame expressed in the animal's own frame (centroid at the origin, body axis pointing right)
# and averaged per state. Top: seen from above. Bottom (3D data): seen from the side.

# %%
def egocentric(p):
    cen = p.mean(axis=1, keepdims=True)
    v = p[:, ANT, [0, 2]] - p[:, POST, [0, 2]]
    th = np.arctan2(v[:, 1], v[:, 0])[:, None]
    dx, dz = p[..., 0] - cen[..., 0], p[..., 2] - cen[..., 2]
    fw = dx * np.cos(th) + dz * np.sin(th)
    lat = -dx * np.sin(th) + dz * np.cos(th)
    return np.stack([fw, lat, p[..., 1]], -1)            # forward, lateral, height


EGO = np.concatenate([egocentric(P[a['id']]) for a in animals])
views = 2 if dims == 3 else 1
fig, axes = plt.subplots(views, K, figsize=(1.9 * K + 0.5, 1.9 * views + 0.4), squeeze=False)
lim = np.nanpercentile(np.abs(EGO[..., :2]), 99) * 1.1
hlim = np.nanpercentile(EGO[..., 2], 99.5) * 1.1 if dims == 3 else 1
for c in range(K):
    sel = allL == c + 1
    for v in range(views):
        ax = axes[v][c]
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if not sel.any():
            continue
        mp = EGO[sel].mean(axis=0)
        yy = mp[:, 1] if v == 0 else mp[:, 2]
        for i, j in bones:
            ax.plot([mp[i, 0], mp[j, 0]], [yy[i], yy[j]], color=COLORS[c], lw=1.6)
        ax.plot(mp[:, 0], yy, 'o', color=COLORS[c], ms=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim) if v == 0 else ax.set_ylim(-0.1 * hlim, hlim)
        if v == 0:
            ax.set_title(STATE_NAMES[c], color=COLORS[c], fontsize=10)
axes[0][0].set_ylabel('top view')
if views == 2:
    axes[1][0].set_ylabel('side view')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11. Summary
# A plain-language description of each state from the numbers above: its share of time, typical bout
# length, what usually follows it, and the measures where its median is well above or below the
# median over all frames (|z| ≥ 0.5).

# %%
lines = []
order = np.argsort(-share.values)
for c in order:
    nmz = STATE_NAMES[c]
    zrow = z.loc[nmz]
    hi = [m.replace('_', ' ') for m in MEASURES if np.isfinite(zrow[m]) and zrow[m] >= 0.5]
    lo = [m.replace('_', ' ') for m in MEASURES if np.isfinite(zrow[m]) and zrow[m] <= -0.5]
    desc = []
    if hi:
        desc.append('higher ' + ', '.join(hi))
    if lo:
        desc.append('lower ' + ', '.join(lo))
    bt = bout_table.loc[nmz] if nmz in bout_table.index else None
    nxt = STATE_NAMES[int(np.argmax(Tn[c]))] if Tn[c].sum() else None
    txt = (f"**{nmz}** — {share.iloc[c] * 100:.1f}% of time"
           + (f", {int(bt['bouts'])} bouts, median {bt['median_s']:.2f} s" if bt is not None else '')
           + (f"; {' and '.join(desc)} than average" if desc else '; close to average on every measure')
           + (f"; usually followed by {nxt}" if nxt else '') + '.')
    lines.append(txt)
display(Markdown('\n\n'.join(lines)))

# %% [markdown]
# ## 12. Tables for further analysis
# `df` holds one row per animal per frame with its state and every measure; `bouts` one row per
# bout. Uncomment to save them next to this notebook.

# %%
# df.to_csv('per_frame.csv', index=False)
# bouts.to_csv('bouts.csv', index=False)
print(f'df: {df.shape[0]} rows × {df.shape[1]} columns; bouts: {len(bouts)} rows')
