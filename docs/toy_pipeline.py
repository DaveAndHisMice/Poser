"""
Toy walk-through of the viewer's analysis pipeline (see README.md, "Methodology").

Every number quoted in the README's worked examples is printed by this script, and every figure in
docs/figures/ is drawn by it. The maths mirrors viewer/viewer_template.html:

  1. gap filling            fillGaps()
  2. centroid and heading   prepare(), egocentricFeatures()
  3. egocentric pose        egocentricFeatures()
  4. social block           buildFeatures()  (rotation onto the axis, nearest-first, Gaussian kernel)
  5. PCA                    runPCA()          (keep the PCs reaching 95% of the variance)
  6. switching linear       arhmm()           (EM: forward-backward + weighted least squares,
     dynamics + BIC                            sticky transitions, Viterbi labels, BIC over k)

Run:  python docs/toy_pipeline.py      (needs numpy and matplotlib)
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=2, suppress=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
C_FOCAL, C_OTHER, C_THIRD, C_MUTED = "#3f8f7a", "#c07a3a", "#5a6fb0", "#9aa0a6"
STATE_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]


def h(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# ---------------------------------------------------------------- step 1: gap filling
def fill_gaps(series):
    """Linear between known samples, held flat before the first and after the last (fillGaps)."""
    s = [None if v is None else np.asarray(v, float) for v in series]
    known = [i for i, v in enumerate(s) if v is not None]
    out = list(s)
    for i in range(len(s)):
        if s[i] is not None:
            continue
        before = [k for k in known if k < i]
        after = [k for k in known if k > i]
        if not before:
            out[i] = s[after[0]]
        elif not after:
            out[i] = s[before[-1]]
        else:
            a, b = before[-1], after[0]
            out[i] = s[a] + (s[b] - s[a]) * (i - a) / (b - a)
    return out


h("STEP 1  gap filling: one joint (snout) of one animal over 6 frames, (x, y, z) in cm")
snout = [None, (6.0, 1.0, 7.0), None, None, (9.0, 1.0, 4.0), None]
for f, (raw, filled) in enumerate(zip(snout, fill_gaps(snout))):
    print(f"frame {f}: raw {str(raw):18s} -> filled ({filled[0]:.1f}, {filled[1]:.1f}, {filled[2]:.1f})")

# ---------------------------------------------------------------- toy scene, one frame
# Three-joint mice (pelvis, neck, snout), 3D, Y up, cm. The body axis runs pelvis -> neck.
JOINTS = ["pelvis", "neck", "snout"]
focal = np.array([[2.0, 1.5, 1.0], [5.0, 2.0, 5.0], [6.8, 1.0, 7.4]])        # heading 53.1 deg
other = np.array([[13.0, 1.5, 4.0], [10.0, 2.0, 4.0], [8.0, 1.2, 4.0]])      # faces the focal snout
third = np.array([[-6.0, 5.0, 9.0], [-6.0, 5.5, 12.0], [-6.0, 4.5, 14.0]])   # on a 3.5 cm block
ANIMALS = {"focal": focal, "other": other, "third": third}


def centroid(P):
    """Mean of the joints, on the floor (y = 0) like the viewer; plus the mean height."""
    c = P.mean(0)
    return np.array([c[0], 0.0, c[2]]), c[1]


def heading(P):
    a, p = P[1], P[0]                    # anterior = neck, posterior = pelvis
    return np.arctan2(a[2] - p[2], a[0] - p[0])


def ego(P, c, th):
    """(forward, up, lateral) of each joint in the frame of an animal with centroid c, heading th."""
    cs, sn = np.cos(th), np.sin(th)
    dx, dz = P[:, 0] - c[0], P[:, 2] - c[2]
    return np.stack([dx * cs + dz * sn, P[:, 1], -dx * sn + dz * cs], -1)


h("STEP 2  centroid and heading of each animal")
for name, P in ANIMALS.items():
    c, hgt = centroid(P)
    print(f"{name:6s} centroid (x, z) = ({c[0]:.2f}, {c[2]:.2f})  mean height {hgt:.2f}  "
          f"heading {np.degrees(heading(P)):.1f} deg")

h("STEP 3  egocentric pose of the focal animal: (forward, up, lateral) per joint")
cF, hF = centroid(focal)
thF = heading(focal)
E = ego(focal, cF, thF)
for j, row in zip(JOINTS, E):
    print(f"{j:7s} {row}")
print("feature vector (own block, 9 numbers):", E.reshape(-1))

# ---------------------------------------------------------------- step 4: social block
h("STEP 4  social block: other animals in the focal frame")


def bearing(cb, c, th):
    bx, bz = cb[0] - c[0], cb[2] - c[2]
    cs, sn = np.cos(th), np.sin(th)
    return np.arctan2(-bx * sn + bz * cs, bx * cs + bz * sn)


def wrap_front_back(phi):
    """Smallest rotation that puts the other animal on the body axis, in front or behind: (-90, 90]."""
    if phi > np.pi / 2:
        return phi - np.pi
    if phi <= -np.pi / 2:
        return phi + np.pi
    return phi


BODY = 11.0           # body length (nose-to-tail extent), cm, for d0 in body lengths
D0 = 1.0 * BODY
rows = []
for name in ("other", "third"):
    P = ANIMALS[name]
    cb, hb = centroid(P)
    d2 = np.hypot(cb[0] - cF[0], cb[2] - cF[2])
    d3 = np.sqrt(d2 ** 2 + (hb - hF) ** 2)
    phi = bearing(cb, cF, thF)
    rows.append((name, P, cb, d2, d3, phi))
rows.sort(key=lambda r: r[4])           # nearest-first by the 3D distance
for slot, (name, P, cb, d2, d3, phi) in enumerate(rows, 1):
    w2, w3 = np.exp(-0.5 * (d2 / D0) ** 2), np.exp(-0.5 * (d3 / D0) ** 2)
    rot = wrap_front_back(phi)
    print(f"slot other{slot} = {name}: d_2D {d2:.2f}  d_3D {d3:.2f}  w_2D {w2:.3f}  w_3D {w3:.3f}  "
          f"bearing {np.degrees(phi):.1f} deg -> rotate by {np.degrees(-rot):.1f} deg "
          f"({'in front' if abs(phi) <= np.pi / 2 else 'behind'})")
    Eo = ego(P, cF, thF)
    cr, sr = np.cos(rot), np.sin(rot)
    aligned = np.stack([Eo[:, 0] * cr + Eo[:, 2] * sr, Eo[:, 1], -Eo[:, 0] * sr + Eo[:, 2] * cr], -1)
    for j, a, b in zip(JOINTS, Eo, aligned):
        print(f"    {j:7s} focal frame {a}  rotated {b}  x w_3D {b * w3}")

# figure: egocentric transform and social alignment
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 0.8, 1.3]})


def draw(axh, P2, color, label, ls="-"):
    axh.plot(P2[:, 0], P2[:, 1], ls, color=color, lw=2, marker="o", ms=4, label=label)
    axh.plot(P2[2, 0], P2[2, 1], marker="^", color=color, ms=7)


for name, P, col in (("focal", focal, C_FOCAL), ("other", other, C_OTHER), ("third", third, C_THIRD)):
    draw(ax[0], P[:, [0, 2]], col, name)
    c, _ = centroid(P)
    ax[0].plot(c[0], c[2], "x", color=col, ms=7)
ax[0].annotate("", xy=(cF[0] + 4 * np.cos(thF), cF[2] + 4 * np.sin(thF)), xytext=(cF[0], cF[2]),
               arrowprops=dict(arrowstyle="->", color=C_FOCAL, lw=1.5))
ax[0].set_title("1  world (top-down, x-z)\n× = centroid, ▲ = snout, arrow = heading")
ax[0].set_xlabel("x"); ax[0].set_ylabel("z"); ax[0].set_aspect("equal"); ax[0].legend(frameon=False, fontsize=7)
Ef = ego(focal, cF, thF)
draw(ax[1], Ef[:, [0, 2]], C_FOCAL, "focal")
for name, P, col in (("other", other, C_OTHER), ("third", third, C_THIRD)):
    draw(ax[1], ego(P, cF, thF)[:, [0, 2]], col, name)
ax[1].axhline(0, color=C_MUTED, lw=0.6); ax[1].axvline(0, color=C_MUTED, lw=0.6)
ax[1].set_title("2  focal frame\ncentroid at 0")
ax[1].set_xlabel("forward"); ax[1].set_ylabel("lateral"); ax[1].set_aspect("equal")
draw(ax[2], Ef[:, [0, 2]], C_FOCAL, "focal")
for name, P, cb, d2, d3, phi in rows:
    rot = wrap_front_back(phi)
    Eo = ego(P, cF, thF)
    cr, sr = np.cos(rot), np.sin(rot)
    al = np.stack([Eo[:, 0] * cr + Eo[:, 2] * sr, -Eo[:, 0] * sr + Eo[:, 2] * cr], -1)
    w3 = np.exp(-0.5 * (d3 / D0) ** 2)
    col = C_OTHER if name == "other" else C_THIRD
    draw(ax[2], al, col, name, ls="--")
    draw(ax[2], al * w3, col, f"{name} × w = {w3:.2f}")
ax[2].axhline(0, color=C_MUTED, lw=0.6); ax[2].axvline(0, color=C_MUTED, lw=0.6)
ax[2].set_title("3  onto the body axis (dashed)\n× proximity weight (solid)")
ax[2].set_xlabel("forward"); ax[2].set_ylabel("lateral"); ax[2].set_ylim(-4, 4)
ax[2].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
fig.savefig(FIG / "1_egocentric_social.png"); plt.close(fig)

# figure: the kernel in 2D vs 3D
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
d = np.linspace(0, 3 * D0, 200)
ax[0].plot(d / BODY, np.exp(-0.5 * (d / D0) ** 2), color=C_FOCAL, lw=2)
for name, P, cb, d2, d3, phi in rows:
    col = C_OTHER if name == "other" else C_THIRD
    ax[0].plot(d2 / BODY, np.exp(-0.5 * (d2 / D0) ** 2), "o", mfc="white", color=col, label=f"{name}, 2D")
    ax[0].plot(d3 / BODY, np.exp(-0.5 * (d3 / D0) ** 2), "o", color=col, label=f"{name}, 3D")
ax[0].axvline(1, color=C_MUTED, lw=0.6, ls=":")
ax[0].set_xlabel("centroid-to-centroid distance (body lengths)"); ax[0].set_ylabel("weight w(d)")
ax[0].set_title("w(d) = exp(-½ (d/d₀)²),  d₀ = 1 body"); ax[0].legend(frameon=False, fontsize=7)
g = np.linspace(-2.2 * D0, 2.2 * D0, 161)
X, Y = np.meshgrid(g, g)
ax[1].contourf(X / BODY, Y / BODY, np.exp(-0.5 * (X ** 2 + Y ** 2) / D0 ** 2), levels=12, cmap="Greens")
ax[1].set_aspect("equal"); ax[1].set_xlabel("Δx or Δfloor (body lengths)"); ax[1].set_ylabel("Δheight (body lengths)")
ax[1].set_title("isotropic: same width along every axis\n(3D adds height; 2D ignores it)")
fig.savefig(FIG / "2_proximity_kernel.png"); plt.close(fig)

# ---------------------------------------------------------------- steps 5-6: dynamics toy
h("STEP 5  PCA: 600 frames of a toy egocentric pose (9 features) driven by 2 hidden coordinates")
rng = np.random.default_rng(7)
T = 600


def rot(theta, r=1.0):
    return r * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


# two behaviours as affine flow fields z(t+1) = A z(t) + b + noise
A_true = [rot(0.25, 0.985), 0.90 * np.eye(2)]          # 0: scanning (slow spiral), 1: rearing (settle at z*)
z_star = np.array([2.5, 0.0])
b_true = [np.zeros(2), (np.eye(2) - A_true[1]) @ z_star]
P_true = np.array([[0.97, 0.03], [0.04, 0.96]])
state = np.zeros(T, int)
z = np.zeros((T, 2))
z[0] = (1.2, 0.0)
for t in range(1, T):
    state[t] = rng.choice(2, p=P_true[state[t - 1]])
    s = state[t - 1]
    z[t] = A_true[s] @ z[t - 1] + b_true[s] + rng.normal(0, 0.06, 2)
# pose = mean pose + loadings · z + tracking noise; z1 lifts and pushes the head forward (rearing),
# z2 swings the neck and snout sideways (scanning)
mean_pose = E.reshape(-1)
Wl = np.zeros((9, 2))
Wl[[3, 4, 6, 7], 0] = [0.6, 1.2, 1.0, 2.0]             # neck/snout forward and up
Wl[[5, 8], 1] = [1.0, 2.2]                             # neck/snout lateral
Xp = mean_pose + z @ Wl.T + rng.normal(0, 0.12, (T, 9))
Xc = Xp - Xp.mean(0)
evals, evecs = np.linalg.eigh(np.cov(Xc.T))
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
for k in range(evecs.shape[1]):                         # fixed sign: largest loading positive
    if evecs[np.argmax(np.abs(evecs[:, k])), k] < 0:
        evecs[:, k] *= -1
ratio = evals / evals.sum()
cum = np.cumsum(ratio)
n95 = int(np.argmax(cum >= 0.95)) + 1
print("variance ratio per PC:", np.round(ratio, 3))
print("cumulative           :", np.round(cum, 3), f"-> keep {n95} PCs (first to reach 95%)")
feat_names = [f"own/{j}/{a}" for j in JOINTS for a in ("forward", "up", "lateral")]
for k in range(2):
    top = np.argsort(-np.abs(evecs[:, k]))[:3]
    print(f"PC{k + 1} strongest loadings:", ", ".join(f"{feat_names[i]} {evecs[i, k]:+.2f}" for i in top))
Z = Xc @ evecs[:, :n95]

fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9))
ax[0].bar(np.arange(1, 10), ratio, color=[C_FOCAL if i < n95 else C_MUTED for i in range(9)])
ax[0].plot(np.arange(1, 10), cum, "-o", color="#333", ms=3)
ax[0].axhline(0.95, ls="--", color=C_MUTED, lw=0.8)
ax[0].set_xlabel("PC"); ax[0].set_ylabel("variance explained"); ax[0].set_title(f"scree: keep {n95} PCs (≥ 95%)")
im = ax[1].imshow(evecs[:, :n95].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax[1].set_yticks(range(n95), [f"PC{k + 1}" for k in range(n95)])
ax[1].set_xticks(range(9), feat_names, rotation=60, ha="right", fontsize=7)
ax[1].set_title("loadings of the kept PCs"); fig.colorbar(im, ax=ax[1], fraction=0.03)
fig.savefig(FIG / "3_pca.png"); plt.close(fig)


# ---------------------------------------------------------------- step 6: switching LDS (EM)
def kmeans(F, k, rng, iters=100):
    cen = F[rng.choice(len(F), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((F[:, None] - cen[None]) ** 2).sum(-1), 1)
        new = np.array([F[lab == c].mean(0) if (lab == c).any() else cen[c] for c in range(k)])
        if np.allclose(new, cen):
            break
        cen = new
    return lab


def arhmm(Z, k, sticky=0.1, iters=80, seed=0):
    """Same model and steps as the viewer's arhmm(): pairs (z_t, z_t+1); per state a weighted least
    squares fit of [A | b] and a residual covariance; sticky transition counts; forward-backward;
    Viterbi; BIC = -2 log L + params · log(pairs)."""
    x, y = Z[:-1], Z[1:]
    n, m = x.shape
    X1 = np.hstack([x, np.ones((n, 1))])
    v = y - x
    reg = 1e-4 * v.var(0).mean()
    feat = np.hstack([x / np.sqrt((x ** 2).sum(1).mean()), v / np.sqrt((v ** 2).sum(1).mean())])
    R = np.eye(k)[kmeans(feat, k, np.random.default_rng(seed))]
    P = np.full((k, k), 0.1 / max(k - 1, 1)) + np.eye(k) * (0.9 - 0.1 / max(k - 1, 1))
    pi = np.full(k, 1 / k)
    kappa = sticky * n / k
    prev = -np.inf
    for it in range(iters):
        # M-step: weighted least squares per state
        W, S = [], []
        for c in range(k):
            r = R[:, c]
            G = (X1 * r[:, None]).T @ X1
            G += 1e-8 * np.trace(G) / (m + 1) * np.eye(m + 1)
            Wc = np.linalg.solve(G, (X1 * r[:, None]).T @ y)
            res = y - X1 @ Wc
            Sc = (res * r[:, None]).T @ res / max(r.sum(), 1e-9) + reg * np.eye(m)
            W.append(Wc); S.append(Sc)
        # emissions
        logE = np.zeros((n, k))
        for c in range(k):
            res = y - X1 @ W[c]
            Si = np.linalg.inv(S[c])
            logE[:, c] = -0.5 * (m * np.log(2 * np.pi) + np.linalg.slogdet(S[c])[1] + np.einsum("ij,jk,ik->i", res, Si, res))
        # forward-backward (scaled)
        mx = logE.max(1, keepdims=True)
        B = np.exp(logE - mx)
        al = np.zeros((n, k)); cs = np.zeros(n)
        for t in range(n):
            a = (pi if t == 0 else al[t - 1] @ P) * B[t]
            cs[t] = a.sum(); al[t] = a / cs[t]
        be = np.ones((n, k))
        for t in range(n - 2, -1, -1):
            be[t] = (P @ (B[t + 1] * be[t + 1])) / cs[t + 1]
        ll = np.log(cs).sum() + mx.sum()
        R = al * be; R /= R.sum(1, keepdims=True)
        xi = np.zeros((k, k))
        for t in range(1, n):
            xi += np.outer(al[t - 1], B[t] * be[t]) * P / cs[t]
        P = xi + kappa * np.eye(k) + 1e-3
        P /= P.sum(1, keepdims=True)
        pi = (R[0] + 1e-3) / (R[0] + 1e-3).sum()
        if abs(ll - prev) < 1e-5 * max(1, abs(ll)):
            break
        prev = ll
    # Viterbi
    lP = np.log(P)
    dl = np.log(pi) + logE[0]
    bp = np.zeros((n, k), int)
    for t in range(1, n):
        cand = dl[:, None] + lP
        bp[t] = cand.argmax(0)
        dl = cand.max(0) + logE[t]
    path = np.zeros(n, int); path[-1] = dl.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    labels = np.append(path, path[-1])               # last frame copies the one before (non-looping)
    params = k * (m * (m + 1) + m * (m + 1) / 2) + k * (k - 1) + (k - 1)
    bic = -2 * ll + params * np.log(n)
    return dict(labels=labels, W=W, S=S, P=P, ll=ll, bic=bic, params=params, iters=it + 1)


h("STEP 6  switching linear dynamics on the kept PCs: BIC over k")
fits = {}
for k in range(1, 5):
    fits[k] = arhmm(Z, k)
    f = fits[k]
    print(f"k = {k}: log L {f['ll']:9.1f}  params {int(f['params']):3d}  BIC {f['bic']:9.1f}  (EM iterations {f['iters']})")
kbest = min(fits, key=lambda k: fits[k]["bic"])
best = fits[kbest]
print(f"-> lowest BIC at k = {kbest}")
# match fitted states to the true ones for reporting
lab = best["labels"]
agree = max(np.mean(lab == state), np.mean(lab == 1 - state)) if kbest == 2 else float("nan")
print(f"agreement with the true behaviour labels: {agree * 100:.1f}%")
for c in range(kbest):
    Wc = best["W"][c]
    A, b = Wc[:2].T, Wc[2]
    eig = np.linalg.eigvals(A)
    try:
        fp = np.linalg.solve(np.eye(2) - A, b)
    except np.linalg.LinAlgError:
        fp = None
    print(f"state {c + 1}: A =\n{A}\n  b = {b}  eigenvalues {np.round(eig, 3)} "
          f"(|eig| {np.round(np.abs(eig), 3)}, rotation {np.degrees(np.abs(np.angle(eig[0]))):.1f} deg/frame)  "
          f"fixed point z* = {None if fp is None else np.round(fp, 2)}")
print("transition matrix:\n", np.round(best["P"], 3))
bouts = np.flatnonzero(np.diff(lab)) + 1
print(f"{len(bouts) + 1} bouts; mean bout {T / (len(bouts) + 1):.1f} frames")

# figure: flow fields per state + ethogram
fig = plt.figure(figsize=(10.5, 4.6))
gs = fig.add_gridspec(2, kbest + 1, height_ratios=[3, 0.55], hspace=0.45)
lo, hi = Z.min(0) - 0.4, Z.max(0) + 0.4
gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 13), np.linspace(lo[1], hi[1], 11))
axa = fig.add_subplot(gs[0, 0])
for c in range(kbest):
    sel = lab[:-1] == c
    segs = np.stack([Z[:-1][sel], Z[1:][sel]], 1)
    for s in segs:
        axa.plot(s[:, 0], s[:, 1], color=STATE_COLORS[c], lw=0.7, alpha=0.7)
axa.set_title("trajectory in PC1-PC2,\neach step colored by its state"); axa.set_xlabel("PC1"); axa.set_ylabel("PC2")
for c in range(kbest):
    ax = fig.add_subplot(gs[0, c + 1], sharex=axa, sharey=axa)
    Wc = best["W"][c]
    A, b = Wc[:2].T, Wc[2]
    U = A[0, 0] * gx + A[0, 1] * gy + b[0] - gx
    V = A[1, 0] * gx + A[1, 1] * gy + b[1] - gy
    ax.quiver(gx, gy, U, V, color=C_MUTED, angles="xy", scale_units="xy", scale=0.25, width=0.004)
    ax.plot(Z[:, 0], Z[:, 1], ".", color=C_MUTED, ms=1, alpha=0.3)
    idx = np.flatnonzero(lab == c)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for r in sorted(runs, key=len)[-6:]:
        ax.plot(Z[r, 0], Z[r, 1], color=STATE_COLORS[c], lw=1.2)
        ax.plot(Z[r[0], 0], Z[r[0], 1], "o", color=STATE_COLORS[c], ms=3)
    try:
        fp = np.linalg.solve(np.eye(2) - A, b)
        ax.plot(*fp, marker="X", color="k", ms=8, ls="none", label="fixed point z*")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    except np.linalg.LinAlgError:
        pass
    ax.set_title(f"state {c + 1}: flow field z(t+1) − z(t)\n= (A − I) z + b, and its bouts")
    ax.set_xlabel("PC1")
axe = fig.add_subplot(gs[1, :])
axe.imshow(np.vstack([state if agree >= np.mean(lab == 1 - state) else 1 - state, lab])[:, None, :].squeeze(1),
           aspect="auto", cmap=matplotlib.colors.ListedColormap(STATE_COLORS[:max(kbest, 2)]), interpolation="nearest")
axe.set_yticks([0, 1], ["true", "Viterbi"]); axe.set_xlabel("frame")
axe.set_title("ethogram: hidden behaviour used to simulate vs. states recovered from the pose alone", fontsize=8)
fig.savefig(FIG / "4_flow_fields.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.6, 2.4))
ks = sorted(fits)
ax.plot(ks, [fits[k]["bic"] for k in ks], "-o", color=C_FOCAL)
ax.axvline(kbest, ls="--", color=C_MUTED)
ax.set_xlabel("k (number of states)"); ax.set_ylabel("BIC (lower = better)"); ax.set_xticks(ks)
ax.set_title("choosing k")
fig.savefig(FIG / "5_bic.png"); plt.close(fig)
print("\nfigures written to", FIG)
