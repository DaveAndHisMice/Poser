"""
Synthesize a noisy 3D dataset of four mice in an open field with blocks they climb.

Writes mouse_group_blocks.json (etho-pose 1.0, see viewer/POSE_FORMAT.md) next to this script.

Scene: an 80 cm open field with three blocks (a 3.5 cm platform, a 5.5 cm tall block and a 2.5 cm
step bar). Four mice spend 60 s starting and ending in a huddle, and in between show as many kinds
of interaction as possible:

  huddling (all four), nose-to-nose, flank sniffing while circling, anogenital sniffing while
  following, fighting (circling and boxing), chasing (the fleeing mouse escapes over a block),
  crawling over a resting mouse, walking side by side, sniffing across the tall block (one mouse
  on top peering down, one rearing up from the floor), approach and avoidance, allogrooming, and a
  three-mouse train that climbs over the step bar,

plus solo behaviour: exploring, climbing up and down every block, rearing (on the floor, against
a wall, on top of a block), self-grooming and stretched-attend.

Bodies come from generate_mouse_datasets.mouse_frames (same 18-joint skeleton, Y up, metres, origin
at the arena centre). Joints are then lifted onto the blocks: the hindquarters follow the surface
under the pelvis, the forequarters the surface under the neck, the spine sits between them and the
paws and tail rest on whatever is under them. A crawling mouse is lifted over the other's back.

Tracking noise, applied after every geometric check:
  * white noise on every coordinate (sd 2.5 mm; 4.5 mm on paws and the tail tip)
  * slowly drifting error per joint (AR(1), sd 1.5 mm)
  * brief outlier jumps (a joint detected 1-4 cm away for 1-3 frames)
  * left/right swaps of ears or forepaws for a few frames
  * missing samples, written as null: short random dropouts (most often paws and the tail tip),
    occlusion where another mouse covers a joint from above, hind paws hidden while rearing, and a
    few frames where a whole animal is lost.

The clip loops: every mouse ends in the huddle spot it started from.
"""
import json
from pathlib import Path

import numpy as np

import generate_mouse_datasets as g

FPS = g.FPS
SECONDS = 60.0
N = int(round(SECONDS * FPS))
T = np.arange(N) / FPS
HALF = 0.40                       # arena is 80 cm square, centred on the origin
OUT = Path(__file__).resolve().parent / "mouse_group_blocks.json"
JI = {name: i for i, (name, _) in enumerate(g.MOUSE_JOINTS)}
NAMES = ["mouse 1", "mouse 2", "mouse 3", "mouse 4"]

# ------------------------------------------------------------------ arena

BLOCKS = [   # centre x, centre z, size x, size z, height (m)
    dict(name="platform", x=-0.22, z=-0.20, sx=0.14, sz=0.14, h=0.035),
    dict(name="tall block", x=0.22, z=0.20, sx=0.12, sz=0.12, h=0.055),
    dict(name="step bar", x=0.20, z=-0.18, sx=0.22, sz=0.07, h=0.025),
]
EDGE = 0.012   # the surface eases from floor to block top over this width (m) around each edge


def surface(x, z):
    """Height of whatever is under (x, z): the floor (0) or a block top, eased across each edge."""
    h = np.zeros(np.broadcast(x, z).shape)
    for b in BLOCKS:
        x0, x1 = b["x"] - b["sx"] / 2, b["x"] + b["sx"] / 2
        z0, z1 = b["z"] - b["sz"] / 2, b["z"] + b["sz"] / 2
        s = (g.smoothstep((x - x0) / EDGE + 0.5) * g.smoothstep((x1 - x) / EDGE + 0.5)
             * g.smoothstep((z - z0) / EDGE + 0.5) * g.smoothstep((z1 - z) / EDGE + 0.5))
        h = np.maximum(h, b["h"] * s)
    return h


# ------------------------------------------------------------------ path building


def W(a, b, ramp=0.3):
    """0..1 per frame: 1 inside [a, b] seconds, smooth ramps of `ramp` s at both ends."""
    return g._window(T, a, b, ramp)


def keys(*k):
    """Path through (t, x, z) keyframes: linear between keys, wrapping at the clip end, then
    lightly smoothed so moves ease in and out. Repeat a position to hold it."""
    k = sorted(k)
    kt = np.array([q[0] for q in k] + [k[0][0] + SECONDS])
    kx = np.array([q[1] for q in k] + [k[0][1]])
    kz = np.array([q[2] for q in k] + [k[0][2]])
    xy = np.stack([np.interp(T, kt, kx, period=None), np.interp(T, kt, kz)], -1)
    return smooth_path(xy, 5)


def smooth_path(xy, sigma):
    return np.stack([g.periodic_smooth(xy[:, i], sigma) for i in range(2)], -1)


def blend(base, other, w):
    return base * (1 - w[:, None]) + other * w[:, None]


def behind(leader, dist):
    """Where a follower is when it trails `dist` metres behind `leader` along the leader's own path."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(leader, axis=0), axis=1))])
    tau = np.interp(arc - dist, arc, np.arange(N))
    return np.stack([np.interp(tau, np.arange(N), leader[:, i]) for i in range(2)], -1)


def unit(a):
    return np.stack([np.cos(a), np.sin(a)], -1)


def heading(xy, target=None, face=None):
    """Heading (rad, from +X toward +Z) from the direction of travel, held while still, turned to
    face `target` (per-frame xz) where `face` (0..1 per frame) says so."""
    if target is None:
        target, face = xy, np.zeros(N)
    return g._heading(xy, target, face, T)[0]


def at(sec):
    return int(round(sec * FPS))


# ------------------------------------------------------------------ choreography

SCRIPT = []   # (start s, end s, animals, what): also written into the file


def note(a, b, who, what):
    SCRIPT.append((a, b, who, what))


def choreograph():
    """Per-mouse floor paths (pelvis xz), facing targets and behaviour weights."""
    face_t = [None] * 4
    face_w = [np.zeros(N) for _ in range(4)]
    beh = [{k: np.zeros(N) for k in ("rear", "groom", "stretch", "peer", "rest", "sniff", "agit")} for _ in range(4)]

    def add(m, kind, a, b, ramp=0.3):
        beh[m][kind] = np.maximum(beh[m][kind], W(a, b, ramp))

    def face(m, target, a, b, ramp=0.3):
        nonlocal face_t
        if face_t[m] is None:
            face_t[m] = np.zeros((N, 2))
        w = W(a, b, ramp)
        on = w > 0
        face_t[m][on] = target[on]
        face_w[m] = np.maximum(face_w[m], w)

    # huddle spots: (x, z) of the pelvis; mice 1 and 3 face +X, mice 2 and 4 face -X
    H1, H2, H3, H4 = (-0.14, 0.10), (-0.06, 0.135), (-0.14, 0.17), (-0.06, 0.205)
    note(0.0, 1.0, [1, 2, 3, 4], "huddling: all four resting side by side")
    note(53.0, 60.0, [1, 2, 3, 4], "huddling: all four gather and settle back together")
    for m in range(4):
        add(m, "rest", -1.0, 1.2 + 0.5 * m)
        add(m, "rest", 55.0 + 0.2 * m, 61.0)

    # ---- fight between mice 1 and 2 (13.8-18.8): circle a shared centre half a turn apart and
    # stop twice to box (both rear, facing)
    c_f = np.array([0.0, -0.08])
    box = np.maximum(W(15.3, 16.1, 0.15), W(17.4, 18.2, 0.15))
    omega = 2 * np.pi * 1.1 * W(13.8, 18.8, 0.4) * (1 - box)
    phi = np.cumsum(omega) / FPS + np.pi / 2
    r = 0.045 + 0.004 * np.sin(2 * np.pi * 0.7 * T)
    ring2 = c_f + r[:, None] * unit(phi)
    ring1 = c_f - r[:, None] * unit(phi)
    fight = W(13.8, 18.8, 0.5)
    e1, e2 = ring1[at(18.8)], ring2[at(18.8)]   # where the fight ends

    # ---- mouse 4: tall block, rests (gets crawled over), walks with mouse 3, grooms, crosses the
    # step bar, gets allogroomed, climbs and rears on the platform, back to the huddle
    m4 = keys((0.0, *H4), (2.4, *H4), (2.9, -0.17, 0.25), (3.6, -0.05, 0.32), (4.3, 0.0, 0.20), (5.0, 0.09, 0.20),
              (6.6, 0.09, 0.20), (7.4, 0.22, 0.20), (9.2, 0.22, 0.20), (9.9, 0.22, 0.06), (10.5, 0.12, 0.13),
              (11.0, 0.05, 0.22), (14.0, 0.05, 0.22), (15.3, -0.09, 0.29), (16.6, -0.24, 0.29), (17.6, -0.30, 0.18),
              (23.8, -0.30, 0.18), (25.0, -0.30, 0.0), (26.4, -0.08, -0.32), (27.8, 0.20, -0.32),
              (29.3, 0.20, -0.06), (30.8, 0.06, 0.04), (32.6, -0.08, 0.02), (34.3, -0.08, -0.08), (35.5, 0.0, 0.0),
              (40.2, 0.0, 0.0), (41.4, -0.08, -0.10), (42.6, -0.22, -0.20), (45.0, -0.22, -0.20),
              (46.0, -0.30, -0.28), (47.6, -0.30, -0.05), (49.5, -0.20, 0.31), (51.5, 0.05, 0.31),
              (52.7, 0.07, 0.205), (54.0, *H4))
    add(3, "rear", 5.1, 6.5)
    note(4.3, 6.6, [4], "rears against the tall block's wall")
    note(6.6, 9.9, [4], "climbs the tall block, scans from the top, climbs down")
    add(3, "rest", 11.2, 14.0)
    add(3, "groom", 18.0, 23.5)
    note(17.6, 23.8, [4], "self-grooming")
    note(27.8, 29.3, [4], "crosses the step bar")
    add(3, "rest", 36.0, 40.2)
    add(3, "rear", 43.0, 44.6)
    note(41.4, 46.0, [4], "climbs the platform and rears on top")

    # ---- mouse 2: crosses the step bar, nose-to-nose, followed, fight, flees over the platform,
    # climbs the step bar end, grooms, approached by mouse 3 and avoids it, leads the train
    m2 = keys((0.0, *H2), (2.6, *H2), (3.2, -0.24, 0.10), (4.3, -0.06, -0.03), (5.2, 0.10, -0.08),
              (6.2, 0.20, -0.30), (7.0, 0.06, -0.29), (8.0, 0.0, -0.17), (10.6, 0.0, -0.17), (11.6, -0.10, -0.03),
              (12.6, 0.05, 0.02), (13.8, 0.02, -0.06),
              # 13.8-18.8 fight (overridden below); 18.8 chase start
              (18.8, *e2), (19.4, -0.10, -0.13), (20.0, -0.22, -0.20), (20.8, -0.05, -0.32),
              (21.8, 0.28, -0.31), (22.6, 0.30, -0.24), (23.4, 0.30, 0.0), (27.5, 0.30, 0.0), (29.0, 0.12, 0.08),
              (30.5, -0.10, 0.0), (32.0, -0.08, -0.28), (37.5, -0.08, -0.28), (38.0, -0.02, -0.33),
              (39.0, 0.24, -0.33), (40.5, 0.33, -0.12), (41.5, 0.30, 0.06), (42.0, 0.30, 0.06),
              (43.5, 0.05, 0.10), (44.7, -0.08, -0.04), (45.7, 0.02, -0.15), (46.6, 0.15, -0.30),
              (47.6, 0.30, -0.28), (48.6, 0.32, -0.05), (50.5, 0.10, 0.06), (51.5, 0.06, 0.135), (52.5, *H2))
    note(5.2, 6.2, [2], "crosses the step bar")
    add(1, "groom", 23.6, 27.3)
    note(23.4, 27.5, [2], "self-grooming beside the step bar")

    m2 = blend(m2, ring2, fight)
    for m in (0, 1):
        beh[m]["agit"] = np.maximum(beh[m]["agit"], W(13.8, 18.8))
        beh[m]["rear"] = np.maximum(beh[m]["rear"], box)
    note(13.8, 18.8, [1, 2], "fighting: fast circling with two boxing bouts (both rear, facing)")

    # ---- mouse 3: explores, rears in the open, crawls over resting mouse 4, walks beside it,
    # rears at the tall block to sniff mouse 1 on top, approaches mouse 2 (which avoids it),
    # follows in the train
    m3 = keys((0.0, *H3), (1.4, *H3), (2.3, 0.02, 0.17), (3.3, 0.13, 0.04), (4.6, 0.27, 0.07),
              (5.6, 0.28, -0.04), (7.4, 0.28, -0.04), (8.4, 0.10, 0.05), (9.2, 0.04, 0.14),
              (10.0, 0.02, 0.30), (10.8, 0.12, 0.31), (13.0, -0.04, 0.17), (13.8, 0.0, 0.19),
              # 14-18.6 side by side with mouse 4 (overridden below)
              (19.0, -0.24, 0.18), (20.5, -0.12, 0.06), (22.0, 0.05, 0.12), (23.5, -0.05, 0.28),
              (25.5, -0.04, 0.20), (27.0, 0.095, 0.20), (31.0, 0.095, 0.20), (32.4, -0.10, 0.10),
              (34.0, -0.20, 0.02), (35.4, -0.08, -0.09), (37.4, -0.08, -0.185), (38.4, -0.08, -0.185),
              (39.4, 0.10, -0.24), (40.6, 0.30, -0.22), (41.8, 0.32, -0.02),
              # 42-48 train (overridden), then home
              (48.6, 0.28, -0.20), (50.2, 0.0, -0.05), (51.8, -0.24, -0.04), (52.9, -0.29, 0.15), (53.6, -0.24, 0.17), (54.4, *H3))
    add(2, "rear", 5.8, 7.2)
    note(5.6, 7.4, [3], "rears in the open")
    # crawl over mouse 4: straight across its back, perpendicular to its body, 11.2-12.6 s
    th4 = heading(m4)
    p4, d4 = m4[at(12.0)], unit(th4[at(12.0)])
    back4 = p4 + 0.025 * d4
    perp4 = np.array([-d4[1], d4[0]])
    s = np.clip((T - 11.2) / 1.4, 0, 1)
    crawl = back4 + (s[:, None] - 0.5) * 0.24 * perp4
    w_crawl = W(11.2, 12.6, 0.3)
    m3 = blend(m3, crawl, w_crawl)
    note(11.2, 12.6, [3, 4], "crawl-over: mouse 3 walks over resting mouse 4's back")
    # side by side with mouse 4, 3.5 cm to its left
    side = m4 - 0.036 * np.stack([np.sin(th4), -np.cos(th4)], -1)   # inner side, away from the wall
    m3 = blend(m3, side, W(14.0, 18.6, 0.5))
    note(13.0, 18.6, [3, 4], "side by side: mice 3 and 4 walk together along the north and west walls")
    add(2, "rear", 27.6, 31.0)
    add(2, "stretch", 35.4, 37.4)
    note(35.4, 39.0, [3, 2], "approach-avoid: mouse 3 approaches mouse 2 in stretched attend, mouse 2 bolts")

    # ---- mouse 1: climbs and rears on the platform, nose-to-nose, flank sniff, anogenital sniff
    # while following, fight, chase, climbs the tall block to peer down at mouse 3, allogrooms
    # mouse 4, end of the train, home
    m1 = keys((0.0, *H1), (1.0, *H1), (1.8, 0.0, 0.07), (2.8, -0.05, -0.10), (3.8, -0.22, -0.20),
              (5.6, -0.22, -0.20), (6.4, -0.10, -0.31), (7.3, -0.11, -0.20), (8.3, -0.11, -0.07),
              (13.8, 0.0, -0.05), (18.8, *e1),
              (21.8, 0.16, -0.31), (23.0, 0.05, -0.10), (24.5, 0.06, 0.02), (25.5, 0.22, 0.05),
              (26.3, 0.22, 0.20), (27.0, 0.245, 0.20), (31.2, 0.245, 0.20), (32.1, 0.22, 0.05),
              (33.4, 0.14, -0.06), (35.0, 0.10, -0.03),
              # 36-40 allogrooming (overridden), then to the train tail
              (40.4, 0.10, -0.06), (41.4, 0.22, -0.20), (42.4, 0.31, -0.14),
              (48.8, 0.20, -0.30), (50.0, -0.05, -0.12), (51.0, -0.27, -0.02), (51.9, -0.25, 0.10), (52.8, *H1))
    add(0, "rear", 4.0, 5.4)
    note(2.8, 6.4, [1], "climbs the platform, rears on top, climbs down")
    add(0, "stretch", 7.3, 8.5)
    # nose-to-nose 8.8-9.9: mouse 1 stands 12.5 cm in front of mouse 2, facing it
    th2 = heading(m2)
    nn = m2 + 0.125 * unit(th2)
    m1 = blend(m1, nn, W(8.7, 9.9, 0.5))
    note(7.3, 8.7, [1, 2], "approach: mouse 1 approaches mouse 2 in stretched attend")
    note(8.8, 9.9, [1, 2], "nose-to-nose")
    # flank sniffing: mouse 1 circles mouse 2 (still) from its front to its rear, 9.9-10.6 s
    a0 = th2[at(9.5)]
    sweep = a0 + np.pi * g.smoothstep((T - 9.9) / 1.0)
    arc = m2 + 0.115 * unit(sweep)
    m1 = blend(m1, arc, W(9.9, 10.9, 0.15))
    note(9.9, 10.9, [1, 2], "flank sniffing: mouse 1 circles mouse 2")
    # anogenital sniffing while following 10.6-13.8: snout at mouse 2's tail base
    m1 = blend(m1, behind(m2, 0.085), W(10.5, 13.8, 0.4))
    note(10.9, 13.8, [1, 2], "following with anogenital sniffing")
    m1 = blend(m1, ring1, fight)
    # chase 18.8-21.8: mouse 2 flees over the platform, mouse 1 10 cm behind on the same path
    m1 = blend(m1, behind(m2, 0.10), W(18.9, 21.8, 0.25))
    note(18.8, 21.8, [1, 2], "chasing: mouse 2 flees over the platform, mouse 1 chases over it too")
    add(0, "peer", 27.6, 31.0)
    note(25.5, 27.0, [1], "climbs the tall block")
    note(27.6, 31.0, [1, 3], "sniffing across the tall block: mouse 1 peers down from the top, mouse 3 rears up")
    note(31.2, 32.1, [1], "climbs down the tall block")
    # allogrooming 36-40: mouse 1 beside mouse 4's shoulder, snout on its neck
    th4 = heading(m4)
    neck4 = m4 + 0.05 * unit(th4)
    groom_spot = neck4 + 0.07 * np.stack([-np.sin(th4), np.cos(th4)], -1)
    m1 = blend(m1, groom_spot, W(35.8, 40.3, 0.6))
    add(0, "sniff", 36.3, 39.9)
    note(36.0, 40.2, [1, 4], "allogrooming: mouse 1 grooms mouse 4's neck")

    # ---- train 42-48: mouse 2 leads, mouse 3 10 cm behind, mouse 1 10 cm behind mouse 3
    m3 = blend(m3, behind(m2, 0.10), W(42.0, 48.4, 0.5))
    m1 = blend(m1, behind(m2, 0.20), W(42.6, 48.4, 0.6))
    note(41.0, 42.6, [1], "walks along the step bar to join the train")
    note(42.0, 48.4, [2, 3, 1], "train: mouse 2 leads, mice 3 and 1 follow nose-to-tail, over the step bar")

    # final light smoothing joins the overridden pieces
    paths = [smooth_path(p, 2) for p in (m1, m2, m3, m4)]

    # facing
    face(0, paths[1], 8.8, 9.9)
    face(1, paths[0], 8.8, 9.9)
    face(0, paths[1], 9.9, 10.9, 0.15)
    for (a, b) in ((15.3, 16.1), (17.4, 18.2)):
        face(0, paths[1], a, b, 0.15)
        face(1, paths[0], a, b, 0.15)
    face(0, paths[2], 27.4, 31.0, 0.5)
    face(2, paths[0], 27.4, 31.0, 0.5)
    face(2, paths[1], 36.0, 37.5)
    face(0, neck4, 36.0, 40.0, 0.5)
    note(0.0, 0.0, [], "")
    SCRIPT.pop()
    return paths, face_t, face_w, beh, box


# ------------------------------------------------------------------ bodies


def body(xy, th, beh, rng):
    """Flat-floor 3D skeleton for one mouse (generate_mouse_datasets.mouse_frames) with extra
    postures: resting (low, quiet), peering down (head pitched down), sniffing (head bobs)."""
    n = N
    sp = np.linalg.norm((np.roll(xy, -1, 0) - np.roll(xy, 1, 0)) * FPS / 2, axis=1)
    dth = np.angle(np.exp(1j * (np.roll(th, -1) - np.roll(th, 1)))) * FPS / 2
    kappa = np.clip(dth / np.maximum(sp, 0.05), -40, 40)
    still = sp < 0.02
    prog = np.full(n, -1.0)
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            prog[i:j] = np.linspace(0, 1, j - i) if j - i > 1 else 0.5
            i = j
        else:
            i += 1
    w = {k: beh[k] for k in ("rear", "groom", "stretch")}
    pose = g.pose_from_weights(n, w, rng, beh["agit"])
    rest, peer, sniff = beh["rest"], beh["peer"], beh["sniff"]
    quiet = 1 - 0.8 * rest
    pose["head_yaw"] = pose["head_yaw"] * quiet
    pose["tail_yaw"] = pose["tail_yaw"] * quiet
    pose["bend"] = pose["bend"] * quiet
    pose["pelvis_h"] = pose["pelvis_h"] - 0.004 * rest
    pose["pitch"]["back"] = pose["pitch"]["back"] - 0.15 * rest
    pose["pitch"]["neck"] = pose["pitch"]["neck"] - 0.25 * rest - 0.75 * peer \
        + 0.30 * np.sin(2 * np.pi * 5.0 * T) * sniff
    pose["pitch"]["snout"] = pose["pitch"]["snout"] - 0.15 * rest - 0.55 * peer
    tr = dict(xy=xy, theta=th, kappa=kappa, speed=sp, rear=beh["rear"], still=still & (rest < 0.5),
              prog=prog, n=n, pose=pose)
    return g.mouse_frames(tr)


FRONT = ["neck", "shoulder_L", "shoulder_R", "head", "snout", "ear_L", "ear_R"]
BACK = ["pelvis", "hip_L", "hip_R", "tail_base"]


def lift(F, bump=None):
    """Put a flat-floor body onto the surface. bump(xz) adds a per-frame height (crawl-over)."""
    def H(name):
        p = F[:, JI[name]]
        h = surface(p[:, 0], p[:, 2])
        if bump is not None:
            h = np.maximum(h, bump(p[:, [0, 2]]))
        return h
    out = F.copy()
    Lp, Ln = H("pelvis"), H("neck")
    for nm in BACK:
        out[:, JI[nm], 1] += Lp
    for nm in FRONT:
        out[:, JI[nm], 1] += Ln
    out[:, JI["spine_mid"], 1] += (Lp + Ln) / 2
    for nm in ("tail_mid", "tail_tip"):
        out[:, JI[nm], 1] += H(nm)
    for side in "LR":
        out[:, JI[f"forepaw_{side}"], 1] += np.maximum(H(f"forepaw_{side}"), Ln - 0.02)
        out[:, JI[f"hindpaw_{side}"], 1] += np.maximum(H(f"hindpaw_{side}"), Lp - 0.02)
    out[..., 1] = np.maximum(out[..., 1], 0.0015)   # a drooping tail or low shoulder rests on the floor
    return out, Lp


def back_bump(F_under, window):
    """Height of the top of a mouse's back (pelvis to neck), for another mouse crawling over it."""
    a, b = F_under[:, JI["pelvis"]][:, [0, 2]], F_under[:, JI["neck"]][:, [0, 2]]
    top = np.maximum(F_under[:, JI["spine_mid"], 1], F_under[:, JI["pelvis"], 1]) + 0.008

    def bump(p):
        ab = b - a
        s = np.clip(np.sum((p - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-9), 0, 1)
        d = np.linalg.norm(p - (a + s[:, None] * ab), axis=1)
        return window * top * np.exp(-0.5 * (d / 0.014) ** 2)
    return bump


# ------------------------------------------------------------------ tracking noise


def corrupt(clean, rng):
    """clean: [animal][frame][joint][xyz]. Returns noisy positions with NaN for missing samples."""
    A, n, J, _ = clean.shape
    out = clean.copy()
    sd = np.full(J, 0.0025)
    for nm in ("forepaw_L", "forepaw_R", "hindpaw_L", "hindpaw_R", "tail_tip"):
        sd[JI[nm]] = 0.0045
    out += rng.normal(size=out.shape) * sd[None, None, :, None]
    # slow drift per joint (AR(1))
    phi, s_ar = 0.92, 0.0015
    e = np.zeros((A, J, 3))
    for f in range(n):
        e = phi * e + rng.normal(0, s_ar * np.sqrt(1 - phi ** 2), size=e.shape)
        out[:, f] += e
    stats = {}
    # outlier jumps
    n_out = 0
    for a in range(A):
        for _ in range(rng.poisson(30)):
            j, f0, L = rng.integers(J), rng.integers(n), rng.integers(1, 4)
            jump = rng.normal(size=3)
            jump *= rng.uniform(0.01, 0.04) / np.linalg.norm(jump)
            out[a, f0:f0 + L, j] += jump
            n_out += 1
    stats["outlier_jumps"] = n_out
    # left/right swaps
    n_sw = 0
    for a in range(A):
        for _ in range(4):
            pair = [("ear_L", "ear_R"), ("forepaw_L", "forepaw_R")][rng.integers(2)]
            f0, L = rng.integers(n), rng.integers(3, 13)
            i, k = JI[pair[0]], JI[pair[1]]
            out[a, f0:f0 + L, [i, k]] = out[a, f0:f0 + L, [k, i]]
            n_sw += 1
    stats["left_right_swaps"] = n_sw

    miss = np.zeros((A, n, J), bool)
    # random dropout bursts
    p_start = np.full(J, 0.002)
    for nm, p in (("forepaw_L", 0.012), ("forepaw_R", 0.012), ("hindpaw_L", 0.012), ("hindpaw_R", 0.012),
                  ("tail_tip", 0.010), ("tail_mid", 0.005), ("ear_L", 0.004), ("ear_R", 0.004)):
        p_start[JI[nm]] = p
    for a in range(A):
        for j in range(J):
            starts = np.nonzero(rng.random(n) < p_start[j])[0]
            for f0 in starts:
                miss[a, f0:f0 + rng.geometric(0.25), j] = True
    dropout = miss.copy()
    # occlusion: another mouse's body is above this joint (seen from above)
    body_idx = [JI[k] for k in ("pelvis", "spine_mid", "neck", "head")]
    occl = np.zeros_like(miss)
    for a in range(A):
        for b in range(A):
            if a == b:
                continue
            for bj in body_idx:
                d = np.linalg.norm(clean[a][:, :, [0, 2]] - clean[b][:, None, bj, [0, 2]], axis=2)   # [frame][joint]
                above = clean[b][:, None, bj, 1] > clean[a][:, :, 1] + 0.004
                occl[a] |= (d < 0.018) & above
    occl &= rng.random(occl.shape) < 0.7
    miss |= occl
    # hind paws hidden under the body while rearing
    for a in range(A):
        rearing = (clean[a][:, JI["neck"], 1] - clean[a][:, JI["pelvis"], 1]) > 0.035
        for nm in ("hindpaw_L", "hindpaw_R"):
            miss[a, :, JI[nm]] |= rearing & (rng.random(n) < 0.5)
    # whole animal lost for a few frames
    lost = 0
    for a in range(A):
        for _ in range(3):
            f0, L = rng.integers(n), rng.integers(2, 9)
            miss[a, f0:f0 + L, :] = True
            lost += min(L, n - f0)
    out[miss] = np.nan
    stats.update({
        "missing_fraction": round(float(miss.mean()), 4),
        "missing_by_joint": {nm: round(float(miss[:, :, i].mean()), 4) for i, (nm, _) in enumerate(g.MOUSE_JOINTS)},
        "occluded_samples": int(occl.sum()),
        "whole_animal_lost_frames": int(lost),
        "white_noise_sd_m": {"body": 0.0025, "paws_and_tail_tip": 0.0045},
        "drift_sd_m": s_ar,
    })
    return out, stats


# ------------------------------------------------------------------ checks


def check(clean, paths, lifts):
    """Geometry checks on the clean data: arena bounds, speeds, close approaches."""
    rep = {}
    xz = clean[..., [0, 2]]
    rep["max_abs_floor_coordinate_m"] = round(float(np.abs(xz).max()), 4)
    assert rep["max_abs_floor_coordinate_m"] < HALF, rep
    assert clean[..., 1].min() > -1e-3, clean[..., 1].min()
    sp = [np.linalg.norm(np.diff(p, axis=0, append=p[:1]), axis=1) * FPS for p in paths]
    rep["max_speed_m_s"] = {NAMES[i]: round(float(s.max()), 3) for i, s in enumerate(sp)}
    rep["pelvis_distance_min_m"] = {}
    for a in range(4):
        for b in range(a + 1, 4):
            d = np.linalg.norm(paths[a] - paths[b], axis=1)
            rep["pelvis_distance_min_m"][f"{a + 1}-{b + 1}"] = round(float(d.min()), 3)
    rep["seconds_on_a_block"] = {NAMES[i]: round(float((L > 0.012).sum() / FPS), 1) for i, L in enumerate(lifts)}
    snout = clean[:, :, JI["snout"]]
    rep["min_snout_to_snout_m"] = {}
    for a in range(4):
        for b in range(a + 1, 4):
            rep["min_snout_to_snout_m"][f"{a + 1}-{b + 1}"] = round(float(np.linalg.norm(snout[a] - snout[b], axis=1).min()), 3)
    return rep


def body_overlaps(clean, allowed):
    """Frames where two mice's spines pass within 2.5 cm (xz) outside the scripted contact windows."""
    sp = [JI[k] for k in ("pelvis", "spine_mid", "neck", "head")]
    bad = []
    for a in range(4):
        for b in range(a + 1, 4):
            d = np.linalg.norm(clean[a][:, sp, None, :][..., [0, 2]] - clean[b][:, None, sp, :][..., [0, 2]], axis=-1)
            close = d.min(axis=(1, 2)) < 0.025
            close &= ~allowed[(a, b)]
            if close.any():
                f = np.nonzero(close)[0]
                bad.append((a + 1, b + 1, round(f[0] / FPS, 2), round(f[-1] / FPS, 2), int(close.sum())))
    return bad


# ------------------------------------------------------------------ main


def main():
    rng = np.random.default_rng(20261109)
    paths, face_t, face_w, beh, box = choreograph()
    heads = [heading(paths[m], face_t[m], face_w[m]) for m in range(4)]
    flat = [body(paths[m], heads[m], beh[m], rng) for m in range(4)]
    lifted, lifts = [], []
    for m in range(4):
        F, Lp = lift(flat[m])
        lifted.append(F)
        lifts.append(Lp)
    # crawl-over: mouse 3 climbs onto mouse 4's back
    crawl_w = W(11.2, 12.6, 0.1)
    lifted[2], lifts[2] = lift(flat[2], back_bump(lifted[3], crawl_w))
    clean = np.stack(lifted)            # [animal][frame][joint][xyz]

    rep = check(clean, paths, lifts)
    allowed = {(a, b): np.zeros(N, bool) for a in range(4) for b in range(a + 1, 4)}
    for (s0, s1, who, _) in SCRIPT:
        for a in who:
            for b in who:
                if a < b:
                    allowed[(a - 1, b - 1)] |= (T >= s0 - 0.5) & (T <= s1 + 0.5)
    bad = body_overlaps(clean, allowed)
    rep["unscripted_close_contacts"] = [dict(mice=f"{a}-{b}", first_s=f0, last_s=f1, frames=k) for a, b, f0, f1, k in bad]

    noisy, noise_stats = corrupt(clean, rng)

    # events: clip-wide social behaviours from the script, plus "on a block" (any mouse)
    def ev(*phrases):
        on = np.zeros(N, bool)
        for (s0, s1, who, what) in SCRIPT:
            if any(what.startswith(p) for p in phrases):
                on |= (T >= s0) & (T < s1)
        return on
    events = {
        "huddling": ev("huddling"),
        "nose-to-nose": ev("nose-to-nose", "sniffing across"),
        "sniffing": ev("flank sniffing", "following with anogenital", "allogrooming"),
        "fighting": ev("fighting"),
        "chasing / avoiding": ev("chasing", "approach-avoid"),
        "crawl-over": ev("crawl-over"),
        "following": ev("side by side", "train"),
        "on a block": np.any(np.stack(lifts) > 0.012, axis=0),
    }

    ids = [f"m{i + 1}" for i in range(4)]
    tracks = {}
    for a in range(4):
        P = np.round(noisy[a], 4)
        frames = []
        for f in range(N):
            if np.isnan(P[f]).all():
                frames.append(None)
            else:
                frames.append([None if np.isnan(p).any() else p.tolist() for p in P[f]])
        tracks[ids[a]] = {"positions": frames}
    skel = g.skeleton_block(g.MOUSE_JOINTS, clean[0, 0])
    data = {
        "format": "etho-pose",
        "version": "1.0",
        "meta": {
            "name": "four mice, open field with blocks",
            "description": "Synthetic 3D skeletons of four mice for 60 s in an 80 cm open field with three blocks they "
                           "climb. Huddling, nose-to-nose, flank and anogenital sniffing, fighting, chasing over a block, "
                           "crawl-over, side-by-side walking, sniffing across the tall block, approach-avoid, allogrooming "
                           "and a three-mouse train, plus climbing, rearing, grooming and stretched attend. Noisy, with "
                           "outliers, left/right swaps and missing (null) samples.",
            "species": "mouse",
            "source": "generate_mouse_group_dataset.py (synthetic)",
            "arena": {"shape": "square", "size_m": 2 * HALF, "center": [0.0, 0.0],
                      "blocks": [{"name": b["name"], "center_xz": [b["x"], b["z"]], "size_xz": [b["sx"], b["sz"]],
                                  "height": b["h"]} for b in BLOCKS]},
        },
        "space": {"dims": 3, "units": "m", "up": "+Y", "origin": "arena center, floor at y = 0"},
        "skeletons": {"mouse": {"joints": [{"name": j["name"], "parent": (None if j["parent"] < 0 else j["parent"])}
                                           for j in skel["joints"]],
                                "bones": skel["bones"]}},
        "animals": [{"id": ids[a], "name": NAMES[a], "skeleton": "mouse", "species": "mouse"} for a in range(4)],
        "videos": [],
        "clips": [{
            "name": "group",
            "fps": FPS,
            "frame_count": N,
            "loop": True,
            "tracks": tracks,
            "events": [{"name": k, "animal": None, "active": [int(x) for x in v]} for k, v in events.items()],
            "extra": {
                "script": [{"start_s": a, "end_s": b, "animals": [ids[i - 1] for i in who], "what": w}
                           for a, b, who, w in sorted(SCRIPT)],
                "validation": rep,
                "noise": noise_stats,
            },
        }],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in noise_stats.items() if k != "missing_by_joint"}, indent=1))


if __name__ == "__main__":
    main()
