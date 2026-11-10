"""
Synthesize ideal (noise-free) mouse pose test datasets.

Writes, into examples/ (the folder above this script):
  mouse_centroid.json  2D centroid only: one (x, y) point per frame on the floor.
  mouse_3d.json        3D skeleton that runs around and rears on its hind legs.
  mouse_2d.json        2D top-down skeleton that runs around.
  mouse_social.json    Two 3D mice for 30 s: apart, investigating, fighting, chasing, apart again.
                       Positions are [frame][animal][joint][xyz] with an "animals" list.
  mouse_3d_large.json  3D skeleton, 30 s touring an 80 cm region near uniformly, with more
                       varied posture (rearing, grooming, stretched attend, head and tail wander)
                       and white noise (sd 2 mm) on every node. Rigid-bone and region checks run
                       before the noise is added.

Layout matches skeleton_dataset.json (skeleton + clips[].positions[frame][joint][coord]),
with these additions:
  "dimensions"    2 or 3 (length of each coordinate)
  clips[].events  per-frame 0/1 flags, e.g. {"rearing": [...]}

Coordinates: meters, origin at the center of the region the animal moves in.
  3D: +X and +Z span the floor, +Y is up (same convention as the quadruped set).
  2D: top-down view of the floor; 2D (x, y) corresponds to 3D (x, z).

Every clip loops seamlessly: the trajectory is a closed path and the gait completes a whole
number of cycles, so frame N-1 flows straight into frame 0.

Guarantees checked by validate():
  * Spine, head and tail bones keep exactly the same length in every frame (3D and 2D).
  * The animal stays inside a 40 cm square region (every joint, every frame).
"""
import json
from pathlib import Path

import numpy as np

FPS = 30
REGION = 0.40         # side of the square region the animal stays inside (m)
HALF = REGION / 2
OUT = Path(__file__).resolve().parent.parent   # examples/

# ------------------------------------------------------------------ trajectory


def closed_spline(waypoints, samples_per_seg=400):
    """Closed Catmull-Rom spline. Returns dense points and cumulative arc length."""
    P = np.asarray(waypoints, float)
    n = len(P)
    t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, None]
    pts = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        pts.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts = np.vstack(pts)
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    knots = s[::samples_per_seg][:n]  # arc position of each waypoint
    return closed, s, knots


def sample_path(closed, s_dense, s):
    total = s_dense[-1]
    s = np.mod(s, total)
    return np.stack([np.interp(s, s_dense, closed[:, k]) for k in range(2)], axis=-1)


def heading_and_curvature(closed, s_dense, s, ds=0.004):
    a, b = sample_path(closed, s_dense, s - ds), sample_path(closed, s_dense, s + ds)
    d = b - a
    theta = np.arctan2(d[..., 1], d[..., 0])
    a2, b2 = sample_path(closed, s_dense, s - 2 * ds), sample_path(closed, s_dense, s)
    a3, b3 = sample_path(closed, s_dense, s), sample_path(closed, s_dense, s + 2 * ds)
    th1 = np.arctan2(*(b2 - a2)[..., ::-1].T)
    th2 = np.arctan2(*(b3 - a3)[..., ::-1].T)
    kappa = np.angle(np.exp(1j * (th2 - th1))) / (2 * ds)
    return theta, kappa


def schedule(total, knots, stops, speeds):
    """
    Per-frame arc position along the closed path.
    stops: list of (waypoint_index, dwell_seconds, kind). Movement between stops eases in and
    out (zero velocity at each stop). Returns s[frame], kind[frame] ('' while moving) and
    progress-in-dwell[frame] (0..1, -1 while moving).
    """
    marks = [0.0] + [knots[i] for i, _, _ in stops] + [total]
    s_out, kind_out, prog_out = [], [], []
    for i in range(len(marks) - 1):
        D = marks[i + 1] - marks[i]
        n = max(int(round(D / speeds[i % len(speeds)] * FPS)), 2)
        tau = np.arange(n) / n
        s_out.append(marks[i] + D * (tau - np.sin(2 * np.pi * tau) / (2 * np.pi)))
        kind_out += [""] * n
        prog_out.append(np.full(n, -1.0))
        if i < len(stops):
            _, dwell, kind = stops[i]
            m = int(round(dwell * FPS))
            s_out.append(np.full(m, marks[i + 1]))
            kind_out += [kind] * m
            prog_out.append(np.arange(m) / max(m - 1, 1))
    return np.concatenate(s_out), kind_out, np.concatenate(prog_out)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def periodic_smooth(x, sigma_frames):
    """Gaussian low-pass of a looping signal (wraps around, so the loop stays seamless)."""
    n = len(x)
    k = np.arange(n)
    k = np.minimum(k, n - k)
    w = np.exp(-0.5 * (k / sigma_frames) ** 2)
    w /= w.sum()
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(w)))


def periodic_smooth_angle(theta, sigma_frames):
    """Smooth a looping heading: unwrap, remove the whole turns per loop, smooth, restore."""
    u = np.unwrap(theta)
    n = len(u)
    turns = np.round((u[-1] + (u[-1] - u[-2]) - u[0]) / (2 * np.pi)) * 2 * np.pi
    trend = turns * np.arange(n) / n
    return periodic_smooth(u - trend, sigma_frames) + trend


def trajectory(waypoints, stops, speeds):
    closed, s_dense, knots = closed_spline(waypoints)
    s, kind, prog = schedule(s_dense[-1], knots, stops, speeds)
    xy = sample_path(closed, s_dense, s)
    theta, kappa = heading_and_curvature(closed, s_dense, s)
    # low-pass heading and curvature over time so tight turns don't whip the head and tail
    theta = periodic_smooth_angle(theta, sigma_frames=4)
    kappa = periodic_smooth(kappa, sigma_frames=6)
    # speed in m/s, wrapping last -> first frame
    step = np.diff(np.concatenate([s, [s[0] + s_dense[-1]]]))
    speed = step * FPS
    # rear factor: ramps 0 -> 1 -> 0 over each rearing dwell
    ramp = 0.18
    rear = np.array([smoothstep(min(p / ramp, (1 - p) / ramp)) if k == "rear" else 0.0
                     for k, p in zip(kind, prog)])
    still = np.array([k != "" for k in kind])
    return dict(xy=xy, theta=theta, kappa=kappa, speed=speed, rear=rear, still=still,
                prog=prog, kind=kind, n=len(s))


# ------------------------------------------------------------------ 3D mouse body

MOUSE_JOINTS = [
    ("pelvis", None),
    ("spine_mid", "pelvis"),
    ("neck", "spine_mid"),
    ("head", "neck"),
    ("snout", "head"),
    ("ear_L", "head"),
    ("ear_R", "head"),
    ("tail_base", "pelvis"),
    ("tail_mid", "tail_base"),
    ("tail_tip", "tail_mid"),
    ("shoulder_L", "neck"),
    ("forepaw_L", "shoulder_L"),
    ("shoulder_R", "neck"),
    ("forepaw_R", "shoulder_R"),
    ("hip_L", "pelvis"),
    ("hindpaw_L", "hip_L"),
    ("hip_R", "pelvis"),
    ("hindpaw_R", "hip_R"),
]

# bone lengths (m) for the rigid chain: adult C57BL/6-sized mouse
SEG = dict(back=0.028, front=0.026, neck=0.014, snout=0.016,
           tail_base=0.008, tail=0.042)
PELVIS_HEIGHT = 0.017
# segment pitch (rad, + is up) while on all fours, and at full rear
PITCH_WALK = dict(back=0.30, front=-0.22, neck=-0.10, snout=-0.38)
PITCH_REAR = dict(back=1.15, front=1.35, neck=0.95, snout=0.15)
# extra postures used by the large dataset: hunched grooming (head down to the forepaws) and
# stretched attend (body long and low, head forward)
PITCH_GROOM = dict(back=0.65, front=-0.55, neck=-1.05, snout=-1.35)
PITCH_STRETCH = dict(back=0.02, front=0.0, neck=0.02, snout=-0.12)


def direction(yaw, pitch):
    """Unit vector: yaw measured in the floor plane from +X toward +Z, pitch up from the floor."""
    return np.array([np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw)])


def mouse_frames(tr):
    n = tr["n"]
    idx = {name: i for i, (name, _) in enumerate(MOUSE_JOINTS)}
    out = np.zeros((n, len(MOUSE_JOINTS), 3))

    # trot gait: stride frequency rises with speed; snap to a whole number of cycles per loop
    freq = 3.0 + 20.0 * tr["speed"]
    phase = np.concatenate([[0.0], np.cumsum(freq / FPS)[:-1]])
    total = phase[-1] + freq[-1] / FPS
    phase *= max(round(total), 1) / total
    gait_offset = {"forepaw_L": 0.0, "hindpaw_R": 0.0, "forepaw_R": 0.5, "hindpaw_L": 0.5}

    # lateral spine bend follows path curvature while moving; smoothed so it eases in and out of stops
    moving_all = np.minimum(1.0, tr["speed"] / 0.03)
    bend_all = periodic_smooth(np.clip(tr["kappa"] * 0.03, -0.35, 0.35) * moving_all, sigma_frames=5)

    for f in range(n):
        x, z = tr["xy"][f]
        th = tr["theta"][f]
        r = tr["rear"][f]
        v = tr["speed"][f]
        moving = min(1.0, v / 0.03)
        bend = bend_all[f]
        # head scans side to side while still (zero at the start and end of each dwell)
        scan = 0.0
        if tr["still"][f]:
            p = tr["prog"][f]
            scan = 0.35 * np.sin(2 * np.pi * 2.0 * p) * np.sin(np.pi * p)

        fwd = np.array([np.cos(th), 0.0, np.sin(th)])
        left = np.array([np.sin(th), 0.0, -np.cos(th)])   # +Z is the animal's right
        up = np.array([0.0, 1.0, 0.0])

        pose = tr.get("pose")   # optional per-frame posture (large dataset); None keeps the original
        if pose is not None:
            pitch = {k: pose["pitch"][k][f] for k in PITCH_WALK}
            pelvis_h = pose["pelvis_h"][f]
            groom = pose["groom"][f]
            scan += pose["head_yaw"][f]
            tail_yaw = pose["tail_yaw"][f]
            bend = bend + pose["bend"][f]
        else:
            pitch = {k: (1 - r) * PITCH_WALK[k] + r * PITCH_REAR[k] for k in PITCH_WALK}
            pelvis_h = PELVIS_HEIGHT - 0.002 * r
            groom = 0.0
            tail_yaw = 0.0
        J = {}
        J["pelvis"] = np.array([x, pelvis_h, z]) - fwd * 0.006 * r
        J["spine_mid"] = J["pelvis"] + SEG["back"] * direction(th + bend, pitch["back"])
        J["neck"] = J["spine_mid"] + SEG["front"] * direction(th + 2 * bend, pitch["front"])
        head_yaw = th + 3 * bend + scan
        if pose is not None:
            # keep the head and snout above the floor by limiting how far those segments pitch down
            # (only the pitch changes, so bone lengths stay exact)
            floor_min = 0.004
            lo = np.arcsin(np.clip((floor_min + 0.002 - J["neck"][1]) / SEG["neck"], -1, 1))
            pitch["neck"] = max(pitch["neck"], lo)
        J["head"] = J["neck"] + SEG["neck"] * direction(head_yaw, pitch["neck"])
        if pose is not None:
            lo = np.arcsin(np.clip((floor_min - J["head"][1]) / SEG["snout"], -1, 1))
            pitch["snout"] = max(pitch["snout"], lo)
        J["snout"] = J["head"] + SEG["snout"] * direction(head_yaw, pitch["snout"])
        h_fwd = direction(head_yaw, pitch["snout"])
        h_left = np.array([np.sin(head_yaw), 0.0, -np.cos(head_yaw)])
        h_up = np.cross(h_left, h_fwd)
        for side, sg in (("L", 1.0), ("R", -1.0)):
            J[f"ear_{side}"] = J["head"] + 0.008 * sg * h_left + 0.005 * h_up - 0.002 * h_fwd

        # tail continues the body curve backwards and droops to the floor
        J["tail_base"] = J["pelvis"] + SEG["tail_base"] * direction(th + np.pi - bend, -0.6)
        J["tail_mid"] = J["tail_base"] + SEG["tail"] * direction(th + np.pi - 2 * bend + tail_yaw, -0.20)
        J["tail_tip"] = J["tail_mid"] + SEG["tail"] * direction(th + np.pi - 3 * bend + 2 * tail_yaw, -0.02)

        # limbs (not length-constrained: paws follow a simple trot and tuck in when rearing)
        stride = v / max(freq[f], 1e-6)
        for side, sg in (("L", 1.0), ("R", -1.0)):
            for limb in ("fore", "hind"):
                paw = f"{limb}paw_{side}"
                if limb == "fore":
                    top = J["neck"] + left * sg * 0.008 - up * 0.004 - fwd * 0.004
                    J[f"shoulder_{side}"] = top
                else:
                    top = J["pelvis"] + left * sg * 0.009 - up * 0.004 + fwd * 0.002
                    J[f"hip_{side}"] = top
                ph = 2 * np.pi * (phase[f] + gait_offset[paw])
                swing = 0.5 * stride * np.sin(ph) * moving
                lift = 0.005 * max(0.0, np.cos(ph)) * moving
                ground = np.array([top[0], 0.0, top[2]]) + fwd * (swing + (0.006 if limb == "fore" else 0.0)) \
                    + left * sg * 0.003 + up * (0.002 + lift)
                if limb == "fore":
                    tucked = top - up * 0.010 + fwd * 0.006 * np.cos(pitch["front"])
                    J[paw] = (1 - r) * ground + r * tucked
                    if groom > 0:   # forepaws come up to the snout
                        at_face = J["snout"] - up * 0.004 + h_left * sg * 0.005 - h_fwd * 0.004
                        J[paw] = (1 - groom) * J[paw] + groom * at_face
                else:
                    J[paw] = ground

        for name, i in idx.items():
            out[f, i] = J[name]
    return out


# ------------------------------------------------------------------ 2D skeleton (top-down)

MOUSE_2D_JOINTS = [
    ("pelvis", None),
    ("spine_mid", "pelvis"),
    ("neck", "spine_mid"),
    ("head", "neck"),
    ("snout", "head"),
    ("ear_L", "head"),
    ("ear_R", "head"),
    ("tail_base", "pelvis"),
    ("tail_mid", "tail_base"),
    ("tail_tip", "tail_mid"),
    ("forepaw_L", "neck"),
    ("forepaw_R", "neck"),
    ("hindpaw_L", "pelvis"),
    ("hindpaw_R", "pelvis"),
]
RIGID_NAMES = {"spine_mid", "neck", "head", "snout", "ear_L", "ear_R", "tail_base", "tail_mid", "tail_tip"}


def project_2d(frames3d):
    idx3 = {name: i for i, (name, _) in enumerate(MOUSE_JOINTS)}
    sel = [idx3[name] for name, _ in MOUSE_2D_JOINTS]
    return frames3d[:, sel][:, :, [0, 2]]  # (x, z) -> 2D (x, y)


# ------------------------------------------------------------------ packaging


def skeleton_block(joints, rest):
    idx = {n: i for i, (n, _) in enumerate(joints)}
    bones = [[idx[p], idx[c]] for c, p in joints if p is not None]
    block = {
        "joint_count": len(joints),
        "joints": [{"index": i, "name": n, "parent": (idx[p] if p else -1)} for i, (n, p) in enumerate(joints)],
        "bones": bones,
    }
    if bones:
        block["bone_lengths_m"] = [round(float(np.linalg.norm(rest[c] - rest[p])), 6) for p, c in bones]
    block["rest_pose"] = np.round(rest, 6).tolist()
    return block


def validate(frames, joints, rigid_children, half=HALF):
    idx = {n: i for i, (n, _) in enumerate(joints)}
    report = {}
    pairs = [(idx[p], idx[c]) for c, p in joints if p is not None and c in rigid_children]
    if pairs:
        lengths = np.array([[np.linalg.norm(fr[c] - fr[p]) for p, c in pairs] for fr in frames])
        report["max_rigid_bone_length_drift_m"] = float((lengths.max(0) - lengths.min(0)).max())
    floor = frames[..., [0, 2]] if frames.shape[-1] == 3 else frames
    report["max_abs_floor_coordinate_m"] = float(np.abs(floor).max())
    assert report["max_abs_floor_coordinate_m"] < half, report
    assert report.get("max_rigid_bone_length_drift_m", 0.0) < 1e-9, report
    return report


def dataset(name, description, dims, joints, frames, events, report):
    coords = ({"handedness": "right", "up": "+Y", "floor_plane": "XZ", "ground_plane": "y = 0",
               "origin": "region center"} if dims == 3 else
              {"view": "top-down", "floor_plane": "XY", "origin": "region center",
               "note": "2D (x, y) equals 3D (x, z) of the other mouse datasets"})
    clip = {
        "name": name,
        "fps": FPS,
        "frame_count": len(frames),
        "duration_s": round(len(frames) / FPS, 4),
        "loop": True,
        "root_motion_per_loop": [0.0] * dims,
        "positions": np.round(frames, 5).tolist(),
        "validation": report,
    }
    if events:
        clip["events"] = {k: [int(x) for x in v] for k, v in events.items()}
    return {
        "format": "pose-testset",
        "version": "1.0",
        "species": "mouse",
        "description": description,
        "dimensions": dims,
        "units": "meters",
        "coordinate_system": coords,
        "skeleton": skeleton_block(joints, frames[0]),
        "clips": [clip],
    }


def write(name, data):
    path = OUT / name
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    clip = data["clips"][0]
    print(f"wrote {path.name:20s} frames={clip['frame_count']:4d}  {clip['validation']}")


# ------------------------------------------------------------------ large, noisy 3D dataset

BIG_REGION = 0.80         # side of the square region the large dataset covers (m)
BIG_SECONDS = 30.0
NOISE_SD = 0.002          # white noise added to every coordinate of every node (m)


def rich_pose(tr, rng):
    """Per-frame posture with much more variety: rearing, grooming and stretched-attend bouts,
    plus slow random wander in head pitch/yaw, back arch and tail swing (periodic, so the
    loop stays seamless)."""
    ramp = 0.2
    w = {k: np.array([smoothstep(min(p / ramp, (1 - p) / ramp)) if kk == k else 0.0
                      for kk, p in zip(tr["kind"], tr["prog"])]) for k in ("rear", "groom", "stretch")}
    return pose_from_weights(tr["n"], w, rng), w


def pose_from_weights(n, w, rng, agitation=None):
    """Posture from behaviour weights (rear / groom / stretch, each 0..1 per frame). agitation
    (0..1 per frame) makes the head, spine and tail move up to 2.5x more, e.g. while fighting."""
    agit = 1 + 1.5 * (agitation if agitation is not None else np.zeros(n))

    def wander(sigma_frames, amp):
        x = periodic_smooth(rng.normal(size=n), sigma_frames)
        return amp * x / (np.std(x) + 1e-12)

    head_pitch, head_yaw = wander(8, 0.30), wander(10, 0.40)
    back_arch, tail_yaw = wander(20, 0.20), wander(15, 0.45)
    side_bend = wander(18, 0.15)      # extra lateral flexion of the spine
    crouch = wander(25, 0.003)        # pelvis height wander (m)
    t = np.arange(n) / FPS
    bob = 0.25 * np.sin(2 * np.pi * 4.5 * t) * w["groom"]      # head bobbing while grooming
    pitch = {k: np.full(n, PITCH_WALK[k]) for k in PITCH_WALK}
    pitch["back"] = pitch["back"] + back_arch
    pitch["neck"] = pitch["neck"] + head_pitch * agit
    pitch["snout"] = pitch["snout"] + 0.8 * head_pitch * agit
    for kind, target in (("rear", PITCH_REAR), ("groom", PITCH_GROOM), ("stretch", PITCH_STRETCH)):
        for k in pitch:
            pitch[k] = (1 - w[kind]) * pitch[k] + w[kind] * target[k]
    pitch["neck"] = pitch["neck"] + bob
    pelvis_h = PELVIS_HEIGHT - 0.002 * w["rear"] - 0.005 * w["stretch"] - 0.001 * w["groom"] + crouch * (1 - w["rear"])
    calm = 1 - 0.6 * np.maximum(w["rear"], w["groom"])
    return dict(pitch=pitch, pelvis_h=pelvis_h, groom=w["groom"], head_yaw=head_yaw * calm * agit,
                tail_yaw=tail_yaw * agit, bend=side_bend * (1 - w["rear"]) * agit)


def grid_tour(m):
    """A closed tour of every point of an m x m grid (m even): up the first column, snake back
    and forth across the rest, return along the bottom row."""
    tour = [(0, r) for r in range(m)]
    for i, r in enumerate(range(m - 1, 0, -1)):
        cols = range(1, m) if i % 2 == 0 else range(m - 1, 0, -1)
        tour += [(c, r) for c in cols]
    tour += [(c, 0) for c in range(m - 1, 0, -1)]
    return tour


def big_trajectory(rng):
    """Visits every point of a jittered 6 x 6 grid on a closed tour, so the animal covers the
    whole 80 cm region near uniformly, with behaviour stops; rescaled to exactly ~30 s."""
    m = 6
    cells = np.linspace(-0.28, 0.28, m)
    wp = [(cells[c] + rng.uniform(-0.025, 0.025), cells[r] + rng.uniform(-0.025, 0.025)) for c, r in grid_tour(m)]
    kinds = ["rear", "groom", "rear", "stretch", "scan", "rear", "groom", "stretch"]
    stop_idx = sorted(rng.choice(np.arange(2, len(wp) - 1), size=len(kinds), replace=False))
    order = rng.permutation(len(kinds))
    stops = [(int(i), float(rng.uniform(1.6, 2.4) if kinds[order[j]] == "rear" else rng.uniform(1.0, 1.8)), kinds[order[j]])
             for j, i in enumerate(stop_idx)]
    speeds = list(rng.uniform(0.10, 0.22, size=len(stops) + 1))
    target = int(round(BIG_SECONDS * FPS))
    dwell = sum(int(round(d * FPS)) for _, d, _ in stops)
    for it in range(40):   # rescale speeds until the clip is exactly 30 s long
        tr = trajectory(wp, stops, speeds)
        if tr["n"] == target:
            break
        if it < 10:
            speeds = [v * (tr["n"] - dwell) / (target - dwell) for v in speeds]
        else:   # rounding: nudge one segment at a time
            j = it % len(speeds)
            speeds[j] *= 1.04 if tr["n"] > target else 0.96
    assert tr["n"] == target, tr["n"]
    return tr


def make_big(rng):
    tr = big_trajectory(rng)
    pose, w = rich_pose(tr, rng)
    tr["pose"] = pose
    clean = mouse_frames(tr)
    report = validate(clean, MOUSE_JOINTS, RIGID_NAMES, half=BIG_REGION / 2)
    noisy = clean + rng.normal(0.0, NOISE_SD, size=clean.shape)
    report["noise_sd_m"] = NOISE_SD
    report["pose_variance_m2"] = float(clean.reshape(len(clean), -1).var(0).sum())
    events = {"rearing": w["rear"] > 0.5, "grooming": w["groom"] > 0.5, "stretching": w["stretch"] > 0.5}
    return noisy, events, report


# ------------------------------------------------------------------ two interacting mice

SOCIAL_REGION = 0.60      # side of the square region the pair stays inside (m)
SOCIAL_SECONDS = 30.0
SOCIAL_SCRIPT = [         # (start s, end s, what happens) - also written into the file
    (0.0, 7.0, "apart: both explore; mouse 1 rears"),
    (6.4, 7.6, "mouse 1 approaches in a stretched-attend posture"),
    (7.6, 11.0, "investigating: mouse 1 follows mouse 2 nose-to-tail"),
    (11.0, 12.0, "investigating: nose-to-nose"),
    (12.0, 17.0, "fighting: fast circling with two boxing bouts (both rear, facing)"),
    (17.0, 19.2, "chasing: mouse 2 flees, mouse 1 chases"),
    (19.2, 30.0, "apart: mouse 2 grooms in a corner, mouse 1 explores and rears"),
]


def _interp_keys(keys, t, period):
    """Linear interpolation through (time, x, z) keyframes, wrapping at `period`."""
    kt = np.array([k[0] for k in keys] + [keys[0][0] + period])
    kx = np.array([k[1] for k in keys] + [keys[0][1]])
    kz = np.array([k[2] for k in keys] + [keys[0][2]])
    return np.stack([np.interp(t, kt, kx), np.interp(t, kt, kz)], axis=-1)


def _window(t, a, b, ramp=0.25):
    """0..1 weight that is 1 inside [a, b] with smooth ramps of `ramp` seconds at each end."""
    return smoothstep((t - a) / ramp) * smoothstep((b - t) / ramp)


def social_paths(rng):
    n = int(round(SOCIAL_SECONDS * FPS))
    t = np.arange(n) / FPS
    # mouse 2: free movement by keyframes, fighting worked out below
    k2 = [(0.0, 0.15, 0.12), (2.0, 0.18, -0.05), (3.5, 0.05, -0.15), (5.0, -0.05, -0.05), (6.5, 0.02, 0.08),
          (8.5, -0.08, 0.15), (10.0, -0.15, 0.05), (11.0, -0.12, -0.04), (12.0, -0.12, -0.04)]
    m2 = _interp_keys(k2 + [(SOCIAL_SECONDS - 0.01, 0.15, 0.12)], t, SOCIAL_SECONDS)
    # nose-to-nose at 11-12 s: pelvises ~12.5 cm apart puts the snouts a couple of cm apart
    p2_nn = np.array([-0.12, -0.04])
    p1_nn = p2_nn + 0.125 * np.array([0.6, 0.8])
    center = (p1_nn + p2_nn) / 2
    # fighting 12-17 s: both circle the shared centre half a turn apart, pausing to box
    fight = (t >= 12.0) & (t < 17.0)
    box = np.maximum(_window(t, 13.5, 14.3, 0.15), _window(t, 15.8, 16.6, 0.15))
    omega = 2 * np.pi * 1.2 * _window(t, 12.0, 17.0, 0.4) * (1 - box)
    phi = np.cumsum(omega) / FPS
    phi0 = np.arctan2(*(p2_nn - center)[::-1])
    r = 0.045 + 0.004 * np.sin(2 * np.pi * 0.7 * t)   # pelvises ~9 cm apart: nose near the other's tail
    wob = 0.004 * np.stack([np.sin(2 * np.pi * 0.9 * t), np.cos(2 * np.pi * 1.3 * t)], -1)
    ring2 = center + wob + r[:, None] * np.stack([np.cos(phi0 + phi), np.sin(phi0 + phi)], -1)
    ring1 = center + wob - r[:, None] * np.stack([np.cos(phi0 + phi), np.sin(phi0 + phi)], -1)
    m2[fight] = ring2[fight]
    end_fight = ring2[int(17.0 * FPS) - 1]
    k2b = [(17.0, *end_fight), (18.5, 0.0, -0.18), (19.5, 0.18, -0.15), (20.5, 0.20, 0.17), (22.0, 0.17, 0.20),
           (23.0, 0.19, 0.19), (26.0, 0.19, 0.19), (28.0, 0.17, 0.14)]
    after = t >= 17.0
    m2[after] = _interp_keys(k2b + [(SOCIAL_SECONDS, 0.15, 0.12)], t, 2 * SOCIAL_SECONDS)[after]
    m2[after & (t >= 28.0)] = _interp_keys([(28.0, 0.17, 0.14), (SOCIAL_SECONDS, 0.15, 0.12)], t[after & (t >= 28.0)], 2 * SOCIAL_SECONDS)

    # mouse 1: free keyframes, then follows / circles / chases relative to mouse 2
    k1 = [(0.0, -0.15, -0.12), (2.0, -0.18, 0.05), (3.5, -0.08, 0.17), (5.0, -0.19, 0.12), (6.4, -0.19, 0.12),
          (19.2, 0.0, 0.0), (20.0, 0.0, 0.0), (21.5, -0.05, -0.05), (23.0, -0.15, -0.18), (24.0, -0.18, -0.15),
          (25.4, -0.18, -0.15), (28.5, -0.17, -0.10)]
    m1 = _interp_keys(k1, t, SOCIAL_SECONDS)
    # following: mouse 1 trails a fixed distance behind mouse 2 along mouse 2's own path
    arc2 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(m2, axis=0), axis=1))])

    def behind(dist):
        tau = np.interp(arc2 - dist, arc2, np.arange(n))
        return np.stack([np.interp(tau, np.arange(n), m2[:, i]) for i in range(2)], -1)

    follow = behind(0.085)   # snout of mouse 1 at the tail base of mouse 2
    w_app = smoothstep((t - 6.4) / 1.2) * (t < 11.0)
    m1 = (1 - w_app[:, None]) * m1 + w_app[:, None] * follow
    nn = (t >= 11.0) & (t < 12.0)
    m1[nn] = p1_nn
    m1[fight] = ring1[fight]
    chase = (t >= 17.0) & (t < 19.2)
    m1[chase] = behind(0.10)[chase]
    hold = (t >= 19.2) & (t < 20.0)
    m1[hold] = m1[int(19.2 * FPS) - 1]
    rest = t >= 20.0
    k1b = [(19.95, *m1[int(19.2 * FPS) - 1])] + [k for k in k1 if k[0] >= 21.5]
    m1[rest] = _interp_keys(k1b + [(SOCIAL_SECONDS, -0.15, -0.12)], t, 2 * SOCIAL_SECONDS)[rest]
    # light periodic smoothing joins the pieces without changing the plan
    m1 = np.stack([periodic_smooth(m1[:, i], 2) for i in range(2)], -1)
    m2 = np.stack([periodic_smooth(m2[:, i], 2) for i in range(2)], -1)
    return t, m1, m2, box


def _heading(xy, partner, face, t):
    """Heading from the direction of travel (held while nearly still), turned to face the
    partner where `face` (0..1) says so."""
    n = len(xy)
    v = (np.roll(xy, -1, 0) - np.roll(xy, 1, 0)) * FPS / 2
    sp = np.linalg.norm(v, axis=1)
    th = np.arctan2(v[:, 1], v[:, 0])
    held = th.copy()
    first = int(np.argmax(sp > 0.03))
    cur = th[first]
    for i in list(range(first, n)) + list(range(0, first)):
        if sp[i] > 0.03:
            cur = th[i]
        held[i] = cur
    to_partner = np.arctan2(partner[:, 1] - xy[:, 1], partner[:, 0] - xy[:, 0])
    d = np.angle(np.exp(1j * (to_partner - held)))
    th = held + face * d
    th = periodic_smooth_angle(th, 3)
    return th, sp


def social_mouse(xy, th, sp, w, agitation, rng):
    n = len(xy)
    dth = np.angle(np.exp(1j * (np.roll(th, -1) - np.roll(th, 1)))) * FPS / 2
    kappa = np.clip(dth / np.maximum(sp, 0.05), -40, 40)
    still = sp < 0.02
    prog = np.full(n, -1.0)
    i = 0
    while i < n:   # progress 0..1 through each still stretch (drives head scanning)
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            prog[i:j] = np.linspace(0, 1, j - i) if j - i > 1 else 0.5
            i = j
        else:
            i += 1
    tr = dict(xy=xy, theta=th, kappa=kappa, speed=sp, rear=w["rear"], still=still, prog=prog, n=n)
    tr["pose"] = pose_from_weights(n, w, rng, agitation)
    return mouse_frames(tr)


def make_social(rng):
    t, m1, m2, box = social_paths(rng)
    n = len(t)
    face = np.maximum(_window(t, 11.0, 12.0, 0.2), box)
    th1, sp1 = _heading(m1, m2, face, t)
    th2, sp2 = _heading(m2, m1, face, t)
    fight_w = _window(t, 12.0, 17.0, 0.3)
    w1 = {"rear": np.maximum(_window(t, 5.0, 6.4), np.maximum(_window(t, 24.0, 25.4), box)),
          "groom": np.zeros(n), "stretch": _window(t, 6.5, 7.6)}
    w2 = {"rear": box, "groom": _window(t, 23.0, 26.0), "stretch": np.zeros(n)}
    f1 = social_mouse(m1, th1, sp1, w1, fight_w, rng)
    f2 = social_mouse(m2, th2, sp2, w2, fight_w, rng)
    frames = np.stack([f1, f2], axis=1)              # [frame][animal][joint][xyz]
    report = {}
    for a, fa in enumerate((f1, f2)):
        rep = validate(fa, MOUSE_JOINTS, RIGID_NAMES, half=SOCIAL_REGION / 2)
        report[f"mouse_{a + 1}"] = rep
    noisy = frames + rng.normal(0.0, NOISE_SD, size=frames.shape)
    report["noise_sd_m"] = NOISE_SD
    dist = np.linalg.norm(m1 - m2, axis=1)
    report["centroid_distance_m"] = {"min": float(dist.min()), "median": float(np.median(dist)), "max": float(dist.max())}
    events = {
        "investigating": (t >= 7.6) & (t < 12.0),
        "fighting": (t >= 12.0) & (t < 17.0),
        "chasing": (t >= 17.0) & (t < 19.2),
    }
    return noisy, events, report


def social_dataset(frames, events, report):
    clip = {
        "name": "pair",
        "fps": FPS,
        "frame_count": len(frames),
        "duration_s": round(len(frames) / FPS, 4),
        "loop": True,
        "root_motion_per_loop": [0.0, 0.0, 0.0],
        "positions": np.round(frames, 5).tolist(),   # [frame][animal][joint][xyz]
        "events": {k: [int(x) for x in v] for k, v in events.items()},
        "script": [{"start_s": a, "end_s": b, "what": w} for a, b, w in SOCIAL_SCRIPT],
        "validation": report,
    }
    return {
        "format": "pose-testset",
        "version": "1.1",
        "species": "mouse",
        "description": "Two synthetic 3D mice in a 60 cm region for 30 s: exploring apart, investigating "
                       "(following, nose-to-nose), fighting (circling and boxing), chasing, then apart again; "
                       f"white noise (sd {NOISE_SD * 1000:.0f} mm) on every node.",
        "dimensions": 3,
        "units": "meters",
        "coordinate_system": {"handedness": "right", "up": "+Y", "floor_plane": "XZ", "ground_plane": "y = 0",
                              "origin": "region center"},
        "animals": [{"name": "mouse 1"}, {"name": "mouse 2"}],
        "skeleton": skeleton_block(MOUSE_JOINTS, frames[0, 0]),
        "clips": [clip],
    }


def inset(waypoints, k=0.66):
    """Pull a body-root path toward the center so the snout and tail tip stay inside the region on turns."""
    return [(x * k, y * k) for x, y in waypoints]


def main():
    # --- 3D: loop around the edge with a crossing; rears three times, pauses once in the middle
    wp3 = [(-0.165, -0.15), (-0.05, -0.168), (0.10, -0.165), (0.165, -0.12), (0.168, 0.03),
           (0.15, 0.16), (0.03, 0.08), (-0.04, 0.0), (-0.10, 0.10), (-0.16, 0.165), (-0.168, 0.02)]
    tr3 = trajectory(inset(wp3), stops=[(3, 2.2, "rear"), (5, 2.0, "rear"), (7, 1.0, "pause"), (9, 2.4, "rear")],
                     speeds=[0.12, 0.10, 0.14, 0.09, 0.11])
    f3 = mouse_frames(tr3)
    rigid = RIGID_NAMES
    rep3 = validate(f3, MOUSE_JOINTS, rigid)
    write("mouse_3d.json", dataset(
        "explore", "Ideal synthetic 3D mouse skeleton running around and rearing on its hind legs.",
        3, MOUSE_JOINTS, f3, {"rearing": tr3["rear"] > 0.5}, rep3))

    # --- 2D: a different loop, no rearing (top-down view)
    wp2 = [(-0.15, -0.16), (0.0, -0.06), (0.15, -0.16), (0.168, 0.0), (0.12, 0.15),
           (0.0, 0.06), (-0.12, 0.15), (-0.168, 0.0)]
    tr2 = trajectory(inset(wp2), stops=[(3, 0.8, "pause"), (6, 1.2, "pause")], speeds=[0.13, 0.10, 0.15])
    f2 = project_2d(mouse_frames(tr2))
    rep2 = validate(f2, MOUSE_2D_JOINTS, RIGID_NAMES - {"tail_base"})
    write("mouse_2d.json", dataset(
        "run", "Ideal synthetic 2D (top-down) mouse skeleton running loops.",
        2, MOUSE_2D_JOINTS, f2, None, rep2))

    # --- centroid: a single point per frame, its own path
    wpc = [(-0.16, -0.16), (0.16, -0.165), (0.165, 0.16), (0.0, 0.0), (-0.165, 0.16)]
    trc = trajectory(inset(wpc, 0.8), stops=[(1, 1.0, "pause"), (3, 1.5, "pause")], speeds=[0.12, 0.09, 0.14])
    fc = trc["xy"][:, None, :]
    repc = validate(fc, [("centroid", None)], set())
    write("mouse_centroid.json", dataset(
        "track", "Ideal synthetic mouse centroid (x, y) track: the simplest tracking output.",
        2, [("centroid", None)], fc, None, repc))

    # --- large: 30 s over an 80 cm region, varied behaviour, white noise on every node
    fb, evb, repb = make_big(np.random.default_rng(20261107))
    write("mouse_3d_large.json", dataset(
        "explore_long", f"Synthetic 3D mouse skeleton, {BIG_SECONDS:.0f} s covering an 80 cm region near uniformly, with "
        f"running, rearing, grooming, stretched-attend and head scanning, plus white noise (sd {NOISE_SD * 1000:.0f} mm) "
        "on every node.", 3, MOUSE_JOINTS, fb, evb, repb))

    # --- two mice: apart, investigating, fighting, chasing, apart again
    fs, evs, reps = make_social(np.random.default_rng(42))
    write("mouse_social.json", social_dataset(fs, evs, reps))


if __name__ == "__main__":
    main()
