"""
Synthesize a standardized, ideal (noise-free) 3D skeleton dataset for a
generic medium-sized quadruped (dog-like).

Guarantees ("ideal" properties, verified by validate()):
  * Every bone keeps exactly the same length in every frame.
  * Feet in stance phase have zero world-space velocity (no foot sliding).
  * Paws in stance touch the ground plane (y = 0) exactly.
  * Walk clip is periodic: last frame + root offset == first frame.

Coordinate system: right-handed, meters, +X forward, +Y up, +Z to the animal's right.
Output: skeleton_dataset.json
"""
import json
from pathlib import Path

import numpy as np

FPS = 30

# ---------------------------------------------------------------- skeleton
# (name, parent)  -- order defines the joint index used in every frame
JOINTS = [
    ("pelvis", None),
    ("spine_mid", "pelvis"),
    ("chest", "spine_mid"),
    ("neck", "chest"),
    ("head", "neck"),
    ("nose", "head"),
    ("tail_base", "pelvis"),
    ("tail_mid", "tail_base"),
    ("tail_tip", "tail_mid"),
]
for side in ("L", "R"):
    JOINTS += [
        (f"shoulder_{side}", "chest"),
        (f"elbow_{side}", f"shoulder_{side}"),
        (f"wrist_{side}", f"elbow_{side}"),
        (f"front_paw_{side}", f"wrist_{side}"),
        (f"hip_{side}", "pelvis"),
        (f"knee_{side}", f"hip_{side}"),
        (f"hock_{side}", f"knee_{side}"),
        (f"hind_paw_{side}", f"hock_{side}"),
    ]
NAMES = [j[0] for j in JOINTS]
IDX = {n: i for i, n in enumerate(NAMES)}

# Body proportions (meters)
P = dict(
    pelvis_height=0.55,
    hip_offset=np.array([0.0, -0.05, 0.09]),       # z sign flipped for left
    shoulder_offset=np.array([0.0, -0.08, 0.07]),
    thigh=0.22, shin=0.22,
    hock_to_paw=np.array([0.03, -0.116, 0.0]),
    upper_arm=0.23, forearm=0.23,
    wrist_to_paw=np.array([0.02, -0.077, 0.0]),
    spine_mid=np.array([0.20, 0.02, 0.0]),
    chest=np.array([0.20, 0.01, 0.0]),
    neck=np.array([0.08, 0.10, 0.0]),
    head=np.array([0.10, 0.12, 0.0]),
    nose=np.array([0.12, -0.03, 0.0]),
    tail1=np.array([-0.06, 0.02, 0.0]),
    tail2=np.array([-0.15, 0.05, 0.0]),
    tail3=np.array([-0.14, 0.02, 0.0]),
)

# Gait (lateral-sequence walk)
GAIT = dict(speed=0.5, period=0.8, duty=0.6, swing_height=0.06,
            phase={"hind_L": 0.0, "front_L": 0.25, "hind_R": 0.5, "front_R": 0.75})


def side_sign(side):
    return -1.0 if side == "L" else 1.0


def rot_z(v, a):  # pitch (rotation in sagittal plane)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def rot_y(v, a):  # yaw (lateral swing)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def two_bone_ik(root, target, l1, l2, bend_forward):
    """Planar (XY) two-bone IK; returns middle joint position."""
    d_vec = target - root
    d = np.linalg.norm(d_vec)
    d = np.clip(d, abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6)
    u = d_vec / np.linalg.norm(d_vec)
    n = np.array([-u[1], u[0], 0.0])  # perpendicular, points forward when u points down
    if not bend_forward:
        n = -n
    a = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    return root + l1 * (np.cos(a) * u + np.sin(a) * n)


def foot_offset(phase, amp):
    """Hock/wrist target relative to neutral, in body frame: (dx, height)."""
    duty = GAIT["duty"]
    if phase < duty:                          # stance: slide back at exactly -speed
        return amp - 2 * amp * phase / duty, 0.0, True
    s = (phase - duty) / (1 - duty)           # swing: smooth return + arc
    return -amp + 2 * amp * (1 - np.cos(np.pi * s)) / 2, GAIT["swing_height"] * np.sin(np.pi * s), False


def pose(t, walking):
    J = np.zeros((len(NAMES), 3))
    contacts = {}
    T, v = GAIT["period"], GAIT["speed"]
    amp = v * GAIT["duty"] * T / 2            # makes stance foot stationary in world
    w = 2 * np.pi * t / T

    root_x = v * t if walking else 0.0
    bob = 0.008 * np.cos(2 * w) if walking else 0.0
    J[IDX["pelvis"]] = [root_x, P["pelvis_height"] + bob, 0.0]
    J[IDX["spine_mid"]] = J[IDX["pelvis"]] + P["spine_mid"]
    J[IDX["chest"]] = J[IDX["spine_mid"]] + P["chest"]
    J[IDX["neck"]] = J[IDX["chest"]] + P["neck"]
    head_pitch = 0.05 * np.sin(2 * w) if walking else 0.0
    J[IDX["head"]] = J[IDX["neck"]] + rot_z(P["head"], head_pitch)
    J[IDX["nose"]] = J[IDX["head"]] + rot_z(P["nose"], head_pitch)
    tail_yaw = 0.35 * np.sin(w) if walking else 0.0
    J[IDX["tail_base"]] = J[IDX["pelvis"]] + P["tail1"]
    J[IDX["tail_mid"]] = J[IDX["tail_base"]] + rot_y(P["tail2"], tail_yaw)
    J[IDX["tail_tip"]] = J[IDX["tail_mid"]] + rot_y(P["tail3"], 1.6 * tail_yaw)

    for side in ("L", "R"):
        sg = side_sign(side)
        for limb in ("hind", "front"):
            if limb == "hind":
                top, mid, low, paw = f"hip_{side}", f"knee_{side}", f"hock_{side}", f"hind_paw_{side}"
                base, off, l1, l2, tip = "pelvis", P["hip_offset"], P["thigh"], P["shin"], P["hock_to_paw"]
                bend_fwd = True
            else:
                top, mid, low, paw = f"shoulder_{side}", f"elbow_{side}", f"wrist_{side}", f"front_paw_{side}"
                base, off, l1, l2, tip = "chest", P["shoulder_offset"], P["upper_arm"], P["forearm"], P["wrist_to_paw"]
                bend_fwd = False
            root = J[IDX[base]] + off * np.array([1, 1, sg])
            J[IDX[top]] = root
            neutral_x = root[0] - tip[0]
            if walking:
                ph = (t / T + GAIT["phase"][f"{limb}_{side}"]) % 1.0
                dx, lift, stance = foot_offset(ph, amp)
            else:
                dx, lift, stance = 0.0, 0.0, True
            target = np.array([neutral_x + dx, -tip[1] + lift, root[2]])
            J[IDX[low]] = target
            J[IDX[mid]] = two_bone_ik(root, target, l1, l2, bend_fwd)
            J[IDX[paw]] = target + tip
            contacts[paw] = stance
    return J, contacts


def make_clip(name, seconds, walking):
    n = int(round(seconds * FPS))
    frames, contacts = [], {k: [] for k in NAMES if "paw" in k}
    for f in range(n):
        J, c = pose(f / FPS, walking)
        frames.append(J)
        for k, val in c.items():
            contacts[k].append(int(val))
    return name, np.array(frames), contacts, n


def validate(frames, contacts, bones):
    report = {}
    lengths = np.array([[np.linalg.norm(fr[c] - fr[p]) for p, c in bones] for fr in frames])
    report["max_bone_length_drift_m"] = float((lengths.max(0) - lengths.min(0)).max())
    worst_slide, worst_ground = 0.0, 0.0
    for paw, flags in contacts.items():
        i = IDX[paw]
        for f in range(1, len(frames)):
            if flags[f] and flags[f - 1]:
                worst_slide = max(worst_slide, np.linalg.norm(frames[f, i] - frames[f - 1, i]))
            if flags[f]:
                worst_ground = max(worst_ground, abs(frames[f, i, 1]))
    report["max_stance_foot_slide_m"] = float(worst_slide)
    report["max_stance_ground_error_m"] = float(worst_ground)
    return report


def main():
    bones = [(IDX[p], IDX[c]) for c, p in JOINTS if p is not None]
    rest, _ = pose(0.0, walking=False)

    cycles = 4
    clips_spec = [("stand", 1.0, False), ("walk", GAIT["period"] * cycles, True)]
    clips = []
    for name, secs, walking in clips_spec:
        name, frames, contacts, n = make_clip(name, secs, walking)
        rep = validate(frames, contacts, bones)
        assert rep["max_bone_length_drift_m"] < 1e-9, rep
        assert rep["max_stance_foot_slide_m"] < 1e-9, rep
        assert rep["max_stance_ground_error_m"] < 1e-9, rep
        print(f"{name:6s} frames={n:3d}  {rep}")
        clips.append({
            "name": name,
            "fps": FPS,
            "frame_count": n,
            "duration_s": n / FPS,
            "loop": True,
            "root_motion_per_loop": [GAIT["speed"] * secs if walking else 0.0, 0.0, 0.0],
            "gait": ({k: v for k, v in GAIT.items()} if walking else None),
            "foot_contacts": contacts,  # 1 = paw planted on ground
            "positions": np.round(frames, 6).tolist(),  # [frame][joint][xyz]
            "validation": rep,
        })

    data = {
        "format": "quadruped-skeleton-testset",
        "version": "1.0",
        "description": "Ideal synthetic 3D joint positions for a generic dog-sized quadruped.",
        "units": "meters",
        "coordinate_system": {"handedness": "right", "forward": "+X", "up": "+Y",
                              "right": "+Z", "ground_plane": "y = 0"},
        "skeleton": {
            "joint_count": len(NAMES),
            "joints": [{"index": i, "name": n, "parent": (IDX[p] if p else -1)}
                       for i, (n, p) in enumerate(JOINTS)],
            "bones": [[p, c] for p, c in bones],
            "bone_lengths_m": [round(float(np.linalg.norm(rest[c] - rest[p])), 6) for p, c in bones],
            "rest_pose": np.round(rest, 6).tolist(),
        },
        "clips": clips,
    }
    out_path = Path(__file__).resolve().parent.parent / "skeleton_dataset.json"   # examples/
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
