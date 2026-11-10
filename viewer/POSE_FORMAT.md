# poser-pose 1.0 — pose data format

Files saying `"format": "etho-pose"` (the project's earlier name) are the same format; the viewer and
`poser.py` read them unchanged.

One JSON file holds one dataset: any number of animals, each tracked with some skeleton (a
single centroid point up to a full 3D skeleton), over one or more clips, with zero or more
videos. The reference importer/validator is `viewer/pose_format.js` (`PoseFormat.load`,
`PoseFormat.validate`); it also converts the legacy `pose-testset` / `quadruped-skeleton-testset`
files (`examples/mouse_*.json`, `examples/skeleton_dataset.json`).

## Top level

```jsonc
{
  "format": "poser-pose",        // required, exactly this
  "version": "1.0",             // required; readers accept any 1.x
  "meta":      { ... },         // optional, free-form description
  "space":     { ... },         // required, coordinate frame
  "skeletons": { "<id>": { ... } },   // required, ≥ 1
  "animals":   [ { ... } ],     // required, ≥ 1
  "videos":    [ { ... } ],     // optional, may be empty or absent
  "clips":     [ { ... } ]      // required, ≥ 1
}
```

### `meta` (optional)
| field | type | |
|---|---|---|
| `name` | string | short display name (the viewer falls back to the file name) |
| `description` | string | |
| `species` | string | default species for animals that don't name one |
| `source` | string | where the data came from (tracker, conversion, ...) |
| anything else | any | kept as-is |

### `space`
| field | type | |
|---|---|---|
| `dims` | 2 \| 3 | **required.** Coordinates per point. |
| `units` | string | length unit of every coordinate, e.g. `"m"`, `"mm"`, `"px"`. Default `"m"`. |
| `up` | `"+Y"`, `"+Z"`, `"-Y"`, ... | 3D only: the vertical axis. Default `"+Y"` (right-handed). |
| `view` | string | 2D only: e.g. `"top-down"`. 2D points are `(x, y)` on the floor plane. |
| `origin` | string | free-text description of the origin |

### `skeletons`
A map from skeleton id to a skeleton. Several animals can share one skeleton.

| field | type | |
|---|---|---|
| `joints` | `[{ "name": str, "parent": int \| null }]` | **required**, ≥ 1. Names unique within the skeleton. A centroid-only track is a skeleton with one joint (e.g. `"centroid"`). |
| `bones` | `[[i, j], ...]` | segments to draw, as joint indices. Optional (none for a centroid). |
| `rest_pose`, `bone_lengths` | | optional reference data |

### `animals`
| field | type | |
|---|---|---|
| `id` | string | **required**, unique; the key used in `clips[].tracks` |
| `name` | string | display name (defaults to `id`) |
| `skeleton` | string | **required**, a key of `skeletons` |
| `species`, `sex`, `group`, ... | any | optional metadata |

### `videos` (optional)
Source recordings. A dataset with no video uses `[]` or leaves the field out.

| field | type | |
|---|---|---|
| `id` | string | **required**, unique |
| `uri` | string | **required**; a path relative to the JSON file, or a URL |
| `fps` | number | video frame rate (may differ from the pose rate) |
| `camera` | string | camera name/view |
| `width`, `height` | int | pixels |
| `animals` | `[id]` | animals visible in it, if not all |

### `clips`
A clip is a continuous stretch of pose samples at a fixed rate.

| field | type | |
|---|---|---|
| `name` | string | |
| `fps` | number | **required**, pose sample rate |
| `frame_count` | int | **required**, frames per track (`n` below) |
| `timestamps` | `[n numbers]` | optional seconds per frame, for irregular sampling |
| `loop` | bool | the clip repeats seamlessly (synthetic data). Default false. |
| `root_motion_per_loop` | `[dims numbers]` | displacement added on every loop. Default zero. |
| `tracks` | `{ animalId: track }` | **required**, ≥ 1. Animals absent from a clip just have no track. |
| `events` | `[event]` | optional behaviour annotations |
| `videos` | `[{ "id": videoId, "frame_offset": int }]` | videos covering this clip. Video frame 0 lines up with pose frame `frame_offset`. |
| `extra` | object | anything else (validation stats, scripts, gait parameters...) |

**track**

| field | type | |
|---|---|---|
| `positions` | `[n][J][dims]` | **required.** `J` = the animal's skeleton joint count. Use `null` for a missing joint, or for a whole missing frame. |
| `confidence` | `[n][J]` | optional per-joint scores in 0–1 (or `null`) |

**event**

| field | type | |
|---|---|---|
| `name` | string | e.g. `"rearing"` |
| `active` | `[n]` of 0/1 | per-frame on/off |
| `animal` | id \| null | the animal it belongs to; `null` = clip-wide / social |
| `joint` | string | optional, for joint-level events such as paw contacts |

## Examples

Simplest case, one animal's 2D centroid with no video:

```json
{
  "format": "poser-pose", "version": "1.0",
  "space": { "dims": 2, "units": "m", "view": "top-down" },
  "skeletons": { "point": { "joints": [{ "name": "centroid", "parent": null }] } },
  "animals": [{ "id": "m1", "skeleton": "point" }],
  "clips": [{
    "name": "session 1", "fps": 30, "frame_count": 3,
    "tracks": { "m1": { "positions": [[[0.0, 0.0]], [[0.01, 0.0]], null] } }
  }]
}
```

A group with 3D skeletons and two cameras (abridged):

```jsonc
{
  "format": "poser-pose", "version": "1.0",
  "space": { "dims": 3, "units": "m", "up": "+Z" },
  "skeletons": { "mouse18": { "joints": [ /* 18 joints */ ], "bones": [ /* ... */ ] } },
  "animals": [
    { "id": "m1", "name": "mouse 1", "skeleton": "mouse18" },
    { "id": "m2", "name": "mouse 2", "skeleton": "mouse18" }
  ],
  "videos": [
    { "id": "top",  "uri": "videos/top.mp4",  "fps": 30, "camera": "top" },
    { "id": "side", "uri": "videos/side.mp4", "fps": 60, "camera": "side" }
  ],
  "clips": [{
    "name": "pair", "fps": 30, "frame_count": 900,
    "tracks": { "m1": { "positions": [ /* 900 × 18 × 3 */ ] }, "m2": { "positions": [ /* ... */ ] } },
    "events": [{ "name": "chasing", "animal": null, "active": [ /* 900 */ ] }],
    "videos": [{ "id": "top", "frame_offset": 0 }, { "id": "side", "frame_offset": 12 }]
  }]
}
```

## Legacy conversion

`PoseFormat.load` accepts the earlier test files and converts them like this:
- `skeleton` becomes one skeleton, keyed by species. Parent `-1` becomes `null`, and `bone_lengths_m` becomes `bone_lengths`.
- `animals` keep their names, and their ids are slugged (`"mouse 1"` becomes `mouse-1`). A file without `animals` gets one animal named after its species.
- `positions` `[frame][animal][joint][coord]` is split into one track per animal.
- `events` `{name: [0/1...]}` become clip-wide events. The dog's `foot_contacts` become `contact` events, each with its `joint` set.
- `coordinate_system.up` becomes `space.up`. `dimensions` becomes `space.dims`, or is inferred from the data if missing.
- Other clip fields (`validation`, `script`, `gait`) move to `clip.extra`.

## Viewer notes

- All animals tracked in one clip must share a skeleton, because the analysis compares animals joint by joint.
- Missing samples (`null`) are linearly interpolated for display, and the viewer reports how many it filled in.
- Joint-level events are not shaded on the timeline.
- `timestamps`, `confidence` and `videos` are read and validated, but not yet used by the viewer.
