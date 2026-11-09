# etho-clusters 1.0 — clustering output format

One JSON file holds the result of one clustering run of the pose viewer on one clip: every animal's
state in every frame, its bouts and its position in PC space, plus everything needed to interpret
the states (the model, the reduction and the input features). The viewer writes it with
**Export → Clusters (.json)**; the analysis notebook export ships it next to the pose data
(etho-pose, see `POSE_FORMAT.md`).

State ids are **1-based** everywhere (state 1 = "Cluster 1" in the viewer). Frame indices are
**0-based** within the clip. Lengths are in `space.units` (default metres) unless noted.

## Top level

```jsonc
{
  "format": "etho-clusters",       // exactly this
  "version": "1.0",                // readers accept any 1.x
  "created": "2026-11-08T15:04:05Z",
  "generator": "Pose Dataset Viewer",
  "source":    { ... },            // which data was clustered
  "space":     { "dims": 3, "units": "m" },
  "features":  { ... },            // what went into the model
  "reduction": { ... },            // PCA (or none)
  "model":     { ... },            // switching linear dynamics: states, transitions, k selection
  "animals":   [ { ... } ]         // per-animal labels, bouts, embedding
}
```

### `source`
| field | type | |
|---|---|---|
| `dataset` | string | dataset name (`meta.name` of the pose file) |
| `file` | string \| null | file name the pose data was loaded from |
| `clip` | object | `index` (into the pose file's `clips`), `name`, `fps`, `frame_count`, `loop` |

### `features`
Each sample is one animal in one frame, expressed in that animal's egocentric frame (centroid at the
origin, body axis forward).

| field | type | |
|---|---|---|
| `mode` | `"ego"` \| `"social"` | ego: own joints only; social: own joints + every other animal's joints in the focal frame |
| `nodes` | `[string]` | skeleton nodes, in feature order |
| `removed_nodes` | `[string]` | nodes excluded from the analysis (their columns are zero) |
| `axes` | `[string]` | per-joint coordinates: `forward, up, lateral` (3D) or `forward, lateral` (2D) |
| `egocentric_axis` | object | `anterior`, `posterior`: the nodes defining the body axis |
| `align_others` | bool | social: each other animal rotated about the focal centroid onto the body axis (in front or behind) |
| `proximity_kernel` | object \| null | social: `{type: "gaussian", d0, d0_body_lengths, dims, distance}`; each other animal scaled by exp(−½ (d/d0)²), an isotropic Gaussian centred on the focal centroid, where d is the centroid-to-centroid distance in `dims` (3: x, height, z; 2: the floor) dimensions |
| `distance_dims` | 2 \| 3 | social: dimensions of the centroid distance used for the kernel and for ordering other animals nearest-first |
| `body_length` | number | the length unit `d0_body_lengths` refers to |
| `feature_names` | `[string]` | one per feature column: `own/<node>/<axis>` or `other<i>/<node>/<axis>` |

### `reduction`
| field | type | |
|---|---|---|
| `method` | `"pca"` \| `"none"` | `none`: the model was fit on the centred raw features |
| `n_components` | int | dimensions of the model space (PCs kept, or raw features used) |
| `embedding_components` | int | PCs stored in `animals[].embedding` (always PCA, even for `none`) |
| `explained_variance_ratio` | `[number]` | per PC (at least the kept ones) |
| `cumulative_variance` | `[number]` | running sum of the above |
| `loadings` | `[[number]]` | `embedding_components` unit vectors over the feature columns |
| `mean` | `[number]` | feature means subtracted before PCA |

PC scores satisfy `score_k = (x − mean) · loadings[k]` for a feature row `x`.

### `model`
Switching linear dynamics (an autoregressive HMM) in the model space: in state *c*,
`z(t+1) = A_c z(t) + b_c + ε`, `ε ~ N(0, noise_cov_c)`, and a Markov chain switches between states.

| field | type | |
|---|---|---|
| `method` | `"switching_linear_dynamics"` | |
| `space` | `"pca"` \| `"raw_features"` | where `z` lives |
| `dims` | int | length of `z` |
| `k`, `k_auto` | int | states used, and the k with the lowest BIC (they differ if k was picked by hand) |
| `criterion` | `"BIC"` | |
| `k_selection` | `[{k, bic}]` | every k that was fit |
| `hyperparameters` | object | `k_min`, `k_max`, `stickiness`, `max_em_iterations` |
| `log_likelihood_per_step`, `bic`, `em_iterations`, `converged` | | fit of the chosen k |
| `transition_matrix` | `[[number]]` | k × k, row = from, per frame |
| `initial_distribution` | `[number]` | k |
| `states` | `[state]` | see below |

**state**

| field | type | |
|---|---|---|
| `id` | int | 1-based |
| `color` | string | the viewer's color for the state |
| `frames`, `share`, `bouts`, `mean_bout_s` | | occupancy over all animals |
| `self_transition` | number | diagonal of the transition matrix |
| `A`, `b` | `[[number]]`, `[number]` | the state's flow field (only when `space` is `pca`; `null` otherwise) |
| `noise_cov` | `[[number]]` | residual covariance (`pca` only) |
| `fixed_point` | `[number]` \| null | z* = (I − A)⁻¹ b, or null if I − A is singular (`pca` only) |

For `space: "raw_features"` the per-state parameters are omitted (they are large); labels, bouts and
the embedding are still complete.

### `animals`
| field | type | |
|---|---|---|
| `id`, `name` | string | as in the pose file |
| `labels` | `[int]` | state of every frame (`frame_count` values, 1-based) |
| `bouts` | `[{state, start, end, duration_s}]` | unbroken runs of one state; `start`/`end` inclusive frame indices |
| `embedding` | `[[number]]` | `frame_count` rows of `embedding_components` PC scores |

## Labels CSV
**Export → Labels (.csv)** writes the same labels as a long table:

```
animal_id,animal,frame,time_s,state
mouse-1,mouse 1,0,0.0000,3
```

## Reading it

```python
import json
cl = json.load(open('session_clusters.json'))
for a in cl['animals']:
    print(a['name'], 'spends', sum(1 for s in a['labels'] if s == 1) / len(a['labels']), 'of the time in state 1')
```
