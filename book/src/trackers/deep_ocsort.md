# Deep OC-SORT

**Deep OC-SORT** ([arXiv 2302.11813](https://arxiv.org/abs/2302.11813), ICIP 2023). Extends OC-SORT
with appearance, blending a Re-ID embedding cost into the motion association:

- **OCM** adds a velocity direction-consistency bonus to the IoU before matching.
- **ORU** replays interpolated observations to correct the Kalman filter after a track is
  re-associated following a gap.
- **Appearance** adds a cosine distance to each track's feature gallery. The appearance weight
  scales with detector confidence (dynamic appearance) and is gated by `max_cosine_distance`.
- **Camera motion compensation** warps track predictions by a caller-supplied affine transform
  before association, for moving-camera footage.

With `appearance_weight = 0` the association reduces to plain OC-SORT, so the appearance term is a
strict add-on. This is a clean-room implementation. CMC is applied from a transform you supply (for
example estimated with OpenCV); the tracker does not estimate camera motion itself, which keeps the
core free of heavy dependencies. Pass the affine as `[a, b, tx, c, d, ty]` to `update`.

```python
from trackforge import DEEPOCSORT

tracker = DEEPOCSORT(max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2,
                     appearance_weight=0.5, max_cosine_distance=0.2, nn_budget=100)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection
tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Parameters

| Parameter             | Default | Description                                                |
| --------------------- | ------- | ---------------------------------------------------------- |
| `max_age`             | 30      | Frames to keep a lost track alive before deletion          |
| `min_hits`            | 3       | Consecutive matched frames required to confirm a track     |
| `iou_threshold`       | 0.3     | Minimum IoU to associate a detection with a track          |
| `delta_t`             | 3       | Observation window (frames) used to compute velocity (OCM) |
| `inertia`             | 0.2     | Weight of the direction-consistency cost bonus (OCM)       |
| `appearance_weight`   | 0.5     | Blend weight of the appearance cost, scaled by det. score  |
| `max_cosine_distance` | 0.2     | Maximum cosine distance for the appearance term to apply   |
| `nn_budget`           | 100     | Maximum appearance features stored per track               |

**Tuning:** raise `appearance_weight` when the Re-ID model is reliable and identities matter; lower
it (toward 0) to fall back to OC-SORT motion. Tighten `max_cosine_distance` to only trust strong
appearance matches. The motion parameters behave as in OC-SORT.

## Citation

```bibtex
@inproceedings{maggiolino2023deepocsort,
  title={Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification},
  author={Maggiolino, Gerard and Ahmad, Adnan and Cao, Jinkun and Kitani, Kris},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2023}
}
```
