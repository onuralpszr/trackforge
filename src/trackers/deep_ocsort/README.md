# Deep OC-SORT: Observation-Centric SORT with appearance

This module implements the Deep OC-SORT algorithm.

> **Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification**
> Gerard Maggiolino, Adnan Ahmad, Jinkun Cao, Kris Kitani
> [arXiv:2302.11813](https://arxiv.org/abs/2302.11813)

## Algorithm overview

Deep OC-SORT extends OC-SORT by adding an appearance term to the association:

- **OCM** adds a velocity direction-consistency bonus to the IoU before matching.
- **ORU** replays interpolated observations to correct the Kalman filter after a track
  is re-associated following a gap.
- **Appearance** association blends a cosine distance to each track's feature gallery
  with the motion cost. The appearance weight scales with detector confidence (dynamic
  appearance) and is gated by `max_cosine_distance`. With `appearance_weight = 0` the
  association reduces to plain OC-SORT.

This is a clean-room implementation focused on the motion (OCM/ORU) and adaptive
appearance association; it does not include camera motion compensation.

## Builds on

- `utils::kalman` - the shared 8-dimensional Kalman filter
- `utils::geometry` - `iou_batch`, `tlwh_to_xyah`, `xyah_to_tlwh`
- `utils::assignment` - `greedy_match`
- `trackers::common` - `KalmanTrack` and `TrackState`
- `trackers::deepsort` - `NearestNeighborDistanceMetric` for the cosine feature gallery

## Parameters

| Parameter             | Default | Description                                                |
| --------------------- | ------- | ---------------------------------------------------------- |
| `max_age`             | 30      | Frames a lost track survives before deletion               |
| `min_hits`            | 3       | Consecutive matches required to confirm a track            |
| `iou_threshold`       | 0.3     | Minimum IoU to associate a detection with a track          |
| `delta_t`             | 3       | Observation window (frames) used to compute velocity (OCM) |
| `inertia`             | 0.2     | Weight of the direction-consistency cost bonus (OCM)       |
| `appearance_weight`   | 0.5     | Blend weight of the appearance cost, scaled by det. score  |
| `max_cosine_distance` | 0.2     | Maximum cosine distance for the appearance term to apply   |
| `nn_budget`           | 100     | Maximum appearance features stored per track               |

## Rust API

```rust,ignore
use trackforge::trackers::deep_ocsort::DeepOcSort;

// `extractor` implements the AppearanceExtractor trait (plug in any Re-ID model).
let mut tracker = DeepOcSort::new(extractor, 30, 3, 0.3, 3, 0.2, 0.5, 0.2, 100);
let tracks = tracker.update(&image, detections)?;
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Python API

```python
from trackforge import DEEPOCSORT

tracker = DEEPOCSORT(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    delta_t=3,
    inertia=0.2,
    appearance_weight=0.5,
    max_cosine_distance=0.2,
    nn_budget=100,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection
tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Credit

Clean-room Rust implementation of the algorithm described in the paper above. Original
reference implementation:
[GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT).

## Citation

```bibtex
@inproceedings{maggiolino2023deepocsort,
  title={Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification},
  author={Maggiolino, Gerard and Ahmad, Adnan and Cao, Jinkun and Kitani, Kris},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2023}
}
```
