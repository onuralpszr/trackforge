# ByteTrack

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
>
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
>
> _[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)_

**ByteTrack** is a simple, fast and strong multi-object tracker.

## Abstract

Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, ByteTrack presents a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, it utilizes their similarities with tracklets to recover true objects and filter out the background detections.

## Builds on

- `utils::kalman` - the shared 8-dimensional Kalman filter
- `utils::geometry` - `iou_cost_matrix`, `tlwh_to_xyah`
- `utils::assignment` - `greedy_match`, `iou_match` (via the shared cascade)
- `trackers::common` - the `byte_cascade` two-stage matching core, `CommonParams`, and `KalmanTrack`

## Parameters

| Parameter             | Default | Description                                                         |
| --------------------- | ------- | ------------------------------------------------------------------- |
| `track_thresh`        | 0.5     | Confidence threshold separating high- and low-confidence detections |
| `track_buffer`        | 30      | Frames a lost track is buffered before deletion                     |
| `match_thresh`        | 0.8     | Stage-1 match cutoff as a maximum IoU distance (lower is stricter)  |
| `det_thresh`          | 0.6     | Minimum confidence to initialise a new track                        |
| `second_match_thresh` | 0.5     | Stage-2 match cutoff for recovering low-confidence detections       |

**Tuning:** lower `track_thresh` (~0.3) to feed more low-confidence detections into stage two; raise
`track_buffer` (~60) when the detector drops frames; raise `match_thresh` (~0.9) for fast-moving
objects where IoU falls off quickly.

## Rust API

```rust
use trackforge::trackers::byte_track::ByteTrack;

let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
let tracks = tracker.update(vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)]);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Python API

```python
from trackforge import BYTETRACK

tracker = BYTETRACK(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]

for track_id, tlwh, score, class_id, det_ind in tracker.update(detections):
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Original Repository

This is a clean-room Rust implementation of the ByteTrack algorithm as described in the original paper. The official reference implementation can be found at [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack).

## Citation

```bibtex
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
