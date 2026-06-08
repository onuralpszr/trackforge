# OC-SORT

**Observation-Centric SORT** ([arXiv 2203.14360](https://arxiv.org/abs/2203.14360), CVPR 2023).
Extends SORT with three observation-centric mechanisms that reduce drift during occlusions:

- **OCV** computes velocity from consecutive detections rather than the Kalman state.
- **OCM** adds a direction-consistency bonus before matching: a candidate whose direction from the
  track's last observation aligns with the track's velocity gets a higher effective IoU.
- **ORU** corrects the Kalman filter after re-association by replaying interpolated observations
  between the last seen position and the current detection.

No appearance features required, making it a strong upgrade over SORT when occlusions are common but
Re-ID is unavailable.

```rust
use trackforge::trackers::ocsort::OcSort;

let mut tracker = OcSort::new(30, 3, 0.3, 3, 0.2);
let tracks = tracker.update(vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)]);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Parameters

| Parameter       | Default | Description                                                |
| --------------- | ------- | ---------------------------------------------------------- |
| `max_age`       | 30      | Frames to keep a lost track alive before deletion          |
| `min_hits`      | 3       | Consecutive matched frames required to confirm a track     |
| `iou_threshold` | 0.3     | Minimum IoU to associate a detection with a track          |
| `delta_t`       | 3       | Observation window (frames) used to compute velocity (OCV) |
| `inertia`       | 0.2     | Weight of the direction-consistency cost bonus (OCM)       |

**Tuning:** raise `max_age` for long occlusions; raise `delta_t` for smoother velocity at the cost
of responsiveness; raise `inertia` toward 1.0 for near-constant-velocity motion, lower it for
erratic motion.
