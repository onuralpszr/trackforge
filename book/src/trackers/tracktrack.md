# TrackTrack

**TrackTrack** ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)). A track-centric online tracker built on a ByteTrack-style two-stage lifecycle, with two contributions.

- **Track-perspective association.** Rather than solving one global assignment, each track picks its own best detection and a pair matches only when the choice is mutual. The loop repeats with a cost gate that tightens each round. High and low confidence detections share one pass; low ones carry a penalty instead of running as a separate stage. The cost fuses a height-modulated IoU, an optional appearance term, a confidence projection, and a velocity-direction term.
- **Track-aware initialization.** A leftover detection starts a new track only if it clears an init threshold and does not overlap an existing active track, or a more confident leftover, by more than `tai_thresh`.

Appearance is optional. Pass embeddings to use the Re-ID term, or an empty slice to track on motion only.

This port keeps the two contributions and the fused cost on the shared 8-dimensional Kalman filter. It uses a simplified velocity-direction term and does not reproduce the paper's detector-level NMS recovery pool, which needs access to the detector's suppressed boxes.

## Parameters

| Parameter | Default | Description |
| ------------- | ------- | ------------------------------------------------------------------- |
| `det_thresh` | 0.6 | Score above which a detection is high confidence |
| `match_thresh` | 0.7 | Association cost gate, lower is stricter |
| `track_buffer` | 30 | Frames a lost track is kept alive |
| `min_hits` | 3 | Matched frames in a row before a new track is confirmed |
| `init_thresh` | 0.7 | Smallest score a leftover detection needs to start a new track |
| `tai_thresh` | 0.55 | Overlap gate for track-aware initialization, a maximum IoU |
| `penalty_low` | 0.2 | Extra cost added to low confidence detections during association |
| `reduce_step` | 0.05 | How much the cost gate tightens per matching round |

## Python

```python
from trackforge import TRACKTRACK

tracker = TRACKTRACK(det_thresh=0.6, match_thresh=0.7, track_buffer=30, min_hits=3)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}")
```

## Rust

```rust,ignore
use trackforge::trackers::tracktrack::TrackTrack;

let mut tracker = TrackTrack::new();

let detections = vec![([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64)];
let tracks = tracker.update(detections, &[]);
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```
