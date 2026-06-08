# ByteTrack

**ByteTrack** ([arXiv 2110.06864](https://arxiv.org/abs/2110.06864)). A two-stage IoU tracker that
associates _every_ detection, not just high-confidence ones, to recover objects that are briefly
occluded or partially off-screen. A strong recall improvement over SORT at minimal extra cost.

```rust
use trackforge::trackers::byte_track::ByteTrack;

let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
let tracks = tracker.update(vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)]);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Parameters

| Parameter      | Default | Description                                                         |
| -------------- | ------- | ------------------------------------------------------------------- |
| `track_thresh` | 0.5     | Confidence threshold separating high- and low-confidence detections |
| `track_buffer` | 30      | Frames a lost track is buffered before deletion                     |
| `match_thresh` | 0.8     | IoU threshold for stage-1 (high-confidence) matching                |
| `det_thresh`   | 0.6     | Minimum confidence to initialise a new track                        |

**Tuning:** lower `track_thresh` (~0.3) to feed more low-confidence detections into stage two; raise
`track_buffer` (~60) when the detector drops frames; lower `match_thresh` (~0.7) for fast-moving
objects where IoU falls off quickly.
