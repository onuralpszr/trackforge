# SORT

**Simple Online and Realtime Tracking** ([arXiv 1602.00763](https://arxiv.org/abs/1602.00763)).
Pairs a Kalman filter with greedy IoU matching. Built for speed; best when objects rarely overlap.

```rust
use trackforge::trackers::sort::Sort;

let mut tracker = Sort::new(1, 3, 0.3);
let tracks = tracker.update(vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)]);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Parameters

| Parameter       | Default | Description                                            |
| --------------- | ------- | ------------------------------------------------------ |
| `max_age`       | 1       | Frames to keep a track alive without a detection match |
| `min_hits`      | 3       | Consecutive matched frames before a track is confirmed |
| `iou_threshold` | 0.3     | Minimum IoU to associate a detection with a track      |

**Tuning:** raise `max_age` to bridge short occlusions (at the cost of more false tracks); lower
`iou_threshold` for densely packed objects; raise `min_hits` to suppress false starts from a noisy
detector.
