# TrackTrack

**TrackTrack** ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)). A track-centric online tracker built on a ByteTrack-style two-stage lifecycle with two contributions.

- **Track-perspective association.** Instead of one global assignment, each track picks its own best detection and a pair matches only when the choice is mutual. The loop repeats with a gate that tightens each round. High and low confidence detections share one pass, with low ones carrying a penalty rather than running as a separate stage. The cost fuses a height-modulated IoU, an optional appearance term, a confidence projection, and a velocity-direction term.
- **Track-aware initialization.** A leftover detection starts a new track only if it clears an init threshold and does not overlap an existing active track, or a more confident leftover, by too much.

Appearance is optional. Pass embeddings to use the Re-ID term, or an empty slice to track on motion only.

This port keeps the two contributions and the fused cost. It uses the shared 8-dimensional Kalman filter, a simplified velocity-direction term, and does not reproduce the paper's detector-level NMS recovery pool, which needs access to the detector's suppressed boxes.

```rust,ignore
use trackforge::trackers::tracktrack::TrackTrack;

let mut tracker = TrackTrack::new();

let detections = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
let tracks = tracker.update(detections, &[]);
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```
