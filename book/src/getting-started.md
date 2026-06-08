# Getting Started

## Install

### Rust

```toml
[dependencies]
trackforge = "0.3"
```

### Python

```bash
pip install trackforge
```

## Your first tracker

The detection format is the same everywhere: a list of `([x, y, w, h], score, class_id)` tuples,
where `x, y` is the top-left corner in pixels.

### Rust

```rust
use trackforge::trackers::byte_track::ByteTrack;

let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

let detections = vec![
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
];

let tracks = tracker.update(detections);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

### Python

```python
from trackforge import BYTETRACK

tracker = BYTETRACK(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
]

for track_id, tlwh, score, class_id in tracker.update(detections):
    print(f"ID: {track_id}, Box: {tlwh}")
```

Call `update` once per frame. Each tracker keeps its own state and returns the confirmed tracks
for the current frame.
