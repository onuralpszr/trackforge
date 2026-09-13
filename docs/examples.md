# Examples

Runnable demos live under [`examples/`](https://github.com/onuralpszr/trackforge/tree/main/examples)
in the repository, with a Python and a Rust entry per tracker.

| Tracker      | Python                                      | Rust                                    |
| ------------ | -------------------------------------------- | ---------------------------------------- |
| ByteTrack    | `byte_track_demo.py`                        | `byte_track_demo.rs`                    |
| DeepSORT     | `deepsort_demo.py`                          | `deepsort_simple.rs`, `deepsort_ort.rs` |
| OC-SORT      | `ocsort_demo.py`                            | -                                       |
| Deep OC-SORT | `deep_ocsort_demo.py`                       | -                                       |
| BoT-SORT     | `botsort_demo.py`                           | `det_ind_demo.rs`                       |
| SORT         | `sort_yolo_demo.py`, `sort_rtdetr_demo.py`  | -                                       |
| TrackTrack   | -                                            | -                                       |

```bash
# Python
python examples/python/byte_track_demo.py

# Rust
cargo run --example byte_track_demo
cargo run --example deepsort_simple --features reid-model
cargo run --example deepsort_ort --features advanced_examples
cargo run --example det_ind_demo
```

The Python demos use the usual detector stacks (`ultralytics`, `transformers` + `torch`,
`torch` + `torchvision`); install what a given demo imports. The `deepsort_simple` Rust demo
needs the `reid-model` feature and the `deepsort_ort` demo needs the `advanced_examples`
feature (ONNX Runtime + OpenCV).

## Quick Start

### Rust - ByteTrack

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

### Python - ByteTrack

```python
from trackforge import BYTETRACK

tracker = BYTETRACK(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

### Python - DeepSORT

```python
from trackforge import DEEPSORT

tracker = DEEPSORT(max_age=70, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # appearance feature vectors

tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID: {track_id}, Box: {tlwh}, Score: {score}")
```

### Python - SORT

```python
from trackforge import SORT

tracker = SORT(max_age=1, min_hits=3, iou_threshold=0.3)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
tracks = tracker.update(detections)
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

See the [README](https://github.com/onuralpszr/trackforge#quick-start) for the full set of
Python and Rust snippets, one per tracker.
