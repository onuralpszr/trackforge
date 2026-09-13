# Examples

Runnable demos live under [`examples/`](https://github.com/onuralpszr/trackforge/tree/main/examples)
in the repository, with a Python and a Rust entry per tracker.

| Tracker      | Python                                     | Rust                                    |
| ------------ | ------------------------------------------ | ---------------------------------------- |
| ByteTrack    | `byte_track_demo.py`                       | `byte_track_demo.rs`                    |
| DeepSORT     | `deepsort_demo.py`                         | `deepsort_simple.rs`, `deepsort_ort.rs` |
| OC-SORT      | `ocsort_demo.py`                           | -                                       |
| Deep OC-SORT | `deep_ocsort_demo.py`                      | -                                       |
| BoT-SORT     | `botsort_demo.py`                          | `det_ind_demo.rs`                       |
| SORT         | `sort_yolo_demo.py`, `sort_rtdetr_demo.py` | -                                       |
| TrackTrack   | -                                           | -                                       |

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
