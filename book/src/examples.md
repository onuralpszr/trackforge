# Examples

Runnable demos live under [`examples/`](https://github.com/onuralpszr/trackforge/tree/main/examples)
in the repository, with a Python and a Rust entry per tracker.

| Tracker   | Python                    | Rust                                  |
| --------- | ------------------------- | ------------------------------------- |
| ByteTrack | `byte_track_demo.py`      | `byte_track_demo.rs`                  |
| DeepSORT  | `deepsort_demo.py`        | `deepsort_simple.rs`, `deepsort_ort.rs` |
| OC-SORT   | `ocsort_demo.py`          | -                                     |
| SORT      | `sort_yolo_demo.py`, `sort_rtdetr_demo.py` | -                    |
| All four  | `tracker_comparison.py`   | -                                     |

```bash
# Python
python examples/python/byte_track_demo.py

# Rust
cargo run --example byte_track_demo
cargo run --example deepsort_simple
cargo run --example deepsort_ort --features advanced_examples
```

The Python demos use the usual detector stacks (`ultralytics`, `transformers` + `torch`,
`torch` + `torchvision`); install what a given demo imports. The `deepsort_ort` Rust demo needs the
`advanced_examples` feature (ONNX Runtime + OpenCV).
