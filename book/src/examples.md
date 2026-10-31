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
| TrackTrack   | `tracktrack_demo.py`                       | `tracktrack_demo.rs`                    |

```bash
# Python
python examples/python/byte_track_demo.py

# Rust
cargo run --example byte_track_demo
cargo run --example deepsort_simple --features reid-model
cargo run --example deepsort_ort --features advanced_examples
cargo run --example det_ind_demo
cargo run --example tracktrack_demo
```

The Python demos use the usual detector stacks (`ultralytics`, `transformers` + `torch`,
`torch` + `torchvision`); install what a given demo imports. The `deepsort_simple` Rust demo
needs the `reid-model` feature and the `deepsort_ort` demo needs the `advanced_examples`
feature (ONNX Runtime + OpenCV).

## Ultralytics

Python: `trackforge.track_ultralytics(model, source, tracker)` runs an Ultralytics model over a video,
webcam, stream or image folder and yields `(result, tracks)` per frame, and
`trackforge.from_ultralytics(result)` converts a single result for your own loop. See
[`ultralytics_track.py`](https://github.com/onuralpszr/trackforge/blob/main/examples/python/ultralytics_track.py).

```python
for result, tracks in trackforge.track_ultralytics(YOLO("yolo11n.pt"), "people.mp4", trackforge.BYTETRACK(), classes=[0]):
    print(len(tracks))
```

Rust: `trackforge::utils::detections_from_xyxy` turns the `[x1, y1, x2, y2, conf, cls]` box data of
[`ultralytics-inference`](https://crates.io/crates/ultralytics-inference) boxes into tracker input. The
[`examples/ultralytics`](https://github.com/onuralpszr/trackforge/tree/main/examples/ultralytics)
crate tracks a video or image folder with any motion tracker. It is a separate crate because
`ultralytics-inference` is AGPL-3.0.

```bash
cd examples/ultralytics
cargo run --release -- "frames/*.jpg" --tracker bytetrack --classes 0
```
