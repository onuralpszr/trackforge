<p align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-dark-transparent.png" media="(prefers-color-scheme: dark)" />
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" media="(prefers-color-scheme: light)" />
        <img src="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" alt="Trackforge logo" width="auto" />
    </picture>
</p>

**Trackforge** is a unified, high-performance computer vision tracking library implemented in Rust with Python bindings. It provides real-time multi-object tracking algorithms, optimized for speed and designed as the CPU "glue" between GPU-based object detectors and your tracking pipeline.

<p align="center">
    <a href="https://pypi.org/project/trackforge/"><img src="https://img.shields.io/pypi/v/trackforge.svg" alt="PyPI version" /></a>
    <a href="https://crates.io/crates/trackforge"><img src="https://img.shields.io/crates/v/trackforge.svg" alt="Crates.io version" /></a>
    <a href="https://docs.rs/trackforge"><img src="https://img.shields.io/docsrs/trackforge" alt="docs.rs" /></a>
    <a href="https://pypi.org/project/trackforge/#downloads"><img src="https://img.shields.io/pypi/dm/trackforge" alt="PyPI downloads" /></a>
    <a href="https://codecov.io/gh/onuralpszr/trackforge"><img src="https://codecov.io/gh/onuralpszr/trackforge/branch/main/graph/badge.svg?token=DHMFYRLJW1" alt="Coverage" /></a>
    <a href="https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml"><img src="https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml/badge.svg" alt="CI" /></a>
    <a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
</p>

## Supported Trackers

| Tracker                                       | Type                         | Appearance (Re-ID) | Language      | Status         |
| --------------------------------------------- | ---------------------------- | ------------------ | ------------- | -------------- |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | IoU + confidence association | No                 | Python & Rust | ✅ Implemented |
| [DeepSORT](https://arxiv.org/abs/1703.07402)  | IoU + cosine distance        | Yes (pluggable)    | Python & Rust | ✅ Implemented |
| [SORT](https://arxiv.org/abs/1602.00763)      | IoU + Kalman filter          | No                 | Python & Rust | ✅ Implemented |

## Features

- 🚀 **Native Rust Core** Blazingly fast tracking (< 1ms/frame for ByteTrack) with full memory safety
- 🐍 **Python Bindings** First-class `pip install trackforge` support via PyO3
- 🎯 **Multi-Algorithm** ByteTrack, DeepSORT, and SORT with a unified API
- 🔌 **Pluggable Re-ID** DeepSORT's appearance extractor is a trait; plug in any feature model
- 📐 **Generic Kalman Filter** Configurable position/velocity weighting, gating distance computation

## Architecture

```text
┌──────────────────┐   bboxes    ┌──────────────────┐   tracks    ┌──────────────────┐
│  GPU Detectors   │ ──────────▶ │    Trackforge    │ ──────────▶ │      Tracks      │
│ YOLO / RT-DETR / │             │  (CPU, no GPU    │             │   ID + bbox +    │
│     custom       │             │   round-trip)    │             │      class       │
└──────────────────┘             └──────────────────┘             └──────────────────┘
```

Trackforge is intentionally CPU-bound. It receives bounding boxes from GPU detectors and handles
association on the CPU no costly device transfers needed. Algorithms like ByteTrack run in under
1ms per frame.

<!-- prettier-ignore -->
> [!IMPORTANT]
> **Under active development.** APIs and features are subject to change. MSRV: Rust 1.87.

## Installation

### Python

```bash
pip install trackforge
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
trackforge = "0.1.9"
```

To build the Python bindings from source (e.g. via `maturin develop`), enable the `python` feature:

```toml
[dependencies]
trackforge = { version = "0.1.9", features = ["python"] }
```

## Quick Start

### Python - ByteTrack

```python
from trackforge import ByteTrack

tracker = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)

# Format: ([x, y, w, h], confidence, class_id)
detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
]

tracks = tracker.update(detections)

for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

### Python - DeepSORT

```python
from trackforge import DeepSort

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=100,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3, ...]]  # appearance feature vectors

tracks = tracker.update(detections, embeddings)

for track in tracks:
    print(f"ID: {track.track_id}, Box: {track.tlwh}, Score: {track.score}")
```

### Rust ByteTrack

```rust
use trackforge::trackers::byte_track::ByteTrack;

fn main() -> anyhow::Result<()> {
    let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

    let detections = vec![
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
    ];

    let tracks = tracker.update(detections);

    for t in tracks {
        println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
    }

    Ok(())
}
```

## Examples

Complete working examples are included in the repository:

### Python Examples

| Example                                                     | Description                                | Requirements                          |
| ----------------------------------------------------------- | ------------------------------------------ | ------------------------------------- |
| [ByteTrack + YOLO](examples/python/byte_track_demo.py)      | ByteTrack with Ultralytics YOLO11 on video | `ultralytics`, `opencv-python`        |
| [DeepSORT + YOLO](examples/python/deepsort_demo.py)         | DeepSORT with YOLO + ResNet18 embeddings   | `ultralytics`, `torch`, `torchvision` |
| [SORT + RT-DETR](examples/python/sort_rtdetr_demo.py)       | SORT with Hugging Face RT-DETR             | `transformers`, `torch`               |
| [SORT + YOLO](examples/python/sort_yolo_demo.py)            | SORT with Ultralytics YOLO                 | `ultralytics`, `opencv-python`        |
| [Tracker Comparison](examples/python/tracker_comparison.py) | side-by-side tracker benchmark             | varies                                |

Run a Python example:

```bash
python examples/python/byte_track_demo.py
```

### Rust Examples

| Example                                            | Description                               | Feature Flag        |
| -------------------------------------------------- | ----------------------------------------- | ------------------- |
| [ByteTrack Demo](examples/rust/byte_track_demo.rs) | Basic ByteTrack with simulated detections | none                |
| [DeepSORT (simple)](examples/deepsort_simple.rs)   | DeepSORT with a mock appearance extractor | none                |
| [DeepSORT + ONNX](examples/deepsort_ort.rs)        | DeepSORT with RT-DETR + ONNX Re-ID        | `advanced_examples` |

Run a Rust example:

```bash
cargo run --example byte_track_demo
cargo run --example deepsort_simple
cargo run --example deepsort_ort --features advanced_examples
```

## API Reference

- [Python API](https://onuralpszr.github.io/trackforge/reference/python.html) Full PyO3 class reference
- [Rust API](https://docs.rs/trackforge) Generated rustdoc

## Parameters

### ByteTrack

| Parameter      | Type  | Default | Description                                     |
| -------------- | ----- | ------- | ----------------------------------------------- |
| `track_thresh` | float | 0.5     | High confidence detection threshold             |
| `track_buffer` | int   | 30      | Frames to keep lost tracks alive                |
| `match_thresh` | float | 0.8     | IoU threshold for matching                      |
| `det_thresh`   | float | 0.6     | Minimum detection confidence for initialization |

### DeepSORT

| Parameter             | Type  | Default | Description                                      |
| --------------------- | ----- | ------- | ------------------------------------------------ |
| `max_age`             | int   | 70      | Max frames to keep a track without detection     |
| `n_init`              | int   | 3       | Consecutive detections needed to confirm a track |
| `max_iou_distance`    | float | 0.7     | Max IoU distance for association                 |
| `max_cosine_distance` | float | 0.2     | Max cosine distance for Re-ID matching           |
| `nn_budget`           | int   | 100     | Max appearance feature library size per track    |

### SORT

| Parameter       | Type  | Default | Description                                  |
| --------------- | ----- | ------- | -------------------------------------------- |
| `max_age`       | int   | 1       | Max frames to keep a track without detection |
| `min_hits`      | int   | 3       | Minimum hits before a track is confirmed     |
| `iou_threshold` | float | 0.3     | IoU threshold for matching                   |

## Development

### Prerequisites

- Rust 1.87+ (MSRV)
- Python 3.8+
- [`maturin`](https://github.com/pyo3/maturin) for Python bindings

### Build

```bash
# Build Python bindings in development mode
maturin develop

# Run Rust tests
cargo test

# Format code
cargo fmt
```

### Run Python Examples

```bash
# After `maturin develop`:
python examples/python/deepsort_demo.py --video your_video.mp4
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- For major changes, open an issue first to discuss what you would like to change.
- PRs should pass CI: `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`.
- Use [Commitizen](https://commitizen-tools.github.io/commitizen/) for commit messages: `cz commit`.

## Roadmap

### Completed

- [x] SORT
- [x] ByteTrack
- [x] DeepSORT
- [x] Python bindings & PyPI package
- [x] Rust & Python examples

### Planned

- [ ] Norfair Lightweight distance-based tracking
- [ ] StrongSORT Improved DeepSORT with stronger Re-ID
- [ ] StrongSORT++ With camera motion compensation
- [ ] BoT-SORT ByteTrack + Re-ID + camera motion compensation
- [ ] Joint detection & tracking (FairMOT, CenterTrack)
- [ ] Transformer-based trackers (TrackFormer, MOTR)
- [ ] TrackTrack: Focusing on Tracks for Online Multi-Object Tracking

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
