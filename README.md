<p align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-dark-transparent.png" media="(prefers-color-scheme: dark)" />
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" media="(prefers-color-scheme: light)" />
        <img src="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" alt="Trackforge logo" width="auto" />
    </picture>
</p>



[![Crates.io](https://img.shields.io/crates/v/trackforge.svg)](https://crates.io/crates/trackforge)
[![PyPI](https://img.shields.io/pypi/v/trackforge.svg)](https://pypi.org/project/trackforge/)
[![docs.rs](https://img.shields.io/docsrs/trackforge)](https://docs.rs/trackforge)
[![codecov](https://codecov.io/gh/onuralpszr/trackforge/branch/main/graph/badge.svg?token=DHMFYRLJW1)](https://codecov.io/gh/onuralpszr/trackforge)
[![CI](https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml/badge.svg)](https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml)
[![Dependabot Updates](https://github.com/onuralpszr/trackforge/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/onuralpszr/trackforge/actions/workflows/dependabot/dependabot-updates)
[![Security audit](https://github.com/onuralpszr/trackforge/actions/workflows/security-audit.yml/badge.svg)](https://github.com/onuralpszr/trackforge/actions/workflows/security-audit.yml)
![Crates.io MSRV](https://img.shields.io/crates/msrv/trackforge)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/onuralpszr/trackforge.svg?style=flat&logo=github)](https://github.com/onuralpszr/trackforge/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/onuralpszr/trackforge.svg?style=flat&logo=github)](https://github.com/onuralpszr/trackforge/network)
[![Crates.io Downloads](https://img.shields.io/crates/d/trackforge)](https://crates.io/crates/trackforge)



> [!IMPORTANT]
> **This project is currently under active development.** APIs and features are subject to change.

**Trackforge** is a unified, high-performance computer vision tracking library, implemented in Rust and exposed as a Python package. It provides state-of-the-art tracking algorithms like **ByteTrack**, optimized for speed and ease of use in both Rust and Python environments.

## Features

- 🚀 **High Performance**: Native Rust implementation for maximum speed and memory safety.
- 🐍 **Python Bindings**: Seamless integration with the Python ecosystem using `pyo3`.
- 🛠 **Unified API**: Consistent interface for tracking tasks across both languages.
- 📸 **ByteTrack**: Robust multi-object tracking using Kalman filters and IoU matching.

## Roadmap

## TODO — Multi-Object Tracking (MOT)

### Core Trackers
- [x] SORT — Simple Online and Realtime Tracking
- [ ] Norfair — Lightweight distance-based tracking

### Appearance-Based (Re-ID)
- [x] DeepSORT — SORT + appearance embeddings
- [ ] StrongSORT — Improved DeepSORT with stronger Re-ID
- [ ] StrongSORT++ — StrongSORT with camera motion compensation

### Detection-Driven Trackers
- [ ] ByteTrack — High/low confidence detection association
- [ ] BoT-SORT — ByteTrack + Re-ID + camera motion compensation

### Joint Detection & Tracking
- [ ] FairMOT — Unified detection and Re-ID network
- [ ] CenterTrack — Motion-aware detection-based tracking

### Transformer-Based Trackers
- [ ] OC-SORT — Observation-centric SORT
- [ ] TrackFormer — Transformer-based MOT
- [ ] MOTR — End-to-end transformer tracking

## GPU Support & Architecture

Trackforge transforms detections into tracks. It is designed to be the high-speed CPU "glue" in your pipeline. 

- **Detectors (GPU)**: Your object detector (YOLOv8, Yolanas, etc.) runs on the GPU to produce bounding boxes.
- **Trackforge (CPU)**: Receives these boxes and associates them on the CPU. Algorithms like ByteTrack are extremely efficient (less than 1ms per frame) and do not typically strictly require GPU acceleration, avoiding complex device transfers for the association step.
- **Future**: We may explore GPU-based association for massive-scale batch processing if data is already on-device.

## Installation

### Python

```bash
pip install trackforge
```

### Rust

Add `trackforge` to your `Cargo.toml`:

```toml
[dependencies]
trackforge = "0.1.6" # Check crates.io for latest version
```

## Usage

### 🐍 Python

#### ByteTrack

```python
import trackforge

# (tlwh, score, class_id)
detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]

tracker = trackforge.ByteTrack(0.5, 30, 0.8, 0.6)
tracks = tracker.update(detections)

for t in tracks:
    print(f"ID: {t[0]}, Box: {t[1]}")
```

#### DeepSORT

DeepSORT requires appearance embeddings (re-id features) alongside detection boxes.

```python
import trackforge
import numpy as np

# detections: [(tlwh, score, class_id), ...]
detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]

# embeddings: List of feature vectors (float32 list) corresponding to detections
embeddings = [[0.1, 0.2, 0.3, ...]] # Example embedding vector

tracker = trackforge.DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100)
tracks = tracker.update(detections, embeddings)

for t in tracks:
		# output adds confidence: (track_id, tlwh, confidence, class_id)
    print(f"ID: {t.track_id}, Box: {t.tlwh}")
```

See `examples/python/deepsort_demo.py` for a full example using `ultralytics` YOLO and `torchvision` ResNet.

### 🦀 Rust

#### ByteTrack

```rust
use trackforge::trackers::byte_track::ByteTrack;

fn main() -> anyhow::Result<()> {
    // Initialize ByteTrack
    let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

    // Detections: Vec<([f32; 4], f32, i64)>
    let detections = vec![
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ];

    // Update
    let tracks = tracker.update(detections);

    for t in tracks {
        println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
    }
    Ok(())
}
```

#### DeepSORT

See `examples/deepsort_ort.rs` for a full example integrating with `ort` (ONNX Runtime) for Re-ID and `usls` for detection.

```rust
// Minimal setup
use trackforge::trackers::deepsort::DeepSort;
use trackforge::traits::AppearanceExtractor;

struct MyExtractor;
impl AppearanceExtractor for MyExtractor {
    // Implement extract ...
}

let extractor = MyExtractor;
let mut tracker = DeepSort::new(extractor, ...);
```

## Development

This project uses `maturin` to manage the Rust/Python interop.

### Prerequisites

- Rust & Cargo
- Python 3.8+
- `maturin`: `pip install maturin`

### Build

```bash
# Build Python bindings
maturin develop

# Run Rust tests
cargo test
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
