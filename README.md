# Trackforge

[![Crates.io](https://img.shields.io/crates/v/trackforge.svg)](https://crates.io/crates/trackforge)
[![PyPI](https://img.shields.io/pypi/v/trackforge.svg)](https://pypi.org/project/trackforge/)
[![docs.rs](https://img.shields.io/docsrs/trackforge)](https://docs.rs/trackforge)
[![codecov](https://codecov.io/gh/onuralpszr/trackforge/branch/main/graph/badge.svg?token=DHMFYRLJW1)](https://codecov.io/gh/onuralpszr/trackforge)
[![CI](https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml/badge.svg)](https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml)
[![Dependabot Updates](https://github.com/onuralpszr/trackforge/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/onuralpszr/trackforge/actions/workflows/dependabot/dependabot-updates)
[![Security audit](https://github.com/onuralpszr/trackforge/actions/workflows/security-audit.yml/badge.svg)](https://github.com/onuralpszr/trackforge/actions/workflows/security-audit.yml)
[![MSRV](https://img.shields.io/badge/rustc-1.60+-ab6000.svg)](https://blog.rust-lang.org/2022/04/07/Rust-1.60.0.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


> [!IMPORTANT]
> **This project is currently under active development.** APIs and features are subject to change.

**Trackforge** is a unified, high-performance computer vision tracking library, implemented in Rust and exposed as a Python package. It provides state-of-the-art tracking algorithms like **ByteTrack**, optimized for speed and ease of use in both Rust and Python environments.

## Features

- 🚀 **High Performance**: Native Rust implementation for maximum speed and memory safety.
- 🐍 **Python Bindings**: Seamless integration with the Python ecosystem using `pyo3`.
- 🛠 **Unified API**: Consistent interface for tracking tasks across both languages.
- 📸 **ByteTrack**: Robust multi-object tracking using Kalman filters and IoU matching.

## Installation

### Python

```bash
pip install trackforge
```
*(Note: Released on PyPI soon. For now, build from source).*

### Rust

Add `trackforge` to your `Cargo.toml`:

```toml
[dependencies]
trackforge = "0.1.1" # Check crates.io for latest version
```

## Usage

### 🐍 Python

```python
import trackforge

# Initialize ByteTrack
# track_thresh: High confidence detection threshold (e.g., 0.5)
# track_buffer: Frames to keep lost tracks alive (e.g., 30)
# match_thresh: IoU matching threshold (e.g., 0.8)
# det_thresh: Initialization threshold (e.g., 0.6)
tracker = trackforge.ByteTrack(0.5, 30, 0.8, 0.6)

# Detections: List of (TLWH_Box, Score, ClassID)
detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0)
]

# Update tracker
tracks = tracker.update(detections)

for t in tracks:
    # t is (track_id, tlwh_box, score, class_id)
    print(f"ID: {t[0]}, Box: {t[1]}")
```

### 🦀 Rust

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
