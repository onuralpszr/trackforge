<p align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-dark-transparent.png" media="(prefers-color-scheme: dark)" />
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" media="(prefers-color-scheme: light)" />
        <img src="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" alt="Trackforge logo" width="auto" />
    </picture>
</p>

**Trackforge** is a unified, high-performance computer vision tracking library implemented in Rust with Python bindings. It provides real-time multi-object tracking algorithms, optimized for speed and designed as the CPU "glue" between GPU-based object detectors and your tracking pipeline.

<p align="center">
    <a href="https://crates.io/crates/trackforge"><img src="https://img.shields.io/crates/v/trackforge?logo=rust&logoColor=white&label=crates.io" alt="Crates.io version" /></a>
    <a href="https://crates.io/crates/trackforge"><img src="https://img.shields.io/crates/d/trackforge?logo=rust&logoColor=white&label=downloads" alt="Crates.io downloads" /></a>
    <a href="https://docs.rs/trackforge"><img src="https://img.shields.io/docsrs/trackforge?logo=docsdotrs&logoColor=white" alt="docs.rs" /></a>
    <a href="https://crates.io/crates/trackforge"><img src="https://img.shields.io/crates/msrv/trackforge?logo=rust&logoColor=white" alt="MSRV" /></a>
    <a href="https://pypi.org/project/trackforge/"><img src="https://img.shields.io/pypi/v/trackforge?logo=python&logoColor=white&label=PyPI" alt="PyPI version" /></a>
    <a href="https://pypi.org/project/trackforge/#downloads"><img src="https://img.shields.io/pypi/dm/trackforge?logo=python&logoColor=white&label=pip%20downloads" alt="PyPI downloads" /></a>
    <a href="https://pypi.org/project/trackforge/"><img src="https://img.shields.io/pypi/pyversions/trackforge?logo=python&logoColor=white&label=python" alt="Python versions" /></a>
    <a href="https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/onuralpszr/trackforge/CI.yml?branch=main&logo=githubactions&logoColor=white&label=CI" alt="CI" /></a>
    <a href="https://codecov.io/gh/onuralpszr/trackforge"><img src="https://img.shields.io/codecov/c/github/onuralpszr/trackforge?logo=codecov&logoColor=white&token=DHMFYRLJW1" alt="Coverage" /></a>
    <a href="https://deps.rs/repo/github/onuralpszr/trackforge"><img src="https://deps.rs/repo/github/onuralpszr/trackforge/status.svg" alt="dependency status" /></a>
    <a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/crates/l/trackforge?logo=opensourceinitiative&logoColor=white" alt="License" /></a>
    <a href="https://www.conventionalcommits.org/en/v1.0.0/"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow?logo=conventionalcommits&logoColor=white" alt="Conventional Commits" /></a>
    <a href="https://github.com/j178/prek"><img src="https://img.shields.io/badge/managed%20by-prek-FAB040?logo=precommit&logoColor=white" alt="prek" /></a>
    <a href="https://github.com/onuralpszr/trackforge/blob/main/CITATION.cff"><img src="https://img.shields.io/badge/Cite%20this-repository-blue?logo=googlescholar&logoColor=white" alt="Cite this repository" /></a>
</p>

## Supported Trackers

| Tracker                                          | Type                              | Appearance (Re-ID) |
| ------------------------------------------------ | --------------------------------- | ------------------ |
| [ByteTrack](https://arxiv.org/abs/2110.06864)    | IoU + confidence association      | No                 |
| [DeepSORT](https://arxiv.org/abs/1703.07402)     | IoU + cosine distance             | Yes (pluggable)    |
| [OC-SORT](https://arxiv.org/abs/2203.14360)      | IoU + velocity direction (OCM)    | No                 |
| [Deep OC-SORT](https://arxiv.org/abs/2302.11813) | IoU + velocity (OCM) + appearance | Yes (pluggable)    |
| [BoT-SORT](https://arxiv.org/abs/2206.14651)     | IoU + appearance + camera motion  | Yes (pluggable)    |
| [SORT](https://arxiv.org/abs/1602.00763)         | IoU + Kalman filter               | No                 |

## Features

- 🚀 **Native Rust Core** Blazingly fast tracking (< 1ms/frame for ByteTrack) with full memory safety
- 🐍 **Python Bindings** First-class `pip install trackforge` support via PyO3
- 🎯 **Multi-Algorithm** ByteTrack, OC-SORT, DeepSORT, Deep OC-SORT, BoT-SORT, and SORT with a unified API
- 🔌 **Pluggable Re-ID** DeepSORT's appearance extractor is a trait; plug in any feature model
- 📐 **Generic Kalman Filter** Configurable position/velocity weighting, gating distance computation

<!-- prettier-ignore -->
> [!IMPORTANT]
> **Under active development.** APIs and features are subject to change. MSRV: Rust 1.89.

## Installation

### Python

```bash
pip install trackforge
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
trackforge = "0.3.0"
```

To build the Python bindings from source (e.g., via `maturin develop`), enable the `python` feature:

```toml
[dependencies]
trackforge = { version = "0.3.0", features = ["python"] }
```

## Quick Start

### Python - ByteTrack

```python
from trackforge import BYTETRACK

tracker = BYTETRACK(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)

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
from trackforge import DEEPSORT

tracker = DEEPSORT(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=100,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3, ...]]  # appearance feature vectors

tracks = tracker.update(detections, embeddings)

for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}, Score: {score}")
```

### Python - OC-SORT

```python
from trackforge import OCSORT

tracker = OCSORT(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    delta_t=3,
    inertia=0.2,
)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
]

tracks = tracker.update(detections)

for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

### Python - Deep OC-SORT

```python
from trackforge import DEEPOCSORT

tracker = DEEPOCSORT(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    delta_t=3,
    inertia=0.2,
    appearance_weight=0.5,
    max_cosine_distance=0.2,
    nn_budget=100,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection

# Pass embeddings for appearance-aware tracking, or omit them for motion only.
tracks = tracker.update(detections, embeddings)

for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

### Python - BoT-SORT

```python
from trackforge import BOTSORT

tracker = BOTSORT(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    det_thresh=0.6,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection

# Pass embeddings for appearance-aware tracking, or omit them for motion only.
tracks = tracker.update(detections, embeddings)

# Moving camera: pass a [a, b, tx, c, d, ty] affine mapping the previous frame
# to the current one.
tracks = tracker.update(detections, embeddings, [1.0, 0.0, 12.0, 0.0, 1.0, -4.0])

for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

### Rust - ByteTrack

```rust
use trackforge::trackers::byte_track::ByteTrack;

let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

// Format: ([x, y, w, h], confidence, class_id)
let detections = vec![
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
];

let tracks = tracker.update(detections);

for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

### Rust - DeepSORT

```rust
use trackforge::trackers::deepsort::DeepSort;

// `extractor` implements the AppearanceExtractor trait (plug in any Re-ID model).
let mut tracker = DeepSort::new(extractor, 30, 3, 0.7, 0.2, 100);

let detections = vec![(BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0)];
let tracks = tracker.update(&image, detections)?;

for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.to_tlwh());
}
```

### Rust - OC-SORT

```rust
use trackforge::trackers::ocsort::OcSort;

let mut tracker = OcSort::new(30, 3, 0.3, 3, 0.2);

let detections = vec![
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
];

let tracks = tracker.update(detections);

for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

### Rust - Deep OC-SORT

```rust,ignore
use trackforge::trackers::deep_ocsort::DeepOcSort;

// `extractor` implements AppearanceExtractor (plug in any Re-ID model).
let mut tracker = DeepOcSort::new(extractor, 30, 3, 0.3, 3, 0.2, 0.5, 0.2, 100);

let tracks = tracker.update(&image, detections)?;

for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

### Rust - BoT-SORT

```rust
use trackforge::trackers::botsort::BotSort;

let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

let detections = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
let embeddings = vec![vec![0.1, 0.2, 0.3]]; // one appearance vector per detection
let tracks = tracker.update(detections, &embeddings);

for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

### Rust - SORT

```rust
use trackforge::trackers::sort::Sort;

let mut tracker = Sort::new(1, 3, 0.3);

let detections = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
let tracks = tracker.update(detections);

for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Examples

Runnable demos live under [`examples/`](examples/), with both a Python and a Rust entry per tracker.

| Tracker      | Python                                                                                                                                  | Rust                                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| ByteTrack    | [`byte_track_demo.py`](examples/python/byte_track_demo.py) (YOLO11)                                                                     | [`byte_track_demo.rs`](examples/rust/byte_track_demo.rs)                                                  |
| DeepSORT     | [`deepsort_demo.py`](examples/python/deepsort_demo.py) (YOLO + ResNet18)                                                                | [`deepsort_simple.rs`](examples/deepsort_simple.rs), [`deepsort_ort.rs`](examples/deepsort_ort.rs) (ONNX) |
| OC-SORT      | [`ocsort_demo.py`](examples/python/ocsort_demo.py)                                                                                      | —                                                                                                         |
| Deep OC-SORT | [`deep_ocsort_demo.py`](examples/python/deep_ocsort_demo.py) (YOLO + ResNet18)                                                          | —                                                                                                         |
| BoT-SORT     | [`botsort_demo.py`](examples/python/botsort_demo.py) (YOLO + ResNet18)                                                                  | —                                                                                                         |
| SORT         | [`sort_yolo_demo.py`](examples/python/sort_yolo_demo.py) (YOLO), [`sort_rtdetr_demo.py`](examples/python/sort_rtdetr_demo.py) (RT-DETR) | —                                                                                                         |
| Comparison   | [`tracker_comparison.py`](examples/python/tracker_comparison.py) (ByteTrack vs SORT side-by-side)                                       | —                                                                                                         |

```bash
# Python
python examples/python/byte_track_demo.py

# Rust
cargo run --example byte_track_demo
cargo run --example deepsort_simple
cargo run --example deepsort_ort --features advanced_examples
```

The Python demos use the usual detector stacks: `ultralytics` (YOLO), `transformers` + `torch`
(RT-DETR), and `torch` + `torchvision` (ResNet Re-ID); install what a given demo imports. The
`deepsort_ort` Rust demo needs the `advanced_examples` feature (ONNX Runtime + OpenCV).

## API Reference

<a href="https://onuralpszr.github.io/trackforge/reference/python.html"><img src="https://img.shields.io/badge/Python%20API-docs-3776AB?logo=python&logoColor=white" alt="Python API" /></a>
<a href="https://docs.rs/trackforge"><img src="https://img.shields.io/badge/Rust%20API-docs.rs-000000?logo=docsdotrs&logoColor=white" alt="Rust API" /></a>
<a href="https://onuralpszr.github.io/trackforge/book/"><img src="https://img.shields.io/badge/Guide-mdBook-1F7087?logo=mdbook&logoColor=white" alt="Guide" /></a>

## Parameters

Each tracker's parameters and defaults (identical across Python and Rust) are documented on the
[Parameters page](https://onuralpszr.github.io/trackforge/parameters.html).

## Development

### Prerequisites

- Rust 1.89+ (MSRV)
- Python 3.8+ and [`maturin`](https://github.com/pyo3/maturin) for the bindings
- [`prek`](https://github.com/j178/prek) for git hooks (optional but recommended)

### Setup

```bash
git clone https://github.com/onuralpszr/trackforge.git
cd trackforge

# Rust core
cargo build
cargo test

# Python bindings (build into the active virtualenv)
maturin develop
```

### Checks

These mirror CI, run them before opening a PR:

```bash
cargo fmt --all -- --check          # formatting
cargo clippy --all-targets -- -D warnings   # lint, warnings are errors
cargo test                          # unit, integration, and doc tests
cargo llvm-cov --summary-only       # coverage (cargo install cargo-llvm-cov)
prek run --all-files                # all pre-commit hooks at once
```

### Feature flags

- `python` builds the PyO3 bindings.
- `advanced_examples` enables the ONNX/OpenCV-backed examples (`deepsort_ort`), which need
  ONNX Runtime and OpenCV on the system.

```bash
cargo test --features python
cargo run --example deepsort_ort --features advanced_examples
```

### Run a Python example

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

Planned trackers and milestones live on the [Roadmap page](https://onuralpszr.github.io/trackforge/roadmap.html).

## Citation

If you use trackforge in your research or project, please cite it. GitHub's "Cite this
repository" button reads the [`CITATION.cff`](CITATION.cff) metadata, or use:

```bibtex
@software{trackforge,
  author  = {Sezer, Onuralp},
  title   = {trackforge: A unified, high-performance multi-object tracking library},
  url     = {https://github.com/onuralpszr/trackforge},
  license = {MIT}
}
```

trackforge provides clean-room implementations of published tracking algorithms. Please
also cite the paper for the tracker you use:

<details>
<summary>Per-tracker citations</summary>

```bibtex
@inproceedings{bewley2016sort,
  title={Simple Online and Realtime Tracking},
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2016}
}

@inproceedings{wojke2017deepsort,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2017}
}

@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}

@inproceedings{cao2023ocsort,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Pang, Jiangmiao and Weng, Xinshuo and Khirodkar, Rawal and Kitani, Kris},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

@inproceedings{maggiolino2023deepocsort,
  title={Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification},
  author={Maggiolino, Gerard and Ahmad, Adnan and Cao, Jinkun and Kitani, Kris},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2023}
}

@article{aharon2022botsort,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```

</details>

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
