<p align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-dark.png" media="(prefers-color-scheme: dark)" />
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-transparent.png" media="(prefers-color-scheme: light)" />
        <img src="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-transparent.png" alt="Trackforge logo" width="auto" />
    </picture>
</p>

<p align="center">
    <a href="https://crates.io/crates/trackforge"><img src="https://img.shields.io/crates/v/trackforge?logo=rust&logoColor=white&label=crates.io" alt="Crates.io version" /></a>
    <a href="https://crates.io/crates/trackforge"><img src="https://img.shields.io/crates/d/trackforge?logo=rust&logoColor=white&label=downloads" alt="Crates.io downloads" /></a>
    <a href="https://docs.rs/trackforge"><img src="https://img.shields.io/docsrs/trackforge?logo=docsdotrs&logoColor=white" alt="docs.rs" /></a>
    <a href="https://pypi.org/project/trackforge/"><img src="https://img.shields.io/pypi/v/trackforge?logo=python&logoColor=white&label=PyPI" alt="PyPI version" /></a>
    <a href="https://pypi.org/project/trackforge/#downloads"><img src="https://img.shields.io/pypi/dm/trackforge?logo=python&logoColor=white&label=pip%20downloads" alt="PyPI downloads" /></a>
    <a href="https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/onuralpszr/trackforge/CI.yml?branch=main&logo=githubactions&logoColor=white&label=CI" alt="CI" /></a>
    <a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/crates/l/trackforge?logo=opensourceinitiative&logoColor=white" alt="License" /></a>
</p>

**Trackforge** is a unified, high-performance multi-object tracking library written in Rust and
exposed to Python via PyO3. It implements four production-ready tracking algorithms on top of a
shared Kalman filter, so you can swap trackers without changing your integration code.

## Features

- **High Performance** — Native Rust implementation; ByteTrack runs in under 1 ms/frame on typical hardware.
- **Python Bindings** — Install from PyPI, import, and track in three lines.
- **Four Algorithms** — SORT, ByteTrack, OC-SORT, and DeepSORT cover the full speed-accuracy spectrum.
- **Unified API** — All trackers accept `(tlwh, score, class_id)` detection tuples.

## Installation

### Python

```bash
pip install trackforge
```

### Rust

```toml
[dependencies]
trackforge = "0.3"
```

To enable the Python bindings feature when building from source:

```bash
maturin develop --features python
```

---

## Choosing a Tracker

| Tracker       | Appearance       | Matching             | When to use                                                 |
| ------------- | ---------------- | -------------------- | ----------------------------------------------------------- |
| **SORT**      | None             | IoU                  | Simple scenes, highest speed, no occlusions                 |
| **ByteTrack** | None             | IoU (2-stage)        | Crowded scenes, low-confidence detections, short occlusions |
| **OC-SORT**   | None             | IoU + velocity (OCM) | Scenes with frequent brief occlusions, no Re-ID available   |
| **DeepSORT**  | Re-ID embeddings | Appearance + IoU     | Long occlusions, dense crowds, identity-sensitive use cases |

All trackers share the same detection input format:

```
(tlwh: [f32; 4], score: f32, class_id: i64)
```

where `tlwh` is `[top-left-x, top-left-y, width, height]`.

---

## SORT

**Simple Online and Realtime Tracking** ([arXiv 1602.00763](https://arxiv.org/abs/1602.00763)).
Pairs a Kalman filter with greedy IoU matching. Designed for speed — ideal when objects rarely
overlap.

### Configuration

| Parameter       | Type    | Default | Description                                                     |
| --------------- | ------- | ------- | --------------------------------------------------------------- |
| `max_age`       | `usize` | `1`     | Frames to keep a track alive without a detection match          |
| `min_hits`      | `usize` | `3`     | Consecutive matched frames required before a track is confirmed |
| `iou_threshold` | `f32`   | `0.3`   | Minimum IoU required to associate a detection with a track      |

**Tuning tips**

- Increase `max_age` to bridge short occlusions (at the cost of more false tracks).
- Decrease `iou_threshold` when objects are densely packed (more permissive matching).
- Increase `min_hits` to reduce false track initialisation in noisy detectors.

### Python

```python
import trackforge

tracker = trackforge.SORT(
    max_age=1,
    min_hits=3,
    iou_threshold=0.3,
)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.85, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}")
```

### Rust

```rust
use trackforge::trackers::sort::Sort;

let mut tracker = Sort::new(1, 3, 0.3);

let detections = vec![
    ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
];

let tracks = tracker.update(detections);
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

---

## ByteTrack

**ByteTrack** ([arXiv 2110.06864](https://arxiv.org/abs/2110.06864)).
A two-stage IoU tracker that associates _every_ detection — not just high-confidence ones — to
recover objects that are temporarily occluded or partially off-screen. Provides a significant
recall improvement over SORT with minimal added cost.

### Configuration

| Parameter      | Type    | Default | Description                                                         |
| -------------- | ------- | ------- | ------------------------------------------------------------------- |
| `track_thresh` | `f32`   | `0.5`   | Confidence threshold separating high- and low-confidence detections |
| `track_buffer` | `usize` | `30`    | Frames a lost track is buffered before deletion                     |
| `match_thresh` | `f32`   | `0.8`   | IoU threshold for stage-1 (high-confidence) matching                |
| `det_thresh`   | `f32`   | `0.6`   | Minimum confidence to initialise a new track                        |

**Tuning tips**

- Lower `track_thresh` (e.g. `0.3`) to include more low-confidence detections in stage 2.
- Increase `track_buffer` (e.g. `60`) when your detector produces intermittent misses.
- Lower `match_thresh` (e.g. `0.7`) in scenes with fast-moving objects where IoU drops quickly.
- `det_thresh` should usually sit a little above `track_thresh` to avoid noise seeding new tracks.

### Python

```python
import trackforge

tracker = trackforge.BYTETRACK(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    det_thresh=0.6,
)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.85, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}")
```

### Rust

```rust
use trackforge::trackers::byte_track::ByteTrack;

let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

let detections = vec![
    ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
    ([200.0_f32, 200.0, 60.0, 120.0], 0.85_f32, 0_i64),
];

let tracks = tracker.update(detections);
for t in &tracks {
    println!("ID: {}, Box: {:?}, Score: {:.2}", t.track_id, t.tlwh, t.score);
}
```

---

## OC-SORT

**OC-SORT** ([arXiv 2203.14360](https://arxiv.org/abs/2203.14360), CVPR 2023).
Extends SORT with three observation-centric mechanisms that reduce tracker drift during occlusions:

- **OCV** — velocity is computed from consecutive detections, not from the Kalman filter state.
- **OCM** — before matching, a direction-consistency bonus is added to each IoU score: pairs
  where the track's stored velocity direction aligns with the vector from the last observation to
  the candidate detection receive a higher effective IoU, improving association after missed frames.
- **ORU** — when a lost track is re-matched, the Kalman filter is corrected by replaying
  linearly interpolated observations between the last seen position and the current detection.

No appearance features are required, making it a strong upgrade over SORT when occlusions are
common but Re-ID is unavailable.

### Configuration

| Parameter       | Type    | Default | Description                                                          |
| --------------- | ------- | ------- | -------------------------------------------------------------------- |
| `max_age`       | `usize` | `30`    | Frames to keep a lost track alive before deletion                    |
| `min_hits`      | `usize` | `3`     | Consecutive matched frames required to confirm a track               |
| `iou_threshold` | `f32`   | `0.3`   | Minimum IoU to associate a detection with a track                    |
| `delta_t`       | `usize` | `3`     | Observation window (frames) used to compute velocity for OCV         |
| `inertia`       | `f32`   | `0.2`   | Weight for the direction-consistency cost bonus during OCM (0.0-1.0) |

**Tuning tips**

- Increase `max_age` (e.g. `60`) when objects undergo long occlusions.
- Increase `delta_t` for smoother velocity at the cost of responsiveness to rapid direction changes.
- Increase `inertia` (up to `1.0`) when objects move at near-constant velocity; lower it for
  erratic or non-linear motion.
- `min_hits=1` gives immediate track output — useful when detections are already filtered upstream.

### Python

```python
import trackforge

tracker = trackforge.OCSORT(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    delta_t=3,
    inertia=0.2,
)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.85, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}")
```

### Rust

```rust
use trackforge::trackers::ocsort::OcSort;

let mut tracker = OcSort::new(30, 3, 0.3, 3, 0.2);

let detections = vec![
    ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
    ([200.0_f32, 200.0, 60.0, 120.0], 0.85_f32, 0_i64),
];

let tracks = tracker.update(detections);
for t in &tracks {
    println!("ID: {}, Box: {:?}, Score: {:.2}", t.track_id, t.tlwh, t.score);
}
```

---

## DeepSORT

**DeepSORT** ([arXiv 1703.07402](https://arxiv.org/abs/1703.07402)).
Extends SORT with a Re-ID appearance metric. Confirmed tracks are first matched by cosine
distance on appearance embeddings (with Mahalanobis gating), then any remaining tracks fall back
to IoU matching. Provides robust long-term identity maintenance.

### Configuration

| Parameter             | Type    | Default | Description                                                     |
| --------------------- | ------- | ------- | --------------------------------------------------------------- |
| `max_age`             | `usize` | `70`    | Frames a track survives without a match                         |
| `n_init`              | `usize` | `3`     | Consecutive detections required to confirm a track              |
| `max_iou_distance`    | `f32`   | `0.7`   | IoU distance threshold for the fallback IoU stage               |
| `max_cosine_distance` | `f32`   | `0.2`   | Cosine distance threshold for appearance matching               |
| `nn_budget`           | `usize` | `100`   | Maximum number of appearance embeddings stored per track (FIFO) |

**Tuning tips**

- Lower `max_cosine_distance` (e.g. `0.15`) for stricter Re-ID — reduces ID switches at the cost
  of more unmatched detections.
- Increase `nn_budget` if your objects undergo gradual appearance changes over many frames.
- Lower `n_init` to `1` if detections are reliable and you need tracks immediately.
- `max_age=70` at 30 fps means tracks survive ~2.3 s of occlusion; increase for longer scenes.

### Implementing an AppearanceExtractor (Rust)

DeepSORT requires you to supply a feature extractor. Implement the `AppearanceExtractor` trait:

```rust,ignore
use trackforge::traits::AppearanceExtractor;
use trackforge::types::BoundingBox;
use image::DynamicImage;

struct MyExtractor;

impl AppearanceExtractor for MyExtractor {
    fn extract(
        &mut self,
        image: &DynamicImage,
        boxes: &[BoundingBox],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        // Crop each box from the image, run through your Re-ID model,
        // and return one embedding vector per box.
        Ok(boxes.iter().map(|_| vec![0.0_f32; 128]).collect())
    }
}
```

### Python

The Python `DeepSort` class accepts embeddings directly, so you can bring your own Re-ID model:

```python
import numpy as np
import trackforge

tracker = trackforge.DEEPSORT(
    max_age=70,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=100,
)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.85, 0),
]

# One 128-D embedding per detection (from your Re-ID model)
embeddings = [
    np.random.rand(128).tolist(),
    np.random.rand(128).tolist(),
]

tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}")
```

### Rust

```rust,ignore
use trackforge::trackers::deepsort::DeepSort;
use trackforge::types::BoundingBox;
use image::DynamicImage;

let mut tracker = DeepSort::new(MyExtractor, 70, 3, 0.7, 0.2, 100);

let frame = DynamicImage::new_rgb8(640, 480);
let detections = vec![
    (BoundingBox { x: 100.0, y: 100.0, width: 50.0, height: 100.0 }, 0.9_f32, 0_i64),
];

let tracks = tracker.update(&frame, &detections).unwrap();
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.to_tlwh());
}
```

---

## Detection Format

All trackers use the same detection tuple format:

```
([x, y, w, h], score, class_id)
```

- `x`, `y` — top-left corner in pixels
- `w`, `h` — width and height in pixels
- `score` — detector confidence in `[0.0, 1.0]`
- `class_id` — integer class label from your detector

## Links

- [Python API Reference](reference/python.md)
- [Rust API Reference](/api/trackforge/index.html)
- [Examples](examples.md)
  </content>
  </invoke>
