<p align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-dark-transparent.png" media="(prefers-color-scheme: dark)" />
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" media="(prefers-color-scheme: light)" />
        <img src="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-light-transparent.png" alt="Trackforge logo" width="auto" />
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
exposed to Python via PyO3. It implements seven production-ready tracking algorithms on top of a
shared Kalman filter, so you can swap trackers without changing your integration code.

## Features

- **High Performance** — Native Rust implementation; ByteTrack runs in under 1 ms/frame on typical hardware.
- **Python Bindings** — Install from PyPI, import, and track in three lines.
- **Seven Algorithms** — SORT, ByteTrack, OC-SORT, DeepSORT, Deep OC-SORT, BoT-SORT, and TrackTrack cover the full speed-accuracy spectrum.
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

| Tracker          | Appearance       | Matching                         | When to use                                                 |
| ---------------- | ---------------- | -------------------------------- | ----------------------------------------------------------- |
| **SORT**         | None             | IoU                              | Simple scenes, highest speed, no occlusions                 |
| **ByteTrack**    | None             | IoU (2-stage)                    | Crowded scenes, low-confidence detections, short occlusions |
| **OC-SORT**      | None             | IoU + velocity (OCM)             | Scenes with frequent brief occlusions, no Re-ID available   |
| **DeepSORT**     | Re-ID embeddings | Appearance + IoU                 | Long occlusions, dense crowds, identity-sensitive use cases |
| **Deep OC-SORT** | Re-ID embeddings | IoU + velocity + appearance      | Occlusions where OC-SORT motion plus Re-ID helps            |
| **BoT-SORT**     | Re-ID embeddings | IoU + appearance + camera motion | Moving cameras, panning and zoom, with optional Re-ID       |
| **TrackTrack**   | Re-ID embeddings | Track-perspective association    | Crowded scenes needing strong identity, with optional Re-ID |

All trackers share the same detection input format:

```text
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

#### Tuning tips

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
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}")
```

**Result:** each element of `tracks` is a tuple
`(track_id, tlwh, score, class_id, det_ind)`. `det_ind` is the index of the
detection the track was last created from or matched to in the current frame's
detection list (or `None` when the track was not matched that frame); use it to
map a track back to its detection and reuse the detection's Re-ID embedding.

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

| Parameter             | Type    | Default | Description                                                         |
| --------------------- | ------- | ------- | ------------------------------------------------------------------- |
| `track_thresh`        | `f32`   | `0.5`   | Confidence threshold separating high- and low-confidence detections |
| `track_buffer`        | `usize` | `30`    | Frames a lost track is buffered before deletion                     |
| `match_thresh`        | `f32`   | `0.8`   | Stage-1 match cutoff as a maximum IoU distance (lower is stricter)  |
| `det_thresh`          | `f32`   | `0.6`   | Minimum confidence to initialise a new track                        |
| `second_match_thresh` | `f32`   | `0.5`   | Stage-2 match cutoff for recovering low-confidence detections       |

#### Tuning tips

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
for track_id, tlwh, score, class_id, det_ind in tracks:
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

#### Tuning tips

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
for track_id, tlwh, score, class_id, det_ind in tracks:
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

#### Tuning tips

- Lower `max_cosine_distance` (e.g. `0.15`) for stricter Re-ID — reduces ID switches at the cost
  of more unmatched detections.
- Increase `nn_budget` if your objects undergo gradual appearance changes over many frames.
- Lower `n_init` to `1` if detections are reliable and you need tracks immediately.
- `max_age=70` at 30 fps means tracks survive ~2.3 s of occlusion; increase for longer scenes.

### Implementing an AppearanceExtractor (Rust)

DeepSORT requires you to supply a feature extractor. Implement the `AppearanceExtractor` trait, which lives behind the `reid-model` feature (`features = ["reid-model"]`); on the default build, produce embeddings yourself and drive `DeepSortTracker` directly:

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
for track_id, tlwh, score, class_id, det_ind in tracks:
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

## Deep OC-SORT

**Deep OC-SORT** ([arXiv 2302.11813](https://arxiv.org/abs/2302.11813), ICIP 2023).
Extends OC-SORT with appearance: a cosine distance to each track's feature gallery is blended into
the motion cost, scaled by detector confidence. With `appearance_weight = 0` it reduces to plain
OC-SORT, so appearance is a strict add-on. Also accepts a caller-supplied camera-motion affine,
applied before association.

### Configuration

| Parameter             | Type    | Default | Description                                                   |
| ---------------------- | ------- | ------- | --------------------------------------------------------------- |
| `max_age`             | `usize` | `30`    | Frames a lost track survives before deletion                    |
| `min_hits`            | `usize` | `3`     | Consecutive matched frames required to confirm a track          |
| `iou_threshold`       | `f32`   | `0.3`   | Minimum IoU to associate a detection with a track                |
| `delta_t`             | `usize` | `3`     | Observation window (frames) used to compute velocity (OCV)       |
| `inertia`             | `f32`   | `0.2`   | Weight of the direction-consistency cost bonus (OCM)              |
| `appearance_weight`   | `f32`   | `0.5`   | Blend weight for the appearance cost, scaled by detection score |
| `max_cosine_distance` | `f32`   | `0.2`   | Cosine distance gate above which appearance is ignored           |
| `nn_budget`           | `usize` | `100`   | Maximum appearance features stored per track                     |

#### Tuning tips

- Raise `appearance_weight` when the Re-ID model is reliable and identities matter; lower it
  toward 0 to fall back to plain OC-SORT motion.
- Tighten `max_cosine_distance` to only trust strong appearance matches.
- The motion parameters (`max_age`, `min_hits`, `iou_threshold`, `delta_t`, `inertia`) behave as
  in OC-SORT.

### Python

```python
import trackforge

tracker = trackforge.DEEPOCSORT(
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

tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID={track_id}  box={tlwh}")
```

### Rust

```rust,ignore
use trackforge::trackers::deep_ocsort::DeepOcSort;

// `extractor` implements AppearanceExtractor (plug in any Re-ID model).
let mut tracker = DeepOcSort::new(extractor, 30, 3, 0.3, 3, 0.2, 0.5, 0.2, 100);

let tracks = tracker.update(&frame, detections).unwrap();
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

---

## BoT-SORT

**BoT-SORT** ([arXiv 2206.14651](https://arxiv.org/abs/2206.14651)).
Extends ByteTrack's two-stage cascade with camera motion compensation, which warps each track's
Kalman prediction by a caller-supplied affine transform before association, and an optional
appearance term fused with IoU in the high-confidence stage. With no embeddings it reduces to
ByteTrack with camera motion, so appearance is a strict add-on.

### Configuration

| Parameter            | Type    | Default | Description                                                    |
| ---------------------- | ------- | ------- | ----------------------------------------------------------------- |
| `track_thresh`        | `f32`   | `0.5`   | Confidence split between high- and low-score detections            |
| `track_buffer`        | `usize` | `30`    | Frames a lost track is kept alive before removal                   |
| `match_thresh`        | `f32`   | `0.8`   | Maximum cost for a first-stage (high-confidence) match              |
| `det_thresh`          | `f32`   | `0.6`   | Minimum score to start a new track                                  |
| `second_match_thresh` | `f32`   | `0.5`   | Stage-2 match cutoff for recovering low-confidence detections       |
| `proximity_thresh`    | `f32`   | `0.5`   | IoU-distance gate above which appearance is ignored                 |
| `appearance_thresh`   | `f32`   | `0.25`  | Cosine-distance gate above which appearance is ignored              |

#### Tuning tips

- Supply a camera-motion affine on moving-camera footage; leave it out for a static camera.
- Provide embeddings when a Re-ID model is available, and tighten `appearance_thresh` to only
  trust strong appearance matches.
- The two-stage thresholds behave as in ByteTrack.

### Python

```python
import trackforge

tracker = trackforge.BOTSORT(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    det_thresh=0.6,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection

# Moving camera: pass a [a, b, tx, c, d, ty] affine mapping the previous frame to the current one.
tracks = tracker.update(detections, embeddings, [1.0, 0.0, 12.0, 0.0, 1.0, -4.0])
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID={track_id}  box={tlwh}")
```

### Rust

```rust
use trackforge::trackers::botsort::BotSort;

let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

let detections = vec![([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64)];
let embeddings = vec![vec![0.1_f32, 0.2, 0.3]];
let tracks = tracker.update(detections, &embeddings);
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

---

## TrackTrack

**TrackTrack** ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)).
A track-centric tracker built on a ByteTrack-style two-stage lifecycle. Each track picks its own
best detection and a pair matches only when the choice is mutual, with a cost gate that tightens
each round; a leftover detection starts a new track only if it clears an init threshold and does
not overlap an existing track by more than `tai_thresh`. Appearance is optional: pass embeddings
for the Re-ID term, or an empty list to track on motion only.

### Configuration

| Parameter      | Type    | Default | Description                                                        |
| -------------- | ------- | ------- | ---------------------------------------------------------------------- |
| `det_thresh`   | `f32`   | `0.6`   | Score above which a detection is high confidence                       |
| `match_thresh` | `f32`   | `0.7`   | Association cost gate, lower is stricter                               |
| `track_buffer` | `usize` | `30`    | Frames a lost track is kept alive                                       |
| `min_hits`     | `usize` | `3`     | Matched frames in a row before a new track is confirmed                 |
| `init_thresh`  | `f32`   | `0.7`   | Smallest score a leftover detection needs to start a new track         |
| `tai_thresh`   | `f32`   | `0.55`  | Overlap gate for track-aware initialization, a maximum IoU              |
| `penalty_low`  | `f32`   | `0.2`   | Extra cost added to low confidence detections during association       |
| `reduce_step`  | `f32`   | `0.05`  | How much the cost gate tightens per matching round                     |

### Python

```python
import trackforge

tracker = trackforge.TRACKTRACK(det_thresh=0.6, match_thresh=0.7, track_buffer=30, min_hits=3)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
tracks = tracker.update(detections)
for track_id, tlwh, score, class_id, det_ind in tracks:
    print(f"ID={track_id}  box={tlwh}")
```

### Rust

```rust
use trackforge::trackers::tracktrack::TrackTrack;

let mut tracker = TrackTrack::new();

let detections = vec![([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64)];
let tracks = tracker.update(detections, &[]);
for t in &tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

---

## Detection Format

All trackers use the same detection tuple format:

```text
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
