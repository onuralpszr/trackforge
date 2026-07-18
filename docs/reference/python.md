# Python API Reference

<p>
    <a href="https://pypi.org/project/trackforge/"><img src="https://img.shields.io/pypi/v/trackforge?logo=python&logoColor=white&label=PyPI" alt="PyPI version" /></a>
    <a href="https://pypi.org/project/trackforge/#downloads"><img src="https://img.shields.io/pypi/dm/trackforge?logo=python&logoColor=white&label=pip%20downloads" alt="PyPI downloads" /></a>
    <a href="https://pypi.org/project/trackforge/"><img src="https://img.shields.io/pypi/pyversions/trackforge?logo=python&logoColor=white" alt="Python versions" /></a>
</p>

Trackforge exposes six tracker classes. All are importable directly from the `trackforge`
module after installing with `pip install trackforge`.

```python
import trackforge

# Available classes
trackforge.BYTETRACK
trackforge.SORT
trackforge.OCSORT
trackforge.DEEPSORT
trackforge.DEEPOCSORT
trackforge.BOTSORT
```

---

## `BYTETRACK`

Two-stage IoU tracker. Associates high-confidence detections first, then attempts to recover
lost tracks using low-confidence detections.

### Constructor

```python
trackforge.BYTETRACK(
    track_thresh: float        = 0.5,
    track_buffer: int          = 30,
    match_thresh: float        = 0.8,
    det_thresh: float          = 0.6,
    second_match_thresh: float = 0.5,
)
```

| Parameter             | Type    | Default | Description                                                          |
| --------------------- | ------- | ------- | -------------------------------------------------------------------- |
| `track_thresh`        | `float` | `0.5`   | Confidence threshold that splits high- and low-confidence detections |
| `track_buffer`        | `int`   | `30`    | Frames a lost track is kept alive before deletion                    |
| `match_thresh`        | `float` | `0.8`   | Stage-1 match cutoff as a maximum IoU distance (lower is stricter)   |
| `det_thresh`          | `float` | `0.6`   | Minimum confidence to initialise a new track                         |
| `second_match_thresh` | `float` | `0.5`   | Stage-2 match cutoff for recovering low-confidence detections        |

### `update`

```python
tracks = tracker.update(detections: list[tuple[list[float], float, int]]) -> list[tuple]
```

#### Parameters

- `detections` — list of `([x, y, w, h], score, class_id)` tuples.

#### Returns

A list of `(track_id, tlwh, score, class_id)` tuples for every active confirmed track in the
current frame.

| Field      | Type          | Description                                                      |
| ---------- | ------------- | ---------------------------------------------------------------- |
| `track_id` | `int`         | Unique, monotonically increasing track identifier                |
| `tlwh`     | `list[float]` | Bounding box `[top-left-x, top-left-y, width, height]` in pixels |
| `score`    | `float`       | Detection confidence of the most recent match                    |
| `class_id` | `int`         | Class label of the most recent match                             |

### Example

```python
import trackforge

tracker = trackforge.BYTETRACK(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    det_thresh=0.6,
)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.92, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.87, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}  class={class_id}")
```

---

## `SORT`

Simple Online and Realtime Tracking. Lightweight IoU-only tracker, no appearance features.

### Constructor

```python
trackforge.SORT(
    max_age: int        = 1,
    min_hits: int       = 3,
    iou_threshold: float = 0.3,
)
```

| Parameter       | Type    | Default | Description                                            |
| --------------- | ------- | ------- | ------------------------------------------------------ |
| `max_age`       | `int`   | `1`     | Frames a track survives without a detection match      |
| `min_hits`      | `int`   | `3`     | Consecutive matched frames before a track is confirmed |
| `iou_threshold` | `float` | `0.3`   | Minimum IoU to associate a detection with a track      |

### `update`

```python
tracks = tracker.update(detections: list[tuple[list[float], float, int]]) -> list[tuple]
```

Same input/output format as `BYTETRACK.update`.

### Example

```python
import trackforge

tracker = trackforge.SORT(max_age=1, min_hits=3, iou_threshold=0.3)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.92, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}")
```

---

## `OCSORT`

Observation-Centric SORT. Extends SORT with velocity-based prediction correction (OCM) and
Kalman filter re-update on re-association (ORU). More robust than SORT in scenes with brief
occlusions without requiring appearance features.

### Constructor

```python
trackforge.OCSORT(
    max_age: int        = 30,
    min_hits: int       = 3,
    iou_threshold: float = 0.3,
    delta_t: int        = 3,
    inertia: float      = 0.2,
)
```

| Parameter       | Type    | Default | Description                                                                   |
| --------------- | ------- | ------- | ----------------------------------------------------------------------------- |
| `max_age`       | `int`   | `30`    | Frames a lost track is kept alive before deletion                             |
| `min_hits`      | `int`   | `3`     | Consecutive matched frames required to confirm a track                        |
| `iou_threshold` | `float` | `0.3`   | Minimum IoU to associate a detection with a track                             |
| `delta_t`       | `int`   | `3`     | Observation window (frames) for velocity computation (OCV)                    |
| `inertia`       | `float` | `0.2`   | Weight for the direction-consistency cost bonus during OCM (range 0.0 to 1.0) |

### `update`

```python
tracks = tracker.update(detections: list[tuple[list[float], float, int]]) -> list[tuple]
```

Same input/output format as `BYTETRACK.update` and `SORT.update`.

### Example

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
    ([100.0, 100.0, 50.0, 100.0], 0.92, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.87, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}  class={class_id}")
```

---

## `DEEPSORT`

DeepSORT with a Re-ID appearance metric. Accepts explicit embedding vectors so you can plug in
any Re-ID model.

### Constructor

```python
trackforge.DEEPSORT(
    max_age: int                 = 70,
    n_init: int                  = 3,
    max_iou_distance: float      = 0.7,
    max_cosine_distance: float   = 0.2,
    nn_budget: int               = 100,
)
```

| Parameter             | Type    | Default | Description                                                      |
| --------------------- | ------- | ------- | ---------------------------------------------------------------- |
| `max_age`             | `int`   | `70`    | Frames a track survives without a match                          |
| `n_init`              | `int`   | `3`     | Consecutive detections required to confirm a track               |
| `max_iou_distance`    | `float` | `0.7`   | IoU distance threshold for the fallback IoU stage                |
| `max_cosine_distance` | `float` | `0.2`   | Cosine distance threshold for appearance matching                |
| `nn_budget`           | `int`   | `100`   | Maximum embeddings stored per track (FIFO, `None` for unlimited) |

### `update`

```python
tracks = tracker.update(
    detections: list[tuple[list[float], float, int]],
    embeddings: list[list[float]],
) -> list[tuple[int, list[float], float, int]]
```

#### Parameters

- `detections` — list of `([x, y, w, h], score, class_id)` tuples.
- `embeddings` — list of appearance embedding vectors, one per detection. Length must equal
  `len(detections)`. Each embedding can be any length but must be consistent within a session.

#### Returns

A list of `(track_id, tlwh, score, class_id)` tuples for confirmed tracks matched in the current
frame. Same format as all other trackers.

### Example

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
    ([100.0, 100.0, 50.0, 100.0], 0.92, 0),
    ([200.0, 150.0, 60.0, 120.0], 0.87, 0),
]

# Produce embeddings from your Re-ID model — here we use random vectors
embeddings = [np.random.rand(128).tolist() for _ in detections]

tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}  score={score:.2f}")
```

---

## `DEEPOCSORT`

Deep OC-SORT. OC-SORT plus an appearance term, so it accepts embeddings and, optionally, a per-frame camera motion transform. Pass an empty embeddings list to track on motion only.

### Constructor

```python
trackforge.DEEPOCSORT(
    max_age: int               = 30,
    min_hits: int              = 3,
    iou_threshold: float       = 0.3,
    delta_t: int               = 3,
    inertia: float             = 0.2,
    appearance_weight: float   = 0.5,
    max_cosine_distance: float = 0.2,
    nn_budget: int             = 100,
)
```

| Parameter             | Type    | Default | Description                                                  |
| --------------------- | ------- | ------- | ------------------------------------------------------------ |
| `max_age`             | `int`   | `30`    | Frames a lost track survives before deletion                 |
| `min_hits`            | `int`   | `3`     | Consecutive matched frames required to confirm a track       |
| `iou_threshold`       | `float` | `0.3`   | Minimum IoU to associate a detection with a track            |
| `delta_t`             | `int`   | `3`     | Frames back used to estimate an object's direction of travel |
| `inertia`             | `float` | `0.2`   | How strongly that direction is trusted, range 0.0 to 1.0     |
| `appearance_weight`   | `float` | `0.5`   | How much appearance counts against motion, range 0.0 to 1.0  |
| `max_cosine_distance` | `float` | `0.2`   | Cosine distance gate for appearance matching                 |
| `nn_budget`           | `int`   | `100`   | Maximum embeddings stored per track                          |

### `update`

```python
tracks = tracker.update(
    detections: list[tuple[list[float], float, int]],
    embeddings: list[list[float]] = [],
    camera_motion: list[float] | None = None,
) -> list[tuple[int, list[float], float, int]]
```

#### Parameters

- `detections` — list of `([x, y, w, h], score, class_id)` tuples.
- `embeddings` — appearance vectors, one per detection, or an empty list for motion only.
- `camera_motion` — six affine coefficients `[a, b, tx, c, d, ty]` mapping the previous frame to the current one, or `None` for a static camera.

#### Returns

A list of `(track_id, tlwh, score, class_id)` tuples for confirmed tracks matched in the current frame.

---

## `BOTSORT`

BoT-SORT. ByteTrack plus camera motion compensation and an optional appearance term. Pass an empty embeddings list to track on motion only.

### Constructor

```python
trackforge.BOTSORT(
    track_thresh: float        = 0.5,
    track_buffer: int          = 30,
    match_thresh: float        = 0.8,
    det_thresh: float          = 0.6,
    second_match_thresh: float = 0.5,
    proximity_thresh: float    = 0.5,
    appearance_thresh: float   = 0.25,
)
```

| Parameter             | Type    | Default | Description                                                   |
| --------------------- | ------- | ------- | ------------------------------------------------------------- |
| `track_thresh`        | `float` | `0.5`   | Score above which a detection is high confidence              |
| `track_buffer`        | `int`   | `30`    | Frames a lost track is kept alive                             |
| `match_thresh`        | `float` | `0.8`   | Stage-1 match cutoff as a maximum IoU distance                |
| `det_thresh`          | `float` | `0.6`   | Minimum score to start a new track                            |
| `second_match_thresh` | `float` | `0.5`   | Stage-2 match cutoff for recovering low-confidence detections |
| `proximity_thresh`    | `float` | `0.5`   | How much boxes must overlap before appearance is used         |
| `appearance_thresh`   | `float` | `0.25`  | How close two embeddings must be for Re-ID to help the match  |

### `update`

```python
tracks = tracker.update(
    detections: list[tuple[list[float], float, int]],
    embeddings: list[list[float]] = [],
    camera_motion: list[float] | None = None,
) -> list[tuple[int, list[float], float, int]]
```

Same input and output format as `DEEPOCSORT.update`.

---

## Detection Format

All trackers accept the same detection tuple format:

```python
([x, y, w, h], score, class_id)
```

| Field      | Type    | Description                         |
| ---------- | ------- | ----------------------------------- |
| `x`        | `float` | Top-left x coordinate in pixels     |
| `y`        | `float` | Top-left y coordinate in pixels     |
| `w`        | `float` | Width in pixels                     |
| `h`        | `float` | Height in pixels                    |
| `score`    | `float` | Detector confidence in `[0.0, 1.0]` |
| `class_id` | `int`   | Integer class label                 |
