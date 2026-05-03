# Python API Reference

Trackforge exposes three tracker classes.  All are importable directly from the `trackforge`
module after installing with `pip install trackforge`.

```python
import trackforge

# Available classes
trackforge.ByteTrack
trackforge.Sort
trackforge.DeepSort
```

---

## `ByteTrack`

Two-stage IoU tracker.  Associates high-confidence detections first, then attempts to recover
lost tracks using low-confidence detections.

### Constructor

```python
trackforge.ByteTrack(
    track_thresh: float = 0.5,
    track_buffer: int   = 30,
    match_thresh: float = 0.8,
    det_thresh: float   = 0.6,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_thresh` | `float` | `0.5` | Confidence threshold that splits high- and low-confidence detections |
| `track_buffer` | `int` | `30` | Frames a lost track is kept alive before deletion |
| `match_thresh` | `float` | `0.8` | IoU threshold for stage-1 (high-confidence) matching |
| `det_thresh` | `float` | `0.6` | Minimum confidence to initialise a new track |

### `update`

```python
tracks = tracker.update(detections: list[tuple[list[float], float, int]]) -> list[tuple]
```

**Parameters**

- `detections` — list of `([x, y, w, h], score, class_id)` tuples.

**Returns**

A list of `(track_id, tlwh, score, class_id)` tuples for every active confirmed track in the
current frame.

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | `int` | Unique, monotonically increasing track identifier |
| `tlwh` | `list[float]` | Bounding box `[top-left-x, top-left-y, width, height]` in pixels |
| `score` | `float` | Detection confidence of the most recent match |
| `class_id` | `int` | Class label of the most recent match |

### Example

```python
import trackforge

tracker = trackforge.ByteTrack(
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

## `Sort`

Simple Online and Realtime Tracking.  Lightweight IoU-only tracker, no appearance features.

### Constructor

```python
trackforge.Sort(
    max_age: int        = 1,
    min_hits: int       = 3,
    iou_threshold: float = 0.3,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age` | `int` | `1` | Frames a track survives without a detection match |
| `min_hits` | `int` | `3` | Consecutive matched frames before a track is confirmed |
| `iou_threshold` | `float` | `0.3` | Minimum IoU to associate a detection with a track |

### `update`

```python
tracks = tracker.update(detections: list[tuple[list[float], float, int]]) -> list[tuple]
```

Same input/output format as `ByteTrack.update`.

### Example

```python
import trackforge

tracker = trackforge.Sort(max_age=1, min_hits=3, iou_threshold=0.3)

detections = [
    ([100.0, 100.0, 50.0, 100.0], 0.92, 0),
]

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID={track_id}  box={tlwh}")
```

---

## `DeepSort`

DeepSORT with a Re-ID appearance metric.  Accepts explicit embedding vectors so you can plug in
any Re-ID model.

### Constructor

```python
trackforge.DeepSort(
    max_age: int                 = 70,
    n_init: int                  = 3,
    max_iou_distance: float      = 0.7,
    max_cosine_distance: float   = 0.2,
    nn_budget: int               = 100,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age` | `int` | `70` | Frames a track survives without a match |
| `n_init` | `int` | `3` | Consecutive detections required to confirm a track |
| `max_iou_distance` | `float` | `0.7` | IoU distance threshold for the fallback IoU stage |
| `max_cosine_distance` | `float` | `0.2` | Cosine distance threshold for appearance matching |
| `nn_budget` | `int` | `100` | Maximum embeddings stored per track (FIFO, `None` for unlimited) |

### `update`

```python
tracks = tracker.update(
    detections: list[tuple[list[float], float, int]],
    embeddings: list[list[float]],
) -> list[DeepSortTrack]
```

**Parameters**

- `detections` — list of `([x, y, w, h], score, class_id)` tuples.
- `embeddings` — list of appearance embedding vectors, one per detection.  Length must equal
  `len(detections)`.  Each embedding can be any length but must be consistent within a session.

**Returns**

A list of `DeepSortTrack` objects for confirmed tracks that were matched in the current frame.

### `DeepSortTrack`

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_id` | `int` | Unique track identifier |
| `tlwh` | `list[float]` | Bounding box `[top-left-x, top-left-y, width, height]` |
| `score` | `float` | Detection confidence of the last match |
| `class_id` | `int` | Class label of the last match |

### Example

```python
import numpy as np
import trackforge

tracker = trackforge.DeepSort(
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
for t in tracks:
    print(f"ID={t.track_id}  box={t.tlwh}  score={t.score:.2f}")
```

---

## Detection Format

All trackers accept the same detection tuple format:

```python
([x, y, w, h], score, class_id)
```

| Field | Type | Description |
|-------|------|-------------|
| `x` | `float` | Top-left x coordinate in pixels |
| `y` | `float` | Top-left y coordinate in pixels |
| `w` | `float` | Width in pixels |
| `h` | `float` | Height in pixels |
| `score` | `float` | Detector confidence in `[0.0, 1.0]` |
| `class_id` | `int` | Integer class label |
</content>
</invoke>