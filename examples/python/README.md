# Python Examples

Runnable demos for every trackforge tracker. Each demo reads a video, runs a detector, feeds detections to a tracker, and writes an annotated video. They share the same command-line interface and the same drawing/plumbing helpers.

## Prerequisites

```bash
# Core
pip install trackforge

# YOLO demos (SORT, ByteTrack, OC-SORT, Deep SORT, Deep OC-SORT, comparison)
pip install ultralytics opencv-python

# RT-DETR demo
pip install transformers torch pillow

# Appearance demos (Deep SORT, Deep OC-SORT) need a Re-ID backbone
pip install torch torchvision pillow
```

## Examples

| Example                                          | Tracker              | Detector           | Output                   |
| ------------------------------------------------ | -------------------- | ------------------ | ------------------------ |
| [`sort_yolo_demo.py`](sort_yolo_demo.py)         | `SORT`               | YOLO11n            | `output_sort_yolo.mp4`   |
| [`sort_rtdetr_demo.py`](sort_rtdetr_demo.py)     | `SORT`               | RT-DETR            | `output_sort_rtdetr.mp4` |
| [`byte_track_demo.py`](byte_track_demo.py)       | `BYTETRACK`          | YOLO11n            | `output_bytetrack.mp4`   |
| [`ocsort_demo.py`](ocsort_demo.py)               | `OCSORT`             | YOLO11n            | `output_ocsort.mp4`      |
| [`deepsort_demo.py`](deepsort_demo.py)           | `DEEPSORT`           | YOLO11n + ResNet18 | `output_deepsort.mp4`    |
| [`deep_ocsort_demo.py`](deep_ocsort_demo.py)     | `DEEPOCSORT`         | YOLO11n + ResNet18 | `output_deep_ocsort.mp4` |
| [`tracker_comparison.py`](tracker_comparison.py) | `BYTETRACK` + `SORT` | YOLO11n            | `output_comparison.mp4`  |

Two shared modules back the demos (not runnable on their own):

- [`common.py`](common.py) - video load, MP4 writer, YOLO-to-`tlwh` conversion, color palette, box/label drawing, progress logging.
- [`reid.py`](reid.py) - ResNet18 appearance embedder shared by the Deep SORT and Deep OC-SORT demos.

## Running

Every demo takes the same options and defaults to `people.mp4`:

```bash
python sort_yolo_demo.py --video people.mp4 --output tracked.mp4 --model yolo11n.pt
```

`deep_ocsort_demo.py` also accepts `--no-reid` to track on motion alone (pure OC-SORT behavior). `sort_rtdetr_demo.py` takes a Hugging Face model id via `--model` (default `PekingU/rtdetr_r50vd`).

## Quick Start

```python
import trackforge
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = trackforge.SORT(max_age=30, min_hits=3, iou_threshold=0.3)

results = model.predict(frame, verbose=False, classes=[0])
detections = []
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                       float(box.conf[0]), int(box.cls[0])))

tracks = tracker.update(detections)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID {track_id}: {tlwh}")
```

Swap the tracker line for any of the others:

```python
trackforge.BYTETRACK(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)
trackforge.OCSORT(max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2)
trackforge.DEEPSORT(max_age=70, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100)
trackforge.DEEPOCSORT(max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2,
                      appearance_weight=0.5, max_cosine_distance=0.2, nn_budget=100)
```

`DEEPSORT` and `DEEPOCSORT` take a parallel list of appearance embeddings: `tracker.update(detections, embeddings)`. For `DEEPOCSORT` the embeddings are optional (omit them to track on motion only).

## API Reference

### BYTETRACK

| Parameter      | Type  | Default | Description                            |
| -------------- | ----- | ------- | -------------------------------------- |
| `track_thresh` | float | 0.5     | High confidence detection threshold    |
| `track_buffer` | int   | 30      | Frames to keep lost tracks alive       |
| `match_thresh` | float | 0.8     | IoU threshold for matching             |
| `det_thresh`   | float | 0.6     | Threshold for new track initialization |

### SORT

| Parameter       | Type  | Default | Description                                  |
| --------------- | ----- | ------- | -------------------------------------------- |
| `max_age`       | int   | 1       | Max frames without detection before deletion |
| `min_hits`      | int   | 3       | Min consecutive hits to confirm track        |
| `iou_threshold` | float | 0.3     | IoU threshold for matching                   |

### OCSORT

| Parameter       | Type  | Default | Description                                  |
| --------------- | ----- | ------- | -------------------------------------------- |
| `max_age`       | int   | 30      | Max frames without detection before deletion |
| `min_hits`      | int   | 3       | Min consecutive hits to confirm track        |
| `iou_threshold` | float | 0.3     | IoU threshold for matching                   |
| `delta_t`       | int   | 3       | Frame window for velocity direction (OCM)    |
| `inertia`       | float | 0.2     | Weight of the velocity-direction bonus       |

### DEEPSORT

| Parameter             | Type  | Default | Description                                  |
| --------------------- | ----- | ------- | -------------------------------------------- |
| `max_age`             | int   | 70      | Max frames without detection before deletion |
| `n_init`              | int   | 3       | Min consecutive hits to confirm track        |
| `max_iou_distance`    | float | 0.7     | Max IoU distance for cascade matching        |
| `max_cosine_distance` | float | 0.2     | Max cosine distance for appearance matching  |
| `nn_budget`           | int   | 100     | Max appearance features stored per track     |

### DEEPOCSORT

| Parameter             | Type  | Default | Description                                    |
| --------------------- | ----- | ------- | ---------------------------------------------- |
| `max_age`             | int   | 30      | Max frames without detection before deletion   |
| `min_hits`            | int   | 3       | Min consecutive hits to confirm track          |
| `iou_threshold`       | float | 0.3     | IoU threshold for matching                     |
| `delta_t`             | int   | 3       | Frame window for velocity direction (OCM)      |
| `inertia`             | float | 0.2     | Weight of the velocity-direction bonus         |
| `appearance_weight`   | float | 0.5     | Blend weight for the appearance cost (0 = off) |
| `max_cosine_distance` | float | 0.2     | Gate above which appearance is ignored         |
| `nn_budget`           | int   | 100     | Max appearance features stored per track       |

## Output Format

Every tracker returns a list of tuples:

```python
(track_id, [x, y, w, h], score, class_id)
```

- `track_id`: unique integer identifier for the track
- `[x, y, w, h]`: bounding box in TLWH format (top-left x, y, width, height)
- `score`: detection confidence
- `class_id`: object class id from the detector
