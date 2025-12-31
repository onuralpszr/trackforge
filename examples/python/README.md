# Python Examples

This directory contains Python examples demonstrating trackforge's tracking capabilities with various object detection models.

## Prerequisites

```bash
# Install trackforge
pip install trackforge

# For YOLO examples
pip install ultralytics opencv-python

# For RT-DETR examples
pip install transformers torch pillow opencv-python
```

## Examples

| Example | Description | Tracker | Detector | Input | Output |
|---------|-------------|---------|----------|-------|--------|
| [`byte_track_demo.py`](byte_track_demo.py) | ByteTrack multi-object tracking | `ByteTrack` | YOLO11n | `test_video.mp4` | `output_tracking.mp4` |
| [`sort_yolo_demo.py`](sort_yolo_demo.py) | SORT tracking with YOLO | `Sort` | YOLO11n | `people.mp4` | `output_sort_yolo.mp4` |
| [`sort_rtdetr_demo.py`](sort_rtdetr_demo.py) | SORT tracking with RT-DETR | `Sort` | RT-DETR (Transformers) | `people.mp4` | `output_sort_rtdetr.mp4` |
| [`tracker_comparison.py`](tracker_comparison.py) | Side-by-side ByteTrack vs SORT | Both | YOLO11n | `people.mp4` | `output_comparison.mp4` |

## Quick Start

### ByteTrack with YOLO

```python
import trackforge
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = trackforge.ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6)

# Process detections
results = model.predict(frame, verbose=False)
detections = [(box.tlwh, box.conf, box.cls) for box in results[0].boxes]
tracks = tracker.update(detections)
```

### SORT with YOLO

```python
import trackforge
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = trackforge.Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Process detections
results = model.predict(frame, verbose=False)
detections = [(box.tlwh, box.conf, box.cls) for box in results[0].boxes]
tracks = tracker.update(detections)
```

## API Reference

### ByteTrack

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_thresh` | float | 0.5 | High confidence detection threshold |
| `track_buffer` | int | 30 | Frames to keep lost tracks alive |
| `match_thresh` | float | 0.8 | IoU threshold for matching |
| `det_thresh` | float | 0.6 | Threshold for new track initialization |

### Sort

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age` | int | 1 | Max frames without detection before deletion |
| `min_hits` | int | 3 | Min consecutive hits to confirm track |
| `iou_threshold` | float | 0.3 | IoU threshold for matching |

## Output Format

Both trackers return a list of tracks, where each track is a tuple:

```python
(track_id, [x, y, w, h], score, class_id)
```

- `track_id`: Unique integer identifier for the track
- `[x, y, w, h]`: Bounding box in TLWH format (top-left x, y, width, height)
- `score`: Detection confidence score
- `class_id`: Object class ID from the detector
