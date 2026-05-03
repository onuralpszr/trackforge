# Examples

## Rust Examples

### ByteTrack Demo

```rust
use trackforge::trackers::byte_track::ByteTrack;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6
    let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

    let frame_1_detections = vec![
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
    ];

    println!("Processing Frame 1...");
    let tracks_1 = tracker.update(frame_1_detections);
    for t in tracks_1 {
        println!("Track ID: {}, Box: {:?}, Score: {:.2}", t.track_id, t.tlwh, t.score);
    }

    let frame_2_detections = vec![
        ([105.0, 102.0, 50.0, 100.0], 0.92, 0),
        ([202.0, 201.0, 60.0, 120.0], 0.88, 0),
    ];

    println!("\nProcessing Frame 2...");
    let tracks_2 = tracker.update(frame_2_detections);
    for t in tracks_2 {
        println!("Track ID: {}, Box: {:?}, Score: {:.2}", t.track_id, t.tlwh, t.score);
    }

    Ok(())
}
```

## Python Examples

### ByteTrack with YOLO

```python
import cv2
from ultralytics import YOLO
import trackforge
import time


def run_tracking(video_path="test_video.mp4", output_path="output_tracking.mp4"):
    model = YOLO("yolo26n.pt")

    # track_thresh=0.1, track_buffer=30, match_thresh=0.8, det_thresh=0.1
    tracker = trackforge.ByteTrack(0.1, 30, 0.8, 0.1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.predict(frame, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append((
                    [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    float(box.conf[0]),
                    int(box.cls[0]),
                ))

        online_tracks = tracker.update(detections)

        for track_id, tlwh, score, class_id in online_tracks:
            x1, y1, w, h = tlwh
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id} {model.names[class_id]} {score:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        out.write(frame)
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    t1 = time.time()
    print(f"Done. {frame_count} frames in {t1 - t0:.2f}s ({frame_count / (t1 - t0):.1f} fps)")
    cap.release()
    out.release()


if __name__ == "__main__":
    run_tracking()
```

### DeepSORT with Random Embeddings

```python
import numpy as np
import trackforge


def main():
    tracker = trackforge.DeepSort(
        max_age=70,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
    )

    # Simulate two frames of detections
    for frame_idx in range(1, 3):
        detections = [
            ([100.0 + frame_idx, 100.0, 50.0, 100.0], 0.92, 0),
            ([200.0 + frame_idx, 150.0, 60.0, 120.0], 0.87, 0),
        ]
        # Replace with real Re-ID embeddings from your model
        embeddings = [np.random.rand(128).tolist() for _ in detections]

        tracks = tracker.update(detections, embeddings)
        print(f"Frame {frame_idx}: {len(tracks)} active track(s)")
        for t in tracks:
            print(f"  ID={t.track_id}  box={t.tlwh}  score={t.score:.2f}")


if __name__ == "__main__":
    main()
```

### SORT Minimal Example

```python
import trackforge


def main():
    tracker = trackforge.Sort(max_age=1, min_hits=3, iou_threshold=0.3)

    for frame_idx in range(1, 6):
        detections = [
            ([100.0 + frame_idx * 2, 100.0, 50.0, 100.0], 0.9, 0),
        ]
        tracks = tracker.update(detections)
        print(f"Frame {frame_idx}: {tracks}")


if __name__ == "__main__":
    main()
```

</content>
</invoke>
