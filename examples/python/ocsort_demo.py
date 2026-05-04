import cv2
from ultralytics import YOLO
import trackforge
import time


def run_tracking(video_path="test_video.mp4", output_path="output_ocsort.mp4"):
    model = YOLO("yolo26n.pt")

    tracker = trackforge.OCSORT(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        inertia=0.2,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append((tlwh, conf, cls))

        online_tracks = tracker.update(detections)

        for track_id, tlwh, score, class_id in online_tracks:
            x1, y1, w, h = tlwh
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2
            )
            label = f"ID:{track_id} {model.names[class_id]} {score:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    t1 = time.time()
    print(
        f"Done. {frame_count} frames in {t1 - t0:.2f}s ({frame_count / (t1 - t0):.1f} fps)"
    )
    cap.release()
    out.release()
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    run_tracking()
