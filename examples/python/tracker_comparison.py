#!/usr/bin/env python3
"""
Tracker Comparison: ByteTrack vs SORT

This example compares ByteTrack and SORT trackers side-by-side on the same video
to demonstrate the differences in tracking behavior.

Requirements:
    pip install ultralytics opencv-python trackforge

Usage:
    python tracker_comparison.py
"""

import cv2
from ultralytics import YOLO
import trackforge
import time
from pathlib import Path


def run_comparison(
    video_path: str = "people.mp4",
    output_path: str = "output_comparison.mp4",
    model_path: str = "yolo11n.pt",
):
    """
    Run both ByteTrack and SORT on the same video for comparison.
    """
    print(f"🚀 Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Initialize both trackers
    print("📦 Initializing trackers...")
    bytetrack = trackforge.ByteTrack(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        det_thresh=0.6,
    )
    sort = trackforge.Sort(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
    )

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Create side-by-side output (double width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    frame_count = 0
    t0 = time.time()

    # Colors for each tracker
    bytetrack_color = (0, 255, 0)  # Green for ByteTrack
    sort_color = (255, 128, 0)  # Orange for SORT

    print("🎬 Processing video (ByteTrack left, SORT right)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO Detection (only persons)
        results = model.predict(frame, verbose=False, classes=[0])

        # Prepare detections
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append((tlwh, conf, cls))

        # Create two copies of the frame
        frame_bt = frame.copy()
        frame_sort = frame.copy()

        # Update ByteTrack
        bt_tracks = bytetrack.update(detections)
        for track in bt_tracks:
            track_id, tlwh, score, class_id = track
            x1, y1, w, h = tlwh
            cv2.rectangle(
                frame_bt,
                (int(x1), int(y1)),
                (int(x1 + w), int(y1 + h)),
                bytetrack_color,
                2,
            )
            cv2.putText(
                frame_bt,
                f"ID:{track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                bytetrack_color,
                2,
            )

        # Update SORT
        sort_tracks = sort.update(detections)
        for track in sort_tracks:
            track_id, tlwh, score, class_id = track
            x1, y1, w, h = tlwh
            cv2.rectangle(
                frame_sort,
                (int(x1), int(y1)),
                (int(x1 + w), int(y1 + h)),
                sort_color,
                2,
            )
            cv2.putText(
                frame_sort,
                f"ID:{track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                sort_color,
                2,
            )

        # Add labels
        cv2.putText(
            frame_bt,
            f"ByteTrack | Tracks: {len(bt_tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            bytetrack_color,
            2,
        )
        cv2.putText(
            frame_sort,
            f"SORT | Tracks: {len(sort_tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            sort_color,
            2,
        )

        # Combine side-by-side
        combined = cv2.hconcat([frame_bt, frame_sort])

        # Add frame counter in center
        cv2.putText(
            combined,
            f"Frame: {frame_count}/{total_frames}",
            (width - 100, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        out.write(combined)

        if frame_count % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Processed {frame_count}/{total_frames} frames ({frame_count / elapsed:.1f} fps)"
            )

    t1 = time.time()
    print(
        f"\n✅ Done! Processed {frame_count} frames in {t1 - t0:.2f}s ({frame_count / (t1 - t0):.1f} fps)"
    )
    print(f"   Output saved to: {output_path}")

    cap.release()
    out.release()


if __name__ == "__main__":
    video_file = Path("people.mp4")
    if not video_file.exists():
        print("⚠️  Video file 'people.mp4' not found in current directory.")
    else:
        run_comparison()
