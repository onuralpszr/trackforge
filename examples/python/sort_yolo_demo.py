#!/usr/bin/env python3
"""
SORT Tracking Example with Ultralytics YOLO

This example demonstrates using the SORT tracker with a YOLO model for
multi-object tracking on video.

Requirements:
    pip install ultralytics opencv-python trackforge

Usage:
    python sort_yolo_demo.py
"""

import cv2
from ultralytics import YOLO
import trackforge
import time
from pathlib import Path


def run_tracking(
    video_path: str = "people.mp4",
    output_path: str = "output_sort_yolo.mp4",
    model_path: str = "yolo11n.pt",
):
    """
    Run SORT tracking with YOLO detection on a video.

    Args:
        video_path: Path to input video file.
        output_path: Path for output video with tracking annotations.
        model_path: Path to YOLO model weights.
    """
    print(f"🚀 Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Initialize SORT Tracker
    # max_age=30: Keep tracks alive for 30 frames without detection
    # min_hits=3: Require 3 consecutive detections to confirm track
    # iou_threshold=0.3: Minimum IoU for matching
    print("📦 Initializing SORT tracker...")
    tracker = trackforge.SORT(max_age=30, min_hits=3, iou_threshold=0.3)

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    t0 = time.time()

    # Color palette for different track IDs
    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]

    print("🎬 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO Detection
        results = model.predict(
            frame, verbose=False, classes=[0]
        )  # Only detect persons (class 0)

        # Prepare detections for SORT tracker: (tlwh, score, class_id)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                tlwh = [float(x1), float(y1), float(w), float(h)]
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append((tlwh, conf, cls))

        # Update SORT Tracker
        # Returns: list of (track_id, tlwh, score, class_id)
        tracks = tracker.update(detections)

        # Draw tracks
        for track in tracks:
            track_id, tlwh, score, class_id = track
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h

            # Get color based on track ID
            color = colors[track_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label with track ID
            label = f"ID:{track_id} {model.names[class_id]} {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # Draw frame info
        info_text = (
            f"SORT + YOLO | Frame: {frame_count}/{total_frames} | Tracks: {len(tracks)}"
        )
        cv2.putText(
            frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        out.write(frame)

        if frame_count % 50 == 0:
            elapsed = time.time() - t0
            fps_actual = frame_count / elapsed
            print(
                f"  Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} fps)"
            )

    t1 = time.time()
    total_time = t1 - t0
    avg_fps = frame_count / total_time

    print("\n✅ Done!")
    print(f"   Processed {frame_count} frames in {total_time:.2f}s ({avg_fps:.1f} fps)")
    print(f"   Output saved to: {output_path}")

    cap.release()
    out.release()


if __name__ == "__main__":
    # Check if video exists
    video_file = Path("people.mp4")
    if not video_file.exists():
        print("⚠️  Video file 'people.mp4' not found in current directory.")
        print("   Please provide a video file or update the path.")
    else:
        run_tracking()
