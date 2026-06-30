#!/usr/bin/env python3
"""Side-by-side comparison of ByteTrack and SORT on the same video.

Runs both trackers over identical YOLO detections and writes a double-width
video with ByteTrack on the left and SORT on the right.

Requirements:
    pip install ultralytics opencv-python trackforge

Example:
    $ python tracker_comparison.py --video people.mp4 --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

import trackforge
from common import (
    create_video_writer,
    draw_hud,
    draw_track,
    load_video,
    log_progress,
    yolo_detections,
)

BYTETRACK_COLOR = (0, 255, 0)  # green
SORT_COLOR = (255, 128, 0)  # orange


def run_comparison(video: str, output: str, model_path: str) -> None:
    """Run ByteTrack and SORT side by side and write the combined video.

    Args:
        video: Path to the input video.
        output: Path for the annotated side-by-side output video.
        model_path: Path to the YOLO model weights.
    """
    model = YOLO(model_path)
    bytetrack = trackforge.BYTETRACK(
        track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6
    )
    sort = trackforge.SORT(max_age=30, min_hits=3, iou_threshold=0.3)

    cap, info = load_video(video)
    writer = create_video_writer(output, info, width=info.width * 2)
    print(
        f"video: {info.width}x{info.height} @ {info.fps}fps, {info.total_frames} frames"
    )

    frame_count = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        detections = yolo_detections(model, frame, classes=[0])
        frame_bt, frame_sort = frame.copy(), frame.copy()

        bt_tracks = bytetrack.update(detections)
        for track_id, tlwh, _, _ in bt_tracks:
            draw_track(
                frame_bt, track_id, tlwh, f"ID:{track_id}", color=BYTETRACK_COLOR
            )

        sort_tracks = sort.update(detections)
        for track_id, tlwh, _, _ in sort_tracks:
            draw_track(frame_sort, track_id, tlwh, f"ID:{track_id}", color=SORT_COLOR)

        draw_hud(frame_bt, f"ByteTrack | tracks {len(bt_tracks)}")
        draw_hud(frame_sort, f"SORT | tracks {len(sort_tracks)}")
        combined = cv2.hconcat([frame_bt, frame_sort])
        writer.write(combined)
        log_progress(frame_count, info.total_frames, start)

    cap.release()
    writer.release()
    print(f"done: {frame_count} frames -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ByteTrack and SORT side by side."
    )
    parser.add_argument("--video", default="people.mp4", help="input video path")
    parser.add_argument(
        "--output", default="output_comparison.mp4", help="output video path"
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"video not found: {args.video}")
        return
    run_comparison(args.video, args.output, args.model)


if __name__ == "__main__":
    main()
