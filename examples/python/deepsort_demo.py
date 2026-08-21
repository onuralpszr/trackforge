#!/usr/bin/env python3
"""Deep SORT tracking demo with YOLO detection and ResNet18 appearance features.

Requirements:
    pip install ultralytics opencv-python trackforge torch torchvision pillow

Example:
    $ python deepsort_demo.py --video people.mp4 --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

import trackforge
from common import (
    create_video_writer,
    draw_hud,
    draw_track,
    label_for,
    load_video,
    log_progress,
    yolo_detections,
)
from reid import extract_features, get_embedder


def run_tracking(video: str, output: str, model_path: str) -> None:
    """Run Deep SORT over a video and write an annotated copy.

    Args:
        video: Path to the input video.
        output: Path for the annotated output video.
        model_path: Path to the YOLO model weights.
    """
    model = YOLO(model_path)
    embedder, transform = get_embedder()
    tracker = trackforge.DEEPSORT(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
    )

    cap, info = load_video(video)
    writer = create_video_writer(output, info)
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
        embeddings = extract_features(
            embedder, transform, frame, [d[0] for d in detections]
        )
        tracks = tracker.update(detections, embeddings)
        for track_id, tlwh, score, class_id, det_ind in tracks:
            draw_track(
                frame, track_id, tlwh, label_for(track_id, model.names[class_id], score)
            )

        draw_hud(
            frame,
            f"Deep SORT | frame {frame_count}/{info.total_frames} | tracks {len(tracks)}",
        )
        writer.write(frame)
        log_progress(frame_count, info.total_frames, start)

    cap.release()
    writer.release()
    print(f"done: {frame_count} frames -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep SORT tracking with YOLO detection and ResNet18 embeddings."
    )
    parser.add_argument("--video", default="people.mp4", help="input video path")
    parser.add_argument(
        "--output", default="output_deepsort.mp4", help="output video path"
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"video not found: {args.video}")
        return
    run_tracking(args.video, args.output, args.model)


if __name__ == "__main__":
    main()
