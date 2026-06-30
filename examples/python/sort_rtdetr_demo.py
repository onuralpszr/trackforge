#!/usr/bin/env python3
"""SORT tracking demo with an RT-DETR detector from Hugging Face Transformers.

RT-DETR is a real-time end-to-end transformer detector. This demo mirrors the
YOLO demos but swaps in RT-DETR for detection.

Requirements:
    pip install transformers torch pillow opencv-python trackforge

Example:
    $ python sort_rtdetr_demo.py --video people.mp4 --model PekingU/rtdetr_r50vd
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

import trackforge
from common import (
    Detection,
    create_video_writer,
    draw_hud,
    draw_track,
    label_for,
    load_video,
    log_progress,
)

# COCO class ids used by RT-DETR; index 0 is "person".
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def pick_device() -> str:
    """Return the best available torch device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def rtdetr_detections(
    model, processor, device, frame, width, height, threshold, classes
):
    """Run RT-DETR and convert its output to trackforge detections.

    Args:
        model: Loaded ``RTDetrForObjectDetection``.
        processor: Matching ``RTDetrImageProcessor``.
        device: Torch device string.
        frame: BGR image frame from OpenCV.
        width: Frame width in pixels.
        height: Frame height in pixels.
        threshold: Minimum detection confidence.
        classes: Class ids to keep, or None for all.

    Returns:
        A list of ``(tlwh, score, class_id)`` detections.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[height, width]]).to(device),
        threshold=threshold,
    )[0]

    detections: list[Detection] = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        class_id = int(label.item())
        if classes is not None and class_id not in classes:
            continue
        x1, y1, x2, y2 = box.cpu().numpy()
        tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        detections.append((tlwh, float(score.item()), class_id))
    return detections


def run_tracking(video: str, output: str, model_name: str) -> None:
    """Run SORT with RT-DETR detection over a video and write an annotated copy.

    Args:
        video: Path to the input video.
        output: Path for the annotated output video.
        model_name: Hugging Face model id for RT-DETR.
    """
    device = pick_device()
    print(f"device: {device}")
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    tracker = trackforge.SORT(max_age=30, min_hits=3, iou_threshold=0.3)

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

        detections = rtdetr_detections(
            model, processor, device, frame, info.width, info.height, 0.5, classes=[0]
        )
        tracks = tracker.update(detections)
        for track_id, tlwh, score, class_id in tracks:
            name = (
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES)
                else f"cls{class_id}"
            )
            draw_track(frame, track_id, tlwh, label_for(track_id, name, score))

        draw_hud(
            frame,
            f"SORT + RT-DETR | frame {frame_count}/{info.total_frames} | tracks {len(tracks)}",
        )
        writer.write(frame)
        log_progress(frame_count, info.total_frames, start)

    cap.release()
    writer.release()
    print(f"done: {frame_count} frames -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SORT tracking with RT-DETR detection."
    )
    parser.add_argument("--video", default="people.mp4", help="input video path")
    parser.add_argument(
        "--output", default="output_sort_rtdetr.mp4", help="output video path"
    )
    parser.add_argument(
        "--model", default="PekingU/rtdetr_r50vd", help="RT-DETR model id"
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"video not found: {args.video}")
        return
    run_tracking(args.video, args.output, args.model)


if __name__ == "__main__":
    main()
