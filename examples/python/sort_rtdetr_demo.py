#!/usr/bin/env python3
"""
SORT Tracking Example with RT-DETR from Transformers

This example demonstrates using the SORT tracker with RT-DETR (Real-Time DEtection TRansformer)
from Hugging Face Transformers for multi-object tracking on video.

RT-DETR is a real-time end-to-end object detector that achieves state-of-the-art
performance while maintaining high inference speed.

Requirements:
    pip install transformers torch opencv-python trackforge pillow

Usage:
    python sort_rtdetr_demo.py
"""

import cv2
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import trackforge
import time
from pathlib import Path


# COCO class names for RT-DETR
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


def run_tracking(
    video_path: str = "people.mp4",
    output_path: str = "output_sort_rtdetr.mp4",
    model_name: str = "PekingU/rtdetr_r50vd",
    confidence_threshold: float = 0.5,
    target_classes: list = None,  # None means all classes, [0] means only person
):
    """
    Run SORT tracking with RT-DETR detection on a video.

    Args:
        video_path: Path to input video file.
        output_path: Path for output video with tracking annotations.
        model_name: Hugging Face model name for RT-DETR.
        confidence_threshold: Minimum confidence for detections.
        target_classes: List of class indices to track (None for all).
    """
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🖥️  Using device: {device}")

    # Load RT-DETR model from Transformers
    print(f"🚀 Loading RT-DETR model: {model_name}")
    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()

    # Initialize SORT Tracker
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

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run RT-DETR Detection
        with torch.no_grad():
            inputs = image_processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)

        # Post-process detections
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[height, width]]).to(device),
            threshold=confidence_threshold,
        )[0]

        # Prepare detections for SORT tracker: (tlwh, score, class_id)
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            # Filter by target classes if specified
            if target_classes is not None and label.item() not in target_classes:
                continue

            # Convert xyxy to tlwh
            x1, y1, x2, y2 = box.cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            tlwh = [float(x1), float(y1), float(w), float(h)]
            conf = float(score.cpu().numpy())
            cls = int(label.cpu().numpy())
            detections.append((tlwh, conf, cls))

        # Update SORT Tracker
        tracks = tracker.update(detections)

        # Draw tracks
        for track in tracks:
            track_id, tlwh, score, class_id = track
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h

            # Get color based on track ID
            color = colors[track_id % len(colors)]

            # Get class name
            class_name = (
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES)
                else f"cls{class_id}"
            )

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label with track ID
            label = f"ID:{track_id} {class_name} {score:.2f}"
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
        info_text = f"SORT + RT-DETR | Frame: {frame_count}/{total_frames} | Tracks: {len(tracks)}"
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
        # Track only persons (class 0) for people.mp4
        run_tracking(target_classes=[0])
