#!/usr/bin/env python3
"""Deep SORT Tracking Example with Ultralytics YOLO.

This example demonstrates using the Deep SORT tracker with a YOLO model for
multi-object tracking on video. It uses a pre-trained ResNet18 for appearance embeddings.

Requirements:
    pip install ultralytics opencv-python trackforge torch torchvision

Example:
    Run with default settings::

        $ python deepsort_demo.py --video people.mp4 --model yolo11n.pt
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

import trackforge


def get_embedder():
    """Load a pre-trained ResNet18 for embedding extraction.

    Uses ResNet18 pretrained on ImageNet as a generic feature extractor.
    Ideally, use a ReID-specific model (OSNet, etc.), but this suffices for demo.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The ResNet18 model with identity FC layer.
            - transform (torchvision.transforms.Compose): Image preprocessing transforms.

    Example:
        >>> model, transform = get_embedder()
        >>> print(model)
    """
    print("🧠 Loading standard ResNet18 for embeddings...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove classification layer to get feature vector (512-dim)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize((128, 64)),  # Typical ReID size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform


def extract_features(model, transform, frame, bboxes):
    """Extract appearance features for a batch of bounding boxes.

    Crops each bounding box from the frame, applies preprocessing transforms,
    and extracts normalized feature embeddings using the provided model.

    Args:
        model (torch.nn.Module): The feature extraction model (e.g., ResNet18).
        transform (torchvision.transforms.Compose): Image preprocessing transforms.
        frame (numpy.ndarray): BGR image frame from OpenCV (H, W, C).
        bboxes (list): List of bounding boxes in TLWH format [[x, y, w, h], ...].

    Returns:
        list: List of feature vectors, each as a list of floats (512-dim for ResNet18).
            Returns empty list if bboxes is empty.

    Example:
        >>> model, transform = get_embedder()
        >>> frame = cv2.imread("image.jpg")
        >>> bboxes = [[100, 100, 50, 100], [200, 200, 60, 120]]
        >>> features = extract_features(model, transform, frame, bboxes)
        >>> print(len(features), len(features[0]))
        2 512
    """
    if not bboxes:
        return []

    crops = []
    h, w, _ = frame.shape

    for bbox in bboxes:
        x1, y1, w_box, h_box = bbox
        x2, y2 = x1 + w_box, y1 + h_box

        # Clamp coordinates
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            # Handle degenerate box by creating a dummy crop
            crop = np.zeros((128, 64, 3), dtype=np.uint8)
        else:
            crop = frame[y1:y2, x1:x2]
            # Convert BGR to RGB for PIL/Torch
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crops.append(Image.fromarray(crop))

    # Batch process
    tensors = [transform(img) for img in crops]
    batch = torch.stack(tensors)

    with torch.no_grad():
        features = model(batch)

    # Normalize features
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    return features.numpy().tolist()


def run_tracking(args):
    """Run Deep SORT tracking on a video file.

    Loads a YOLO model for detection, ResNet18 for appearance embeddings,
    and Deep SORT for multi-object tracking. Processes the input video
    frame by frame and writes annotated output.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - video (str): Path to input video file.
            - output (str): Path to output video file.
            - model (str): Path to YOLO model file.

    Returns:
        None

    Example:
        >>> args = argparse.Namespace(
        ...     video="people.mp4",
        ...     output="output_deepsort.mp4",
        ...     model="yolo11n.pt"
        ... )
        >>> run_tracking(args)
    """
    video_path = args.video
    output_path = args.output
    model_path = args.model

    print(f"🚀 Loading YOLO model: {model_path}")
    yolo = YOLO(model_path)

    embedder, transform = get_embedder()

    # Initialize Deep SORT Tracker
    print("📦 Initializing Deep SORT tracker...")
    tracker = trackforge.DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    t0 = time.time()
    colors = np.random.randint(0, 255, (1000, 3)).tolist()

    print("🎬 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Detect
        results = yolo.predict(frame, verbose=False, classes=[0])  # Class 0 = person

        detections = []  # (tlwh, score, cls)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w_box = x2 - x1
                h_box = y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append((
                    [float(x1), float(y1), float(w_box), float(h_box)],
                    conf,
                    cls
                ))

        # 2. Extract Embeddings
        bboxes = [d[0] for d in detections]
        embeddings = extract_features(embedder, transform, frame, bboxes)

        # 3. Track
        tracks = tracker.update(detections, embeddings)

        # 4. Draw
        for track in tracks:
            tid = track.track_id
            x1, y1, w, h = track.tlwh
            x2, y2 = x1 + w, y1 + h

            color = colors[tid % len(colors)]

            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
            )
            label = f"ID:{tid}"
            cv2.putText(
                frame, label, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        out.write(frame)

        if frame_count % 20 == 0:
            print(f"Frame {frame_count} - Tracks: {len(tracks)}")

    elapsed = time.time() - t0
    print(f"✅ Done! Processed {frame_count} frames in {elapsed:.2f}s")
    print(f"📹 Saved to {output_path}")
    cap.release()
    out.release()


def main():
    """Parse command line arguments and run tracking."""
    parser = argparse.ArgumentParser(
        description="Deep SORT tracking with YOLO detection and ResNet18 embeddings."
    )
    parser.add_argument(
        "--video", type=str, default="people.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output", type=str, default="output_deepsort.mp4",
        help="Path to output video"
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt",
        help="YOLO model path"
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"⚠️ Video {args.video} not found.")
        return

    run_tracking(args)


if __name__ == "__main__":
    main()

