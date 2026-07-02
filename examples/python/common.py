#!/usr/bin/env python3
"""Shared helpers for the trackforge Python examples.

Every demo in this directory reads a video, runs a detector, feeds detections
to a trackforge tracker, draws the results, and writes an annotated video. The
functions here factor out that shared plumbing so each demo only has to wire up
its detector and tracker.

Requirements:
    pip install opencv-python
"""

from __future__ import annotations

import time
from typing import NamedTuple, Optional, Sequence

import cv2

# BGR palette cycled by track id so the same id keeps a stable color.
PALETTE: list[tuple[int, int, int]] = [
    (255, 0, 0),  # blue
    (0, 255, 0),  # green
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 255),  # purple
    (255, 128, 0),  # orange
]

# A trackforge detection: (tlwh, score, class_id).
Detection = tuple[list[float], float, int]


class VideoInfo(NamedTuple):
    """Basic properties of an opened video.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total frame count (0 if the backend cannot report it).
    """

    width: int
    height: int
    fps: int
    total_frames: int


def color_for(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a track id.

    Args:
        track_id: Track identifier.

    Returns:
        A BGR color tuple drawn from :data:`PALETTE`.
    """
    return PALETTE[track_id % len(PALETTE)]


def label_for(track_id: int, class_name: str, score: float) -> str:
    """Format the on-frame label shared by every demo.

    Args:
        track_id: Track identifier.
        class_name: Human-readable class name from the detector.
        score: Detection confidence in ``[0, 1]``.

    Returns:
        A label string of the form ``"ID:3 person 0.87"``.
    """
    return f"ID:{track_id} {class_name} {score:.2f}"


def load_video(video_path: str) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open a video file and read its properties.

    Args:
        video_path: Path to the input video.

    Returns:
        A tuple of the open capture and its :class:`VideoInfo`.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=int(cap.get(cv2.CAP_PROP_FPS)) or 30,
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, info


def create_video_writer(
    output_path: str, info: VideoInfo, width: Optional[int] = None
) -> cv2.VideoWriter:
    """Create an MP4 writer matched to a video's size and frame rate.

    Args:
        output_path: Path for the annotated output video.
        info: Source video properties.
        width: Override output width (used by side-by-side comparisons).

    Returns:
        An opened ``cv2.VideoWriter``.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (width or info.width, info.height)
    return cv2.VideoWriter(output_path, fourcc, info.fps, size)


def yolo_detections(
    model, frame, classes: Optional[Sequence[int]] = None
) -> list[Detection]:
    """Run an Ultralytics YOLO model and convert boxes to trackforge detections.

    Args:
        model: A loaded ``ultralytics.YOLO`` model.
        frame: BGR image frame from OpenCV.
        classes: Optional class ids to keep (``None`` keeps all classes).

    Returns:
        A list of ``(tlwh, score, class_id)`` detections.
    """
    results = model.predict(frame, verbose=False, classes=classes)
    detections: list[Detection] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            score = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            detections.append((tlwh, score, class_id))
    return detections


def draw_track(
    frame,
    track_id: int,
    tlwh: Sequence[float],
    label: str,
    color: Optional[tuple[int, int, int]] = None,
) -> None:
    """Draw one track's box and a filled label onto a frame in place.

    Args:
        frame: BGR image frame to draw on.
        track_id: Track identifier (selects the color when ``color`` is None).
        tlwh: Bounding box as ``[x, y, w, h]``.
        label: Text to render above the box.
        color: Optional BGR override; defaults to :func:`color_for`.
    """
    if color is None:
        color = color_for(track_id)

    x, y, w, h = (int(v) for v in tlwh)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )


def draw_hud(frame, text: str) -> None:
    """Draw a heads-up banner (tracker name, frame index, track count).

    Args:
        frame: BGR image frame to draw on.
        text: Banner text rendered near the top-left corner.
    """
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def log_progress(
    frame_count: int, total_frames: int, start_time: float, every: int = 50
) -> None:
    """Print a throughput line every ``every`` frames.

    Args:
        frame_count: Frames processed so far.
        total_frames: Total frames in the video (0 if unknown).
        start_time: ``time.time()`` captured before the loop started.
        every: Print cadence in frames.
    """
    if frame_count % every:
        return
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed else 0.0
    total = total_frames or "?"
    print(f"  processed {frame_count}/{total} frames ({fps:.1f} fps)")
