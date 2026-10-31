"""Helpers for feeding Ultralytics YOLO results into trackforge trackers.

They only read the result objects, so ``ultralytics`` stays an optional dependency
and is never imported here.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

Detection = Tuple[List[float], float, int]
Embedder = Callable[[Any, List[List[float]]], Sequence[Sequence[float]]]

__all__ = ["from_ultralytics", "track_ultralytics"]


def from_ultralytics(result: Any) -> List[Detection]:
    """Convert one Ultralytics result into trackforge detections.

    Args:
        result: A ``Results`` object from ``model.predict(...)`` or ``model(...)``, or
            the one-element list those calls return for a single image.

    Returns:
        A list of ``(tlwh, score, class_id)`` detections, in the same order as
        ``result.boxes``, so a track's ``det_ind`` indexes into ``result.boxes``.

    Raises:
        ValueError: If a list holding more than one result is passed.
    """
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ValueError(
                f"expected the result of a single image, got {len(result)} results"
            )
        result = result[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    return [
        ([x1, y1, x2 - x1, y2 - y1], float(score), int(cls))
        for (x1, y1, x2, y2), score, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
        )
    ]


def track_ultralytics(
    model: Any,
    source: Any,
    tracker: Any,
    embedder: Optional[Embedder] = None,
    **predict_kwargs: Any,
) -> Iterator[Tuple[Any, list]]:
    """Run an Ultralytics model over a source and track every frame.

    Args:
        model: A loaded ``ultralytics.YOLO`` (or RT-DETR) model.
        source: Anything ``model.predict`` accepts: a video path, webcam index,
            stream URL, image folder, and so on.
        tracker: A trackforge tracker such as ``trackforge.BYTETRACK()``.
        embedder: Optional ``embedder(frame, tlwh_boxes) -> embeddings`` callable for
            appearance trackers. ``DEEPSORT`` needs one; ``DEEPOCSORT``, ``BOTSORT``
            and ``TRACKTRACK`` use it when given.
        **predict_kwargs: Passed to ``model.predict``, for example ``classes=[0]``
            or ``conf=0.3``.

    Yields:
        ``(result, tracks)`` per frame, where ``tracks`` is the tracker output and
        ``result.orig_img`` is the BGR frame.
    """
    predict_kwargs.setdefault("stream", True)
    predict_kwargs.setdefault("verbose", False)
    for result in model.predict(source, **predict_kwargs):
        detections = from_ultralytics(result)
        if embedder is None:
            tracks = tracker.update(detections)
        else:
            boxes = [d[0] for d in detections]
            tracks = tracker.update(detections, embedder(result.orig_img, boxes))
        yield result, tracks
