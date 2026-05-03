from typing import List, Tuple

__all__ = ["BYTETRACK", "SORT", "OCSORT", "DEEPSORT"]

class BYTETRACK:
    """
    ByteTrack tracker implementation.

    **Usage Example:**

    ```python
    from trackforge import BYTETRACK

    tracker = BYTETRACK(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        det_thresh=0.6,
    )

    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
    ]

    tracks = tracker.update(detections)
    for track_id, box, score, class_id in tracks:
        print(f"Track ID: {track_id}, Box: {box}")
    ```
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        det_thresh: float = 0.6,
    ) -> None: ...

    def update(
        self, output_results: List[Tuple[List[float], float, int]]
    ) -> List[Tuple[int, List[float], float, int]]: ...

class SORT:
    """
    SORT (Simple Online and Realtime Tracking) tracker implementation.

    **Usage Example:**

    ```python
    from trackforge import SORT

    tracker = SORT(max_age=1, min_hits=3, iou_threshold=0.3)

    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ]

    tracks = tracker.update(detections)
    for track_id, box, score, class_id in tracks:
        print(f"Track ID: {track_id}, Box: {box}")
    ```
    """

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None: ...

    def update(
        self, detections: List[Tuple[List[float], float, int]]
    ) -> List[Tuple[int, List[float], float, int]]: ...

class OCSORT:
    """
    OC-SORT (Observation-Centric SORT) tracker implementation.

    **Usage Example:**

    ```python
    from trackforge import OCSORT

    tracker = OCSORT(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        inertia=0.2,
    )

    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ]

    tracks = tracker.update(detections)
    for track_id, box, score, class_id in tracks:
        print(f"Track ID: {track_id}, Box: {box}")
    ```
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
    ) -> None: ...

    def update(
        self, detections: List[Tuple[List[float], float, int]]
    ) -> List[Tuple[int, List[float], float, int]]: ...

class DEEPSORT:
    """
    DeepSORT tracker implementation with appearance feature matching.

    **Usage Example:**

    ```python
    from trackforge import DEEPSORT
    import numpy as np

    tracker = DEEPSORT(
        max_age=70,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
    )

    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
    ]
    embeddings = [np.random.rand(128).tolist()]

    tracks = tracker.update(detections, embeddings)
    for track_id, box, score, class_id in tracks:
        print(f"Track ID: {track_id}, Box: {box}")
    ```
    """

    def __init__(
        self,
        max_age: int = 70,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
    ) -> None: ...

    def update(
        self,
        detections: List[Tuple[List[float], float, int]],
        embeddings: List[List[float]],
    ) -> List[Tuple[int, List[float], float, int]]: ...
