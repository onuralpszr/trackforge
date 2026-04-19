from typing import List, Tuple

__all__ = ["ByteTrack", "Sort", "DeepSort", "DeepSortTrack"]

class ByteTrack:
    """
    ByteTrack tracker implementation.

    Use `ByteTrack()` to initialize and `update()` to process frames.

    **Usage Example:**

    ```python
    from trackforge import ByteTrack
    import numpy as np

    # Initialize tracker with default parameters
    tracker = ByteTrack(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        det_thresh=0.6
    )

    # Simulated detections: [x, y, w, h]
    # Format: (box, score, class_id)
    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0)
    ]

    # Update tracker
    tracks = tracker.update(detections)

    # Process active tracks
    for track in tracks:
        track_id, box, score, class_id = track
        print(f"Track ID: {track_id}, Box: {box}")
    ```
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        det_thresh: float = 0.6,
    ) -> None:
        """
        Initialize the ByteTrack tracker.

        Args:
            track_thresh (float, optional): High confidence detection threshold. Defaults to 0.5.
            track_buffer (int, optional): Number of frames to keep lost tracks alive. Defaults to 30.
            match_thresh (float, optional): IoU matching threshold. Defaults to 0.8.
            det_thresh (float, optional): Initialization threshold. Defaults to 0.6.
        """
        ...

    def update(
        self, output_results: List[Tuple[List[float], float, int]]
    ) -> List[Tuple[int, List[float], float, int]]:
        """
        Update the tracker with detections from the current frame.

        Args:
            output_results (list): A list of detections, where each detection is a tuple of
                ([x, y, w, h], score, class_id).

        Returns:
            list: A list of active tracks, where each track is a tuple of
                (track_id, [x, y, w, h], score, class_id).
        """
        ...

class Sort:
    """
    SORT (Simple Online and Realtime Tracking) tracker implementation.

    A simple yet effective multi-object tracker using Kalman filtering and IoU matching.

    **Usage Example:**

    ```python
    from trackforge import Sort

    # Initialize tracker with default parameters
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

    # Simulated detections: [x, y, w, h]
    # Format: (box, score, class_id)
    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0)
    ]

    # Update tracker
    tracks = tracker.update(detections)

    # Process confirmed tracks
    for track in tracks:
        track_id, box, score, class_id = track
        print(f"Track ID: {track_id}, Box: {box}")
    ```
    """

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the SORT tracker.

        Args:
            max_age (int, optional): Maximum frames to keep track without detection. Defaults to 1.
            min_hits (int, optional): Minimum hits before track is confirmed. Defaults to 3.
            iou_threshold (float, optional): IoU threshold for matching. Defaults to 0.3.
        """
        ...

    def update(
        self, detections: List[Tuple[List[float], float, int]]
    ) -> List[Tuple[int, List[float], float, int]]:
        """
        Update the tracker with detections from the current frame.

        Args:
            detections (list): A list of detections, where each detection is a tuple of
                ([x, y, w, h], score, class_id).

        Returns:
            list: A list of confirmed tracks, where each track is a tuple of
                (track_id, [x, y, w, h], score, class_id).
        """
        ...

class DeepSortTrack:
    """
    A confirmed track from the DeepSort tracker.

    Attributes:
        track_id (int): Unique track identifier.
        tlwh (List[float]): Bounding box in TLWH format [top, left, width, height].
        score (float): Detection confidence score.
        class_id (int): Object class identifier.
    """

    track_id: int
    tlwh: List[float]
    score: float
    class_id: int

class DeepSort:
    """
    Deep SORT tracker implementation with appearance feature matching.

    Deep SORT extends SORT by adding appearance descriptors (embeddings) for
    improved re-identification and reduced ID switches.

    **Usage Example:**

    ```python
    from trackforge import DeepSort
    import cv2
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image

    # Initialize tracker
    tracker = DeepSort(
        max_age=70,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100
    )

    # Load an embedder (e.g., ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize((128, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # For each frame:
    # 1. Get detections from your detector
    detections = [
        ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
        ([200.0, 200.0, 60.0, 120.0], 0.85, 0)
    ]

    # 2. Extract appearance embeddings for each detection
    embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # 512-dim vectors

    # 3. Update tracker
    tracks = tracker.update(detections, embeddings)

    # 4. Process tracks
    for track in tracks:
        print(f"ID: {track.track_id}, Box: {track.tlwh}")
    ```
    """

    def __init__(
        self,
        max_age: int = 70,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
    ) -> None:
        """
        Initialize the Deep SORT tracker.

        Args:
            max_age (int, optional): Maximum frames to keep track without detection. Defaults to 70.
            n_init (int, optional): Minimum consecutive detections to confirm a track. Defaults to 3.
            max_iou_distance (float, optional): Max IoU distance for matching. Defaults to 0.7.
            max_cosine_distance (float, optional): Max cosine distance for appearance matching. Defaults to 0.2.
            nn_budget (int, optional): Maximum appearance feature library size per track. Defaults to 100.
        """
        ...

    def update(
        self,
        detections: List[Tuple[List[float], float, int]],
        embeddings: List[List[float]],
    ) -> List[DeepSortTrack]:
        """
        Update the tracker with detections and their appearance embeddings.

        Args:
            detections (list): A list of detections, where each detection is a tuple of
                ([x, y, w, h], score, class_id).
            embeddings (list): A list of appearance embeddings (e.g., 512-dim vectors)
                corresponding to each detection. Must have same length as detections.

        Returns:
            list[DeepSortTrack]: A list of confirmed tracks with track_id, tlwh, score, class_id.

        Raises:
            ValueError: If the number of detections and embeddings don't match.
        """
        ...
