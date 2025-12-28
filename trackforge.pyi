from typing import List, Tuple, Optional

__all__ = ["ByteTrack"]


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
