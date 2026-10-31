#!/usr/bin/env python3
"""Shortest path from an Ultralytics model to tracked objects.

``trackforge.track_ultralytics`` runs the model over any source Ultralytics can
read (video, webcam, stream, image folder) and yields the tracks for every frame.

Requirements:
    pip install ultralytics trackforge

Example:
    $ python ultralytics_track.py --video people.mp4 --tracker BOTSORT
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

import trackforge

TRACKERS = ["SORT", "BYTETRACK", "OCSORT", "BOTSORT", "TRACKTRACK"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", default="people.mp4", help="any Ultralytics source")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    parser.add_argument("--tracker", default="BYTETRACK", choices=TRACKERS)
    args = parser.parse_args()

    model = YOLO(args.model)
    tracker = getattr(trackforge, args.tracker)()

    for frame_idx, (result, tracks) in enumerate(
        trackforge.track_ultralytics(model, args.video, tracker, classes=[0])
    ):
        print(f"frame {frame_idx}: {len(tracks)} tracks")
        for track_id, tlwh, score, class_id, det_ind in tracks:
            x, y, w, h = tlwh
            print(
                f"  id {track_id:>3} {result.names[class_id]:<8} {score:.2f}"
                f" [{x:.0f} {y:.0f} {w:.0f} {h:.0f}]"
            )


if __name__ == "__main__":
    main()
