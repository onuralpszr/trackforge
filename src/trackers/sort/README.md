# SORT: Simple Online and Realtime Tracking

This module implements the SORT (Simple Online and Realtime Tracking) algorithm.

## Algorithm Overview

SORT is a simple yet effective multi-object tracking algorithm that combines:
- **Kalman Filtering** for motion prediction
- **Hungarian Algorithm** for data association using IoU (Intersection over Union)

## Key Features

- **Real-time performance**: Designed for speed with minimal computational overhead
- **No appearance features**: Uses only bounding box information
- **Simple track management**: Tracks are created, confirmed, and deleted based on hit/miss counts

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 1 | Maximum frames to keep a track without detection |
| `min_hits` | 3 | Minimum consecutive hits before track is confirmed |
| `iou_threshold` | 0.3 | Minimum IoU for matching detection to track |

## References

> **Simple Online and Realtime Tracking**
> Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft
> IEEE International Conference on Image Processing (ICIP), 2016
> [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)
