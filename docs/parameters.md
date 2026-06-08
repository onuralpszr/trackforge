# Parameters

Every tracker exposes the same parameters in Python and Rust. In Python they are keyword
arguments to the tracker class (`BYTETRACK`, `DEEPSORT`, `OCSORT`, `SORT`); in Rust they are
positional arguments to `Tracker::new(...)`. The defaults below are identical across both.

## ByteTrack

`BYTETRACK(...)` / `ByteTrack::new(track_thresh, track_buffer, match_thresh, det_thresh)`

| Parameter      | Type  | Default | Description                                     |
| -------------- | ----- | ------- | ----------------------------------------------- |
| `track_thresh` | float | 0.5     | High confidence detection threshold             |
| `track_buffer` | int   | 30      | Frames to keep lost tracks alive                |
| `match_thresh` | float | 0.8     | IoU threshold for matching                      |
| `det_thresh`   | float | 0.6     | Minimum detection confidence for initialization |

## DeepSORT

`DEEPSORT(...)` / `DeepSort::new(extractor, max_age, n_init, max_iou_distance, max_cosine_distance, nn_budget)`

| Parameter             | Type  | Default | Description                                      |
| --------------------- | ----- | ------- | ------------------------------------------------ |
| `max_age`             | int   | 70      | Max frames to keep a track without detection     |
| `n_init`              | int   | 3       | Consecutive detections needed to confirm a track |
| `max_iou_distance`    | float | 0.7     | Max IoU distance for association                 |
| `max_cosine_distance` | float | 0.2     | Max cosine distance for Re-ID matching           |
| `nn_budget`           | int   | 100     | Max appearance feature library size per track    |

## OC-SORT

`OCSORT(...)` / `OcSort::new(max_age, min_hits, iou_threshold, delta_t, inertia)`

| Parameter       | Type  | Default | Description                                            |
| --------------- | ----- | ------- | ------------------------------------------------------ |
| `max_age`       | int   | 30      | Max frames to keep a lost track alive before deletion  |
| `min_hits`      | int   | 3       | Consecutive matched frames required to confirm a track |
| `iou_threshold` | float | 0.3     | IoU threshold for matching                             |
| `delta_t`       | int   | 3       | Observation window (frames) for velocity computation   |
| `inertia`       | float | 0.2     | Weight for the direction-consistency cost bonus (OCM)  |

## SORT

`SORT(...)` / `Sort::new(max_age, min_hits, iou_threshold)`

| Parameter       | Type  | Default | Description                                  |
| --------------- | ----- | ------- | -------------------------------------------- |
| `max_age`       | int   | 1       | Max frames to keep a track without detection |
| `min_hits`      | int   | 3       | Minimum hits before a track is confirmed     |
| `iou_threshold` | float | 0.3     | IoU threshold for matching                   |
