# Parameters

Every tracker takes the same parameters in Python and Rust. In Python they are keyword arguments to the tracker class. In Rust they are positional arguments to `Tracker::new(...)`, or fields on a params struct passed to `Tracker::from_params(...)`. The defaults are identical across both languages.

## Two ideas that show up everywhere

Two settings mean the same thing in almost every tracker, even though the names differ.

- Lost track survival. How many frames a track stays alive after it stops matching any detection. If an object is missed or hidden for up to this many frames, the track keeps its id and can be picked back up. Past that it is dropped. Larger rides out longer occlusions but risks handing an old id to a new object. It is called `max_age` in SORT, OC-SORT, DeepSORT, and Deep OC-SORT, and `track_buffer` in ByteTrack and BoT-SORT.
- Confirmation. How many matched frames in a row a new track needs before it is reported. Larger hides flickering false tracks but delays real ones. It is called `min_hits` in SORT, OC-SORT, and Deep OC-SORT, and `n_init` in DeepSORT. ByteTrack and BoT-SORT confirm on the first high confidence match, so they have no such setting.

In Rust these two live together in `CommonParams { max_age, min_hits }`, which every tracker's params struct embeds as `common`.

## A note on IoU thresholds

IoU thresholds come in two opposite flavors, matching the original papers. Read the description before tuning.

- Minimum IoU. The boxes must overlap by at least this much. Higher is stricter. Used by `iou_threshold` in SORT, OC-SORT, and Deep OC-SORT.
- Maximum IoU distance. The cost is one minus IoU, and a pair matches only when that cost is at or below the value, so lower is stricter. A value of 0.8 means the boxes only need an IoU of 0.2. Used by `match_thresh` in ByteTrack and BoT-SORT, and `max_iou_distance` in DeepSORT.

## SORT

`SORT(...)` and `Sort::new(max_age, min_hits, iou_threshold)`

| Parameter       | Type  | Default | Meaning                                                                                      |
| --------------- | ----- | ------- | -------------------------------------------------------------------------------------------- |
| `max_age`       | int   | 1       | Frames a track survives with no matched detection before it is dropped.                      |
| `min_hits`      | int   | 3       | Matched frames in a row before a track is reported.                                          |
| `iou_threshold` | float | 0.3     | Minimum IoU overlap to treat a detection and a track as the same object. Higher is stricter. |

## ByteTrack

`BYTETRACK(...)` and `ByteTrack::new(track_thresh, track_buffer, match_thresh, det_thresh)`

| Parameter             | Type  | Default | Meaning                                                                                                                        |
| --------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `track_thresh`        | float | 0.5     | Score above which a detection is high confidence and matched first. Lower scores are held for the second pass.                 |
| `track_buffer`        | int   | 30      | Frames a lost track is kept alive so it can be recovered.                                                                      |
| `match_thresh`        | float | 0.8     | First stage match cutoff as a maximum IoU distance. Lower is stricter.                                                         |
| `det_thresh`          | float | 0.6     | Smallest score an unmatched high confidence detection needs to start a new track.                                              |
| `second_match_thresh` | float | 0.5     | Second stage match cutoff for recovering low confidence detections, a maximum IoU distance. This is ByteTrack's recovery step. |

## OC-SORT

`OCSORT(...)` and `OcSort::new(max_age, min_hits, iou_threshold, delta_t, inertia)`

| Parameter       | Type  | Default | Meaning                                                                                                    |
| --------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `max_age`       | int   | 30      | Frames a lost track survives before deletion.                                                              |
| `min_hits`      | int   | 3       | Matched frames in a row before a track is reported.                                                        |
| `iou_threshold` | float | 0.3     | Minimum IoU overlap to associate. Higher is stricter.                                                      |
| `delta_t`       | int   | 3       | How many frames back to look when estimating an object's direction of travel from its real past positions. |
| `inertia`       | float | 0.2     | How strongly that direction of travel is trusted when matching, from zero to one. Zero turns it off.       |

## DeepSORT

`DEEPSORT(...)` and `DeepSort::new(extractor, max_age, n_init, max_iou_distance, max_cosine_distance, nn_budget)`

| Parameter             | Type  | Default | Meaning                                                                                                    |
| --------------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `max_age`             | int   | 70      | Frames a track survives with no matched detection before deletion.                                         |
| `n_init`              | int   | 3       | Matched frames in a row before a track is confirmed.                                                       |
| `max_iou_distance`    | float | 0.7     | IoU fallback match cutoff as a maximum IoU distance. Lower is stricter.                                    |
| `max_cosine_distance` | float | 0.2     | How different two appearance embeddings may be and still count as the same object. Lower cuts id switches. |
| `nn_budget`           | int   | 100     | How many past appearance embeddings to keep per track. When full the oldest is dropped.                    |

## Deep OC-SORT

`DEEPOCSORT(...)` and `DeepOcSort::new(extractor, max_age, min_hits, iou_threshold, delta_t, inertia, appearance_weight, max_cosine_distance, nn_budget)`

| Parameter             | Type  | Default | Meaning                                                                                                            |
| --------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `max_age`             | int   | 30      | Frames a lost track survives before deletion.                                                                      |
| `min_hits`            | int   | 3       | Matched frames in a row before a track is reported.                                                                |
| `iou_threshold`       | float | 0.3     | Minimum IoU overlap to associate. Higher is stricter.                                                              |
| `delta_t`             | int   | 3       | Frames back to look when estimating direction of travel.                                                           |
| `inertia`             | float | 0.2     | How strongly direction of travel is trusted, from zero to one.                                                     |
| `appearance_weight`   | float | 0.5     | How much the appearance match counts against the motion match, from zero to one. Zero falls back to plain OC-SORT. |
| `max_cosine_distance` | float | 0.2     | How different two embeddings may be and still match. Lower is stricter.                                            |
| `nn_budget`           | int   | 100     | Past embeddings kept per track.                                                                                    |

## BoT-SORT

`BOTSORT(...)` and `BotSort::new(track_thresh, track_buffer, match_thresh, det_thresh, proximity_thresh, appearance_thresh)`

| Parameter             | Type  | Default | Meaning                                                                                                  |
| --------------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------- |
| `track_thresh`        | float | 0.5     | Score above which a detection is high confidence and matched first.                                      |
| `track_buffer`        | int   | 30      | Frames a lost track is kept alive.                                                                       |
| `match_thresh`        | float | 0.8     | First stage match cutoff as a maximum IoU distance. Lower is stricter.                                   |
| `det_thresh`          | float | 0.6     | Smallest score an unmatched high confidence detection needs to start a new track.                        |
| `second_match_thresh` | float | 0.5     | Second stage match cutoff for recovering low confidence detections, a maximum IoU distance.              |
| `proximity_thresh`    | float | 0.5     | How much boxes must overlap before appearance is allowed to influence the match, a maximum IoU distance. |
| `appearance_thresh`   | float | 0.25    | How close two embeddings must be for Re-ID to help the match, a maximum cosine distance.                 |

## TrackTrack

`TRACKTRACK(...)` and `TrackTrack::from_params(...)`

| Parameter      | Type  | Default | Meaning                                                                                                                                                                    |
| -------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det_thresh`   | float | 0.6     | Score above which a detection is high confidence and matched first.                                                                                                        |
| `match_thresh` | float | 0.7     | Association cost gate. A pair matches only when its fused cost is below this. Lower is stricter.                                                                           |
| `track_buffer` | int   | 30      | Frames a lost track is kept alive.                                                                                                                                         |
| `min_hits`     | int   | 3       | Matched frames in a row before a new track is confirmed.                                                                                                                   |
| `init_thresh`  | float | 0.7     | Smallest score a leftover detection needs before it may start a new track.                                                                                                 |
| `tai_thresh`   | float | 0.55    | Overlap gate for track-aware initialization, a maximum IoU. A leftover detection is dropped if it overlaps an active track or a more confident leftover by more than this. |
| `penalty_low`  | float | 0.2     | Extra cost added to low confidence detections during association, so they only win when nothing better is available.                                                       |
| `reduce_step`  | float | 0.05    | How much the cost gate tightens on each round of the track-perspective matching loop.                                                                                      |
