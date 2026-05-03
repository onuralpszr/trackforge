# OC-SORT: Observation-Centric SORT

This module implements OC-SORT (Observation-Centric SORT), a robust extension of SORT that
addresses tracker drift during occlusions by anchoring motion estimation to raw detections
rather than Kalman filter predictions.

## Algorithm Overview

SORT accumulates prediction errors during occlusions because its Kalman filter integrates
noisy velocity estimates when no observation is available. OC-SORT corrects this with two
complementary mechanisms:

**Observation-Centric Velocity (OCV)**
Velocity is computed directly from consecutive detections, not from the Kalman filter state.
This produces a more reliable momentum signal because it is grounded in actual observations.

**Observation-Centric Momentum (OCM)**
Before IoU matching, a direction-consistency bonus is added to each IoU score based on how
well the observed velocity direction aligns with the vector from the track's last observation
to each candidate detection. Matches that are consistent with the track's momentum receive a
higher effective IoU, improving association after missed frames.

**Observation-Centric Re-Update (ORU)**
When a lost track is re-matched after a gap, the Kalman filter is "re-wound" by replaying
linearly interpolated observations between the last seen position and the current detection.
This corrects the accumulated drift so that future predictions start from an accurate state.

## Parameters

| Parameter       | Default | Description                                                  |
| --------------- | ------- | ------------------------------------------------------------ |
| `max_age`       | 30      | Frames a lost track is kept alive before deletion            |
| `min_hits`      | 3       | Consecutive matched frames required to confirm a track       |
| `iou_threshold` | 0.3     | IoU threshold for detection-to-track association             |
| `delta_t`       | 3       | Observation window (frames) used to compute velocity         |
| `inertia`       | 0.2     | Weight applied to the direction-consistency cost bonus (OCM) |

## Tuning Tips

- Increase `max_age` (e.g. 60) when objects undergo long occlusions.
- Increase `delta_t` for smoother velocity at the cost of responsiveness to rapid direction changes.
- Increase `inertia` (max 1.0) if objects move at near-constant velocity; lower it for erratic motion.
- `min_hits=1` gives immediate track output — useful when detections are already filtered upstream.

## References

> **Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking**
> Jinkun Cao, Xinshuo Weng, Rawal Khirodkar, Jiangmiao Pang, Kris Kitani
> CVPR 2023
> [arXiv:2203.14360](https://arxiv.org/abs/2203.14360)
