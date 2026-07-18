# Trackers

All trackers share the same Kalman filter and the same detection input, and differ only in how
they associate detections to existing tracks. Pick based on your scene and whether you have a Re-ID
model.

| Tracker                          | Appearance       | Matching                    | When to use                                                 |
| -------------------------------- | ---------------- | --------------------------- | ----------------------------------------------------------- |
| [SORT](./sort.md)                | None             | IoU                         | Simple scenes, highest speed, no occlusions                 |
| [ByteTrack](./byte_track.md)     | None             | IoU (two-stage)             | Crowded scenes, low-confidence detections, short occlusions |
| [OC-SORT](./ocsort.md)           | None             | IoU + velocity (OCM)        | Frequent brief occlusions, no Re-ID available               |
| [DeepSORT](./deepsort.md)        | Re-ID embeddings | Appearance + IoU            | Long occlusions, dense crowds, identity-sensitive cases     |
| [Deep OC-SORT](./deep_ocsort.md) | Re-ID embeddings | IoU + velocity + appearance | Occlusions where Re-ID helps recover identities             |
| [BoT-SORT](./botsort.md)         | Re-ID embeddings | IoU + appearance + camera motion | Moving cameras, panning and zoom, optional Re-ID        |
| [TrackTrack](./tracktrack.md)    | Re-ID embeddings | Track-perspective association | Crowded scenes needing strong identity, optional Re-ID     |

The Kalman filter uses an 8-dimensional state `[x, y, a, h, vx, vy, va, vh]`, where `(x, y)` is the
box centre, `a` is the aspect ratio, and `h` is the height. Detections in `[x, y, w, h]` (top-left)
form are converted to and from this representation internally.
