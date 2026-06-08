# Introduction

Trackforge is a unified, high-performance multi-object tracking library written in Rust and
exposed to Python via PyO3. It implements four production-ready tracking algorithms on top of a
shared Kalman filter, so you can swap trackers without changing your integration code.

It is designed as the CPU "glue" between a GPU object detector and your application: you pass in
detection boxes each frame and get back stable track identities.

## Trackers at a glance

| Tracker   | Appearance       | Matching             | Best for                                      |
| --------- | ---------------- | -------------------- | --------------------------------------------- |
| SORT      | None             | IoU                  | Simple scenes, maximum speed                  |
| ByteTrack | None             | IoU (two-stage)      | Crowded scenes, low-confidence detections     |
| OC-SORT   | None             | IoU + velocity (OCM) | Frequent brief occlusions, no Re-ID available |
| DeepSORT  | Re-ID embeddings | Appearance + IoU     | Long occlusions, identity-sensitive use cases |

Every tracker accepts the same detection tuple, `([x, y, w, h], score, class_id)`, and ships for
both Python and Rust.

This book is a narrative guide. For the full API surface, see the
[Rust API on docs.rs](https://docs.rs/trackforge) and the
[Python API reference](https://onuralpszr.github.io/trackforge/reference/python.html).
