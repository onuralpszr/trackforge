# BoT-SORT: Robust associations multi-object tracking

This module implements the BoT-SORT algorithm.

> **BoT-SORT: Robust Associations Multi-Pedestrian Tracking**
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky
> [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)

## Algorithm overview

BoT-SORT builds on ByteTrack's two-stage association and adds two pieces:

- **Camera motion compensation (CMC)** warps each track's Kalman prediction by a
  caller-supplied affine transform before association, so tracking survives panning and
  zooming cameras. The transform is the shared `common::cmc` infrastructure.
- **Appearance fusion** combines a cosine distance to each track's appearance embedding
  with the IoU distance in the high-confidence stage. Appearance is used only when it is
  confident (below `appearance_thresh`) and the pair is spatially close (IoU distance
  below `proximity_thresh`); the fused cost is the smaller of the two. Each track keeps
  an exponential moving average of its embeddings. With no embeddings the association
  reduces to ByteTrack with camera motion.

The two-stage cascade is unchanged from ByteTrack: high-confidence detections are matched
first (on the fused cost), then low-confidence detections recover fragmented tracks on
IoU alone.

This is a clean-room implementation. The tracker applies a camera-motion transform but
does not estimate it: the caller supplies the affine (for example from image
registration), keeping the core free of heavy computer-vision dependencies.

## Builds on

- `utils::kalman` - the shared 8-dimensional Kalman filter
- `utils::geometry` - `iou_batch`, `tlwh_to_xyah`
- `utils::assignment` - `greedy_match`, `iou_match`
- `trackers::common` - `KalmanTrack` and `CameraMotion` (CMC)
- `trackers::byte_track` - the `TrackState` lifecycle shared with ByteTrack

## Parameters

| Parameter           | Default | Description                                             |
| ------------------- | ------- | ------------------------------------------------------- |
| `track_thresh`      | 0.5     | Confidence split between high- and low-score detections |
| `track_buffer`      | 30      | Frames a lost track is kept alive before removal        |
| `match_thresh`      | 0.8     | Maximum cost for a first-stage (high-confidence) match  |
| `det_thresh`        | 0.6     | Minimum score to start a new track                      |
| `proximity_thresh`  | 0.5     | IoU-distance gate above which appearance is ignored     |
| `appearance_thresh` | 0.25    | Cosine-distance gate above which appearance is ignored  |

## Rust API

```rust,ignore
use trackforge::trackers::botsort::BotSort;

let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);

let detections = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
let embeddings = vec![vec![0.1, 0.2, 0.3]]; // one appearance vector per detection
let tracks = tracker.update(detections, &embeddings);
for t in tracks {
    println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
}
```

## Python API

```python
from trackforge import BOTSORT

tracker = BOTSORT(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    det_thresh=0.6,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection; omit for motion only
tracks = tracker.update(detections, embeddings)

# Moving camera: pass a [a, b, tx, c, d, ty] affine mapping the previous frame
# to the current one (estimate it however you like, e.g. with OpenCV).
camera_motion = [1.0, 0.0, 12.0, 0.0, 1.0, -4.0]
tracks = tracker.update(detections, embeddings, camera_motion)

for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Credit

Clean-room Rust implementation of the algorithm described in the paper above. Original
reference implementation: [NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT).

## Citation

```bibtex
@article{aharon2022botsort,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```
