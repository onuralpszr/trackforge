# BoT-SORT

**BoT-SORT** ([arXiv 2206.14651](https://arxiv.org/abs/2206.14651)). Extends ByteTrack's two-stage
cascade with two additions:

- **Camera motion compensation** warps each track's Kalman prediction by a caller-supplied affine
  transform before association, so tracking survives panning and zooming cameras.
- **Appearance fusion** combines a cosine distance to each track's appearance embedding with the IoU
  distance in the high-confidence stage. Appearance is used only when it is confident (below
  `appearance_thresh`) and the pair is spatially close (IoU distance below `proximity_thresh`); the
  fused cost is the smaller of the two. Each track keeps an exponential moving average of its
  embeddings.

With no embeddings the association reduces to ByteTrack with camera motion, so appearance is a strict
add-on. This is a clean-room implementation. CMC is applied from a transform you supply (for example
estimated with OpenCV); the tracker does not estimate camera motion itself. Pass the affine as
`[a, b, tx, c, d, ty]` to `update`.

```python
from trackforge import BOTSORT

tracker = BOTSORT(track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6,
                  proximity_thresh=0.5, appearance_thresh=0.25)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1, 0.2, 0.3]]  # one appearance vector per detection; omit for motion only
tracks = tracker.update(detections, embeddings)
for track_id, tlwh, score, class_id in tracks:
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Parameters

| Parameter           | Default | Description                                             |
| ------------------- | ------- | ------------------------------------------------------- |
| `track_thresh`      | 0.5     | Confidence split between high- and low-score detections |
| `track_buffer`      | 30      | Frames a lost track is kept alive before removal        |
| `match_thresh`      | 0.8     | Maximum cost for a first-stage (high-confidence) match  |
| `det_thresh`        | 0.6     | Minimum score to start a new track                      |
| `proximity_thresh`  | 0.5     | IoU-distance gate above which appearance is ignored     |
| `appearance_thresh` | 0.25    | Cosine-distance gate above which appearance is ignored  |

**Tuning:** supply a camera-motion affine on moving-camera footage; leave it out for a static camera.
Provide embeddings when a Re-ID model is available and identities matter, and tighten
`appearance_thresh` to only trust strong appearance matches. The two-stage thresholds behave as in
ByteTrack.

## Citation

```bibtex
@article{aharon2022botsort,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```
