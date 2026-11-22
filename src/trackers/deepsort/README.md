# Deep SORT

**Deep SORT** (Simple Online and Realtime Tracking with a Deep Association Metric) integrates appearance information into the tracking pipeline to improve tracking through occlusions and reduce identity switches.

## Features

- **Kalman Filtering**: Tracks object motion using the shared 8-dimensional state
  `[x, y, a, h, vx, vy, va, vh]`, where `(x, y)` is the box centre, `a` is the aspect
  ratio, and `h` is the height.
- **Appearance Matching**: Uses a feature gallery to match detections based on cosine distance of embeddings.
- **Cascaded Matching**: Prioritizes frequently seen objects.
- **Custom Extractor**: Use any model to generate embeddings via the `AppearanceExtractor` trait.

## Builds on

- `utils::kalman` - the shared 8-dimensional Kalman filter
- `utils::geometry` - `iou_cost_matrix`, `tlwh_to_xyah`, `xyah_to_tlwh`
- `utils::assignment` - `greedy_match`
- `trackers::common` - `CommonParams` and the `TrackState` lifecycle

## Parameters

| Parameter             | Default | Description                                        |
| --------------------- | ------- | -------------------------------------------------- |
| `max_age`             | 70      | Frames a track survives without a match            |
| `n_init`              | 3       | Consecutive detections required to confirm a track |
| `max_iou_distance`    | 0.7     | IoU distance threshold for the fallback IoU stage  |
| `max_cosine_distance` | 0.2     | Cosine distance threshold for appearance matching  |
| `nn_budget`           | 100     | Max appearance embeddings stored per track (FIFO)  |

## Rust API

```rust,ignore
use trackforge::trackers::deepsort::DeepSort;
use trackforge::traits::AppearanceExtractor;
use trackforge::types::BoundingBox;
use image::DynamicImage;
use std::error::Error;

struct MyExtractor;
impl AppearanceExtractor for MyExtractor {
    fn extract(&mut self, _img: &DynamicImage, bboxes: &[BoundingBox]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        // Run your ReID model here
        Ok(vec![vec![0.0; 128]; bboxes.len()])
    }
}

let extractor = MyExtractor;
let mut tracker = DeepSort::new_default(extractor);

// In your loop:
// let tracks = tracker.update(&frame, detections)?;
```

## Python API

```python
from trackforge import DEEPSORT

tracker = DEEPSORT(
    max_age=70,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=100,
)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1] * 128]  # one vector per detection, from your Re-ID model

for track_id, tlwh, score, class_id, det_ind in tracker.update(detections, embeddings):
    print(f"ID: {track_id}, Box: {tlwh}")
```

## Credit

Clean-room Rust implementation of the algorithm described in the paper above. Original
reference implementation: [nwojke/deep_sort](https://github.com/nwojke/deep_sort).

## References

- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

## Citation

```bibtex
@inproceedings{wojke2017deepsort,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2017}
}
```
