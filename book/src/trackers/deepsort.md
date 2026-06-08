# DeepSORT

**DeepSORT** ([arXiv 1703.07402](https://arxiv.org/abs/1703.07402)). Extends SORT with a Re-ID
appearance metric. Confirmed tracks are matched first by cosine distance on appearance embeddings
(with Mahalanobis gating), then any remaining tracks fall back to IoU matching. Best for long-term
identity maintenance.

Unlike the other trackers, DeepSORT needs an appearance embedding per detection. In Rust you supply
an `AppearanceExtractor`; in Python you pass embeddings directly.

## Rust: implement an extractor

```rust,ignore
use trackforge::traits::AppearanceExtractor;
use trackforge::types::BoundingBox;
use image::DynamicImage;

struct MyExtractor;

impl AppearanceExtractor for MyExtractor {
    fn extract(
        &mut self,
        image: &DynamicImage,
        boxes: &[BoundingBox],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        // Crop each box, run your Re-ID model, return one embedding per box.
        Ok(boxes.iter().map(|_| vec![0.0_f32; 128]).collect())
    }
}

use trackforge::trackers::deepsort::DeepSort;
let mut tracker = DeepSort::new(MyExtractor, 70, 3, 0.7, 0.2, 100);
```

## Python: bring your own embeddings

```python
from trackforge import DEEPSORT

tracker = DEEPSORT(max_age=70, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100)

detections = [([100.0, 100.0, 50.0, 100.0], 0.9, 0)]
embeddings = [[0.1] * 128]  # one vector per detection, from your Re-ID model

for track_id, tlwh, score, class_id in tracker.update(detections, embeddings):
    print(track_id, tlwh)
```

## Parameters

| Parameter             | Default | Description                                        |
| --------------------- | ------- | -------------------------------------------------- |
| `max_age`             | 70      | Frames a track survives without a match            |
| `n_init`              | 3       | Consecutive detections required to confirm a track |
| `max_iou_distance`    | 0.7     | IoU distance threshold for the fallback IoU stage  |
| `max_cosine_distance` | 0.2     | Cosine distance threshold for appearance matching  |
| `nn_budget`           | 100     | Max appearance embeddings stored per track (FIFO)  |

**Tuning:** lower `max_cosine_distance` (~0.15) for stricter Re-ID; raise `nn_budget` if appearance
drifts over time; lower `n_init` to 1 when detections are reliable and you need tracks immediately.
