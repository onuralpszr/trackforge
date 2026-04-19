# Deep SORT

**Deep SORT** (Simple Online and Realtime Tracking with a Deep Association Metric) integrates appearance information into the tracking pipeline to improve tracking through occlusions and reduce identity switches.

## Features

- **Kalman Filtering**: Tracks object motion using an 8-dimensional state space $(u, v, \gamma, h, \dot{u}, \dot{v}, \dot{\gamma}, \dot{h})$.
- **Appearance Matching**: Uses a feature gallery to match detections based on cosine distance of embeddings.
- **Cascaded Matching**: Prioritizes frequently seen objects.
- **Custom Extractor**: Use any model to generate embeddings via the `AppearanceExtractor` trait.

## Usage

```rust
use trackforge::trackers::deepsort::DeepSort;
use trackforge::traits::AppearanceExtractor;
use trackforge::types::BoundingBox;
use image::DynamicImage;
use std::error::Error;

struct MyExtractor;
impl AppearanceExtractor for MyExtractor {
    fn extract(&self, _img: &DynamicImage, bboxes: &[BoundingBox]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        // Run your ReID model here
        Ok(vec![vec![0.0; 128]; bboxes.len()])
    }
}

let extractor = MyExtractor;
let mut tracker = DeepSort::new_default(extractor);

// In your loop:
// let tracks = tracker.update(&frame, detections)?;
```

## References

- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
