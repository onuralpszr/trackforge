//! [![Crates.io](https://img.shields.io/crates/v/trackforge?logo=rust&logoColor=white&label=crates.io)](https://crates.io/crates/trackforge)
//! [![Downloads](https://img.shields.io/crates/d/trackforge?logo=rust&logoColor=white&label=downloads)](https://crates.io/crates/trackforge)
//! [![docs.rs](https://img.shields.io/docsrs/trackforge?logo=docsdotrs&logoColor=white)](https://docs.rs/trackforge)
//! [![MSRV](https://img.shields.io/crates/msrv/trackforge?logo=rust&logoColor=white)](https://crates.io/crates/trackforge)
//! [![CI](https://img.shields.io/github/actions/workflow/status/onuralpszr/trackforge/CI.yml?branch=main&logo=githubactions&logoColor=white&label=CI)](https://github.com/onuralpszr/trackforge/actions/workflows/CI.yml)
//! [![License: MIT](https://img.shields.io/crates/l/trackforge?logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
//! [![Guide](https://img.shields.io/badge/Guide-mdBook-1F7087?logo=mdbook&logoColor=white)](https://onuralpszr.github.io/trackforge/book/)
//!
//! Trackforge is a unified, high-performance computer vision tracking library implemented in
//! Rust and exposed as a Python package via PyO3.
//!
//! It provides five production-ready multi-object tracking algorithms built on a shared
//! 8-dimensional Kalman filter with state `[x, y, a, h, vx, vy, va, vh]`, where `(x, y)`
//! is the bounding-box centre, `a` the aspect ratio, and `h` the height.
//!
//! # Choose a Tracker
//!
//! | Tracker | Appearance Features | Matching Strategy | Best For |
//! |---------|-------------------|-------------------|----------|
//! | [`sort`] | None | IoU only | Simple scenes, maximum speed |
//! | [`byte_track`] | None | IoU two-stage | Crowded scenes, low-confidence detections |
//! | [`ocsort`] | None | IoU + velocity correction | Scenes with frequent occlusions |
//! | [`deepsort`] | Re-ID embeddings | Appearance + IoU | Long occlusions, dense crowds |
//! | [`deep_ocsort`] | Re-ID embeddings | IoU + velocity + appearance | Occlusions with re-identification |
//! | [`botsort`] | Re-ID embeddings | IoU + appearance + camera motion | Moving cameras, dense crowds |
//!
//! # SORT
//!
//! Simple Online and Realtime Tracking. Pure IoU matching with a Kalman filter.
//! Fastest option with minimal configuration.
//!
//! ```rust
//! use trackforge::trackers::sort::Sort;
//!
//! // max_age=1, min_hits=3, iou_threshold=0.3
//! let mut tracker = Sort::new(1, 3, 0.3);
//!
//! let detections = vec![
//!     ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
//! ];
//!
//! let tracks = tracker.update(detections);
//! for t in &tracks {
//!     println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
//! }
//! ```
//!
//! # ByteTrack
//!
//! Two-stage IoU matching that also associates low-confidence detections, recovering
//! objects that are temporarily occluded or partially out of frame.
//!
//! ```rust
//! use trackforge::trackers::byte_track::ByteTrack;
//!
//! // track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6
//! let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
//!
//! let detections = vec![
//!     ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
//!     ([200.0_f32, 200.0, 60.0, 120.0], 0.85_f32, 0_i64),
//! ];
//!
//! let tracks = tracker.update(detections);
//! for t in &tracks {
//!     println!("ID: {}, Box: {:?}, Score: {:.2}", t.track_id, t.tlwh, t.score);
//! }
//! ```
//!
//! # DeepSORT
//!
//! Augments IoU matching with Re-ID appearance embeddings for reliable re-identification
//! across long occlusions. Requires an [`AppearanceExtractor`] implementation that produces
//! a feature vector for each detected crop.
//!
//! ```rust,ignore
//! use trackforge::trackers::deepsort::DeepSort;
//! use trackforge::traits::AppearanceExtractor;
//! use trackforge::types::BoundingBox;
//! use image::DynamicImage;
//!
//! struct MyExtractor;
//!
//! impl AppearanceExtractor for MyExtractor {
//!     fn extract(
//!         &mut self,
//!         image: &DynamicImage,
//!         boxes: &[BoundingBox],
//!     ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
//!         // Return one embedding vector per bounding box
//!         Ok(boxes.iter().map(|_| vec![0.0_f32; 128]).collect())
//!     }
//! }
//!
//! // max_age=70, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100
//! let mut tracker = DeepSort::new(MyExtractor, 70, 3, 0.7, 0.2, 100);
//!
//! let frame = DynamicImage::new_rgb8(640, 480);
//! let detections = vec![
//!     (BoundingBox { x: 100.0, y: 100.0, width: 50.0, height: 100.0 }, 0.9_f32, 0_i64),
//! ];
//!
//! let tracks = tracker.update(&frame, &detections).unwrap();
//! for t in &tracks {
//!     println!("ID: {}, Box: {:?}", t.track_id, t.to_tlwh());
//! }
//! ```
//!
//! # OC-SORT
//!
//! Observation-Centric SORT. Extends SORT with velocity correction (OCM) and Kalman filter
//! re-update on re-association (ORU), making it robust to brief occlusions without appearance
//! features.
//!
//! ```rust
//! use trackforge::trackers::ocsort::OcSort;
//!
//! // max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2
//! let mut tracker = OcSort::new(30, 3, 0.3, 3, 0.2);
//!
//! let detections = vec![
//!     ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
//! ];
//!
//! let tracks = tracker.update(detections);
//! for t in &tracks {
//!     println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
//! }
//! ```
//!
//! # Deep OC-SORT
//!
//! Extends OC-SORT with an appearance term: a cosine distance to a per-track feature
//! gallery is blended into the motion cost, scaled by detector confidence. Like
//! DeepSORT it needs an [`AppearanceExtractor`], or embeddings passed directly.
//!
//! ```rust,ignore
//! use trackforge::trackers::deep_ocsort::DeepOcSort;
//!
//! // extractor implements AppearanceExtractor (plug in any Re-ID model).
//! // max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2,
//! // appearance_weight=0.5, max_cosine_distance=0.2, nn_budget=100
//! let mut tracker = DeepOcSort::new(extractor, 30, 3, 0.3, 3, 0.2, 0.5, 0.2, 100);
//!
//! let frame = image::DynamicImage::new_rgb8(640, 480);
//! let detections = vec![
//!     (trackforge::types::BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0),
//! ];
//!
//! let tracks = tracker.update(&frame, detections).unwrap();
//! for t in &tracks {
//!     println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
//! }
//! ```
//!
//! [`sort`]: trackers::sort
//! [`byte_track`]: trackers::byte_track
//! [`ocsort`]: trackers::ocsort
//! [`deepsort`]: trackers::deepsort
//! [`deep_ocsort`]: trackers::deep_ocsort
//! [`botsort`]: trackers::botsort
//! [`AppearanceExtractor`]: traits::AppearanceExtractor

pub mod trackers;
pub mod traits;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The Python module for Trackforge.
#[cfg(feature = "python")]
#[pymodule]
fn trackforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<trackers::byte_track::PyByteTrack>()?;
    m.add_class::<trackers::sort::PySort>()?;
    m.add_class::<trackers::deepsort::python::PyDeepSort>()?;
    m.add_class::<trackers::ocsort::PyOcSort>()?;
    m.add_class::<trackers::deep_ocsort::python::PyDeepOcSort>()?;
    m.add_class::<trackers::botsort::PyBotSort>()?;
    Ok(())
}
