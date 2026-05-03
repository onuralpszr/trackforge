//! Trackforge is a unified, high-performance computer vision tracking library, implemented in
//! Rust and exposed as a Python package via PyO3.
//!
//! It provides three state-of-the-art multi-object tracking algorithms — **SORT**, **ByteTrack**,
//! and **DeepSORT** — each optimized for a different trade-off between speed, accuracy, and
//! robustness.
//!
//! # Choosing a Tracker
//!
//! | Tracker | Appearance Features | Matching Strategy | Best For |
//! |---------|-------------------|-------------------|----------|
//! | [`sort`] | None | IoU only | Simple scenes, maximum speed |
//! | [`byte_track`] | None | IoU two-stage | Crowded scenes, low-confidence detections |
//! | [`deepsort`] | Re-ID embeddings | Appearance + IoU | Long occlusions, dense crowds |
//!
//! All three share the same 8-dimensional Kalman filter state `[x, y, a, h, vx, vy, va, vh]`
//! where `(x, y)` is the bounding-box centre, `a` the aspect ratio and `h` the height.
//!
//! # SORT — Simple Online and Realtime Tracking
//!
//! Pure IoU matching with a Kalman filter.  Fastest tracker with minimal configuration.
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
//! Two-stage IoU matching that also associates low-confidence detections, improving recall
//! of partially occluded objects.
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
//! across long occlusions.  Requires an [`AppearanceExtractor`] implementation that produces
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
//!         &self,
//!         image: &DynamicImage,
//!         boxes: &[BoundingBox],
//!     ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
//!         // Return one 128-D embedding per bounding box
//!         Ok(boxes.iter().map(|_| vec![0.0_f32; 128]).collect())
//!     }
//! }
//!
//! // max_age=70, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100
//! let mut tracker = DeepSort::new(MyExtractor, 70, 3, 0.7, 0.2, 100);
//!
//! let frame: DynamicImage = DynamicImage::new_rgb8(640, 480);
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
//! [`sort`]: trackers::sort
//! [`byte_track`]: trackers::byte_track
//! [`deepsort`]: trackers::deepsort
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
    m.add_class::<trackers::deepsort::python::PyDeepSortTrack>()?;
    Ok(())
}
