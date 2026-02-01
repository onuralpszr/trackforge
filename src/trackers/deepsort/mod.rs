//! DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) implementation.
//!
//! This module provides a DeepSORT tracker that uses appearance features for more robust tracking.

mod nn_matching;
mod track;
mod tracker;

#[cfg(feature = "python")]
pub mod python;

pub use nn_matching::Metric;
pub use track::{Track, TrackState};
pub use tracker::DeepSortTracker;

use crate::traits::AppearanceExtractor;
use crate::types::BoundingBox;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::error::Error;
use image::DynamicImage;
use nn_matching::NearestNeighborDistanceMetric;

/// Deep SORT tracker implementation.
///
/// Wraps the tracker logic and appearance feature extraction.
pub struct DeepSort<E: AppearanceExtractor> {
    extractor: E,
    tracker: DeepSortTracker,
}

impl<E: AppearanceExtractor> DeepSort<E> {
    /// Create a new Deep SORT tracker.
    ///
    /// # Arguments
    /// * `extractor` - The appearance feature extractor.
    /// * `max_age` - Maximum frames to keep a track without detection. Default: 70.
    /// * `n_init` - Minimum hits to confirm a track. Default: 3.
    /// * `max_iou_distance` - Threshold for IoU matching. Default: 0.7.
    /// * `max_cosine_distance` - Threshold for cosine distance matching. Default: 0.2.
    /// * `nn_budget` - Maximum library size for appearance features. Default: 100.
    pub fn new(
        extractor: E,
        max_age: usize,
        n_init: usize,
        max_iou_distance: f32,
        max_cosine_distance: f32,
        nn_budget: usize,
    ) -> Self {
        let metric = NearestNeighborDistanceMetric::new(
            Metric::Cosine,
            max_cosine_distance,
            Some(nn_budget),
        );
        let tracker = DeepSortTracker::new(metric, max_age, n_init, max_iou_distance);

        Self { extractor, tracker }
    }

    /// Update the tracker with new frame and detections.
    ///
    /// # Arguments
    /// * `image` - The current video frame.
    /// * `detections` - List of (BoundingBox, Score, ClassID).
    ///
    /// # Returns
    /// List of confirmed tracks.
    pub fn update(
        &mut self,
        image: &DynamicImage,
        detections: Vec<(BoundingBox, f32, i64)>,
    ) -> Result<Vec<Track>, Box<dyn Error>> {
        // 1. Predict
        self.tracker.predict();

        // 2. Extract features
        let bboxes: Vec<BoundingBox> = detections.iter().map(|(bbox, _, _)| *bbox).collect();
        // If detections are empty, we still run predict (above) and update (to mark missed).
        // Extraction might fail on empty bbox list or image issues?
        let embeddings = if !bboxes.is_empty() {
            self.extractor.extract(image, &bboxes)?
        } else {
            Vec::new()
        };

        // 3. Update
        self.tracker.update(&detections, &embeddings);

        // 4. Return confirmed current tracks
        Ok(self
            .tracker
            .tracks
            .iter()
            .filter(|t: &&Track| t.is_confirmed() && t.time_since_update == 0) // Only active matched tracks? Or all active confirmed? Sort returns all confirmed active in recent update.
            // Usually we return tracks present in current frame.
            // sort.rs: filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .cloned()
            .collect())
    }
}

// Default convenience constructor
impl<E: AppearanceExtractor> DeepSort<E> {
    pub fn new_default(extractor: E) -> Self {
        Self::new(extractor, 70, 3, 0.7, 0.2, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BoundingBox;

    struct MockExtractor;
    impl AppearanceExtractor for MockExtractor {
        fn extract(
            &mut self,
            _image: &DynamicImage,
            bboxes: &[BoundingBox],
        ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
            // Return constant feature
            Ok(vec![vec![1.0, 0.0]; bboxes.len()])
        }
    }

    #[test]
    fn test_deepsort_initialization() {
        let tracker = DeepSort::new_default(MockExtractor);
        assert_eq!(tracker.tracker.tracks.len(), 0);
    }

    #[test]
    fn test_deepsort_track_lifecycle() {
        let mut tracker = DeepSort::new_default(MockExtractor);
        let image = DynamicImage::new_rgb8(100, 100);

        // Frame 1: Detection
        let det1 = vec![(BoundingBox::new(10.0, 10.0, 20.0, 20.0), 0.9, 0)];
        let tracks = tracker.update(&image, det1.clone()).unwrap();

        // Should be empty (tentative)
        assert!(tracks.is_empty());
        assert_eq!(tracker.tracker.tracks.len(), 1);
        assert!(!tracker.tracker.tracks[0].is_confirmed());

        // Frame 2: Match
        let tracks = tracker.update(&image, det1.clone()).unwrap();
        // Still tentative (hits=2) if n_init=3
        assert!(tracks.is_empty());

        // Frame 3: Match -> Confirmed
        let tracks = tracker.update(&image, det1).unwrap();
        assert_eq!(tracks.len(), 1);
        assert!(tracks[0].is_confirmed());
    }
}
