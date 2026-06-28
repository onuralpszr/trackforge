#![doc = include_str!("README.md")]

mod track;
mod tracker;

#[cfg(feature = "python")]
pub mod python;

pub use track::DeepOcSortTrack;
pub use tracker::DeepOcSortTracker;

use crate::trackers::deepsort::{Metric, NearestNeighborDistanceMetric};
use crate::traits::AppearanceExtractor;
use crate::types::BoundingBox;
use image::DynamicImage;
use std::error::Error;

/// Build the inner Deep OC-SORT tracker shared by the Rust and Python constructors.
#[allow(clippy::too_many_arguments)]
fn build_tracker(
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    delta_t: usize,
    inertia: f32,
    appearance_weight: f32,
    max_cosine_distance: f32,
    nn_budget: usize,
) -> DeepOcSortTracker {
    let metric =
        NearestNeighborDistanceMetric::new(Metric::Cosine, max_cosine_distance, Some(nn_budget));
    DeepOcSortTracker::new(
        max_age,
        min_hits,
        iou_threshold,
        delta_t,
        inertia,
        appearance_weight,
        max_cosine_distance,
        metric,
    )
}

/// Deep OC-SORT tracker.
///
/// Wraps the association core with an [`AppearanceExtractor`] so the caller can pass a
/// frame and detections and have the embeddings produced internally. To pass
/// embeddings directly, drive [`DeepOcSortTracker`] instead.
pub struct DeepOcSort<E: AppearanceExtractor> {
    extractor: E,
    tracker: DeepOcSortTracker,
}

impl<E: AppearanceExtractor> DeepOcSort<E> {
    /// Create a new Deep OC-SORT tracker.
    ///
    /// # Arguments
    /// * `extractor` - The appearance feature extractor.
    /// * `max_age` - Frames a lost track survives before deletion. Default: 30.
    /// * `min_hits` - Consecutive matches required to confirm a track. Default: 3.
    /// * `iou_threshold` - Minimum IoU to associate a detection with a track. Default: 0.3.
    /// * `delta_t` - Observation window for velocity computation. Default: 3.
    /// * `inertia` - Weight for the OCM direction-consistency bonus. Default: 0.2.
    /// * `appearance_weight` - Blend weight of the appearance cost. Default: 0.5.
    /// * `max_cosine_distance` - Gate for the appearance term. Default: 0.2.
    /// * `nn_budget` - Maximum appearance features stored per track. Default: 100.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        extractor: E,
        max_age: usize,
        min_hits: usize,
        iou_threshold: f32,
        delta_t: usize,
        inertia: f32,
        appearance_weight: f32,
        max_cosine_distance: f32,
        nn_budget: usize,
    ) -> Self {
        let tracker = build_tracker(
            max_age,
            min_hits,
            iou_threshold,
            delta_t,
            inertia,
            appearance_weight,
            max_cosine_distance,
            nn_budget,
        );
        Self { extractor, tracker }
    }

    /// Create a tracker with the default parameters.
    pub fn new_default(extractor: E) -> Self {
        Self::new(extractor, 30, 3, 0.3, 3, 0.2, 0.5, 0.2, 100)
    }

    /// Update the tracker with a frame and its detections.
    ///
    /// Extracts an appearance embedding per detection, then runs the association.
    /// Returns the confirmed tracks matched in this frame.
    pub fn update(
        &mut self,
        image: &DynamicImage,
        detections: Vec<(BoundingBox, f32, i64)>,
    ) -> Result<Vec<DeepOcSortTrack>, Box<dyn Error>> {
        let bboxes: Vec<BoundingBox> = detections.iter().map(|(bbox, _, _)| *bbox).collect();
        let embeddings = if bboxes.is_empty() {
            Vec::new()
        } else {
            self.extractor.extract(image, &bboxes)?
        };

        let det_tuples: Vec<([f32; 4], f32, i64)> = detections
            .iter()
            .map(|(b, score, class_id)| ([b.x, b.y, b.width, b.height], *score, *class_id))
            .collect();

        Ok(self.tracker.update(&det_tuples, &embeddings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, s: f32) -> ([f32; 4], f32, i64) {
        ([x, y, w, h], s, 0)
    }

    #[test]
    fn confirms_after_min_hits() {
        let mut tracker = build_tracker(30, 3, 0.3, 3, 0.2, 0.5, 0.2, 100);
        for _ in 0..2 {
            assert!(
                tracker
                    .update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[])
                    .is_empty()
            );
        }
        let tracks = tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 1);
    }

    #[test]
    fn empty_detections_return_no_tracks() {
        let mut tracker = build_tracker(30, 1, 0.3, 3, 0.2, 0.5, 0.2, 100);
        assert!(tracker.update(&[], &[]).is_empty());
    }

    #[test]
    fn zero_weight_matches_without_appearance() {
        // appearance_weight = 0 reduces to OC-SORT; embeddings are still accepted.
        let mut tracker = build_tracker(30, 1, 0.3, 3, 0.2, 0.0, 0.2, 100);
        let emb = vec![vec![1.0, 0.0]];
        let t1 = tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &emb);
        assert_eq!(t1.len(), 1);
        let id = t1[0].track_id;
        let t2 = tracker.update(&[det(103.0, 100.0, 50.0, 100.0, 0.9)], &emb);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].track_id, id);
    }

    #[test]
    fn appearance_keeps_track_id_across_frames() {
        let mut tracker = build_tracker(30, 1, 0.3, 3, 0.2, 0.5, 0.3, 100);
        let emb = vec![vec![1.0, 0.0, 0.0]];
        let t1 = tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &emb);
        let id = t1[0].track_id;
        let t2 = tracker.update(&[det(104.0, 100.0, 50.0, 100.0, 0.9)], &emb);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].track_id, id);
    }

    struct MockExtractor;
    impl AppearanceExtractor for MockExtractor {
        fn extract(
            &mut self,
            _image: &DynamicImage,
            bboxes: &[BoundingBox],
        ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
            Ok(vec![vec![1.0, 0.0]; bboxes.len()])
        }
    }

    #[test]
    fn extractor_wrapper_confirms_track() {
        let mut tracker = DeepOcSort::new(MockExtractor, 30, 1, 0.3, 3, 0.2, 0.5, 0.2, 100);
        let image = DynamicImage::new_rgb8(200, 200);
        let dets = vec![(BoundingBox::new(10.0, 10.0, 20.0, 40.0), 0.9, 0)];
        let tracks = tracker.update(&image, dets).unwrap();
        assert_eq!(tracks.len(), 1);
    }
}
