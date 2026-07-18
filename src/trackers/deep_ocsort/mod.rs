#![doc = include_str!("README.md")]

mod tracker;

#[cfg(feature = "python")]
pub mod python;

/// A single tracked object managed by Deep OC-SORT.
///
/// Alias of the shared observation-centric [`ObsTrack`], which carries the Kalman
/// state, observation history (OCM/ORU), and the appearance-embedding buffer Deep
/// OC-SORT flushes into its feature gallery.
///
/// [`ObsTrack`]: crate::trackers::common::ObsTrack
pub use crate::trackers::common::ObsTrack as DeepOcSortTrack;
pub use tracker::DeepOcSortTracker;

use crate::trackers::common::CameraMotion;
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
        self.run(image, detections, &CameraMotion::identity())
    }

    /// Update the tracker, first warping track predictions by `camera_motion`.
    ///
    /// Use this on moving-camera footage; estimate the affine transform between the
    /// previous and current frame (for example with image registration) and pass it
    /// in. See [`CameraMotion`].
    pub fn update_with_camera_motion(
        &mut self,
        image: &DynamicImage,
        detections: Vec<(BoundingBox, f32, i64)>,
        camera_motion: &CameraMotion,
    ) -> Result<Vec<DeepOcSortTrack>, Box<dyn Error>> {
        self.run(image, detections, camera_motion)
    }

    fn run(
        &mut self,
        image: &DynamicImage,
        detections: Vec<(BoundingBox, f32, i64)>,
        camera_motion: &CameraMotion,
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

        Ok(self
            .tracker
            .update_with_camera_motion(&det_tuples, &embeddings, camera_motion))
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

    struct FailingExtractor;
    impl AppearanceExtractor for FailingExtractor {
        fn extract(
            &mut self,
            _image: &DynamicImage,
            _bboxes: &[BoundingBox],
        ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
            Err("extraction failed".into())
        }
    }

    #[test]
    fn wrapper_propagates_extractor_error() {
        let mut tracker = DeepOcSort::new_default(FailingExtractor);
        let image = DynamicImage::new_rgb8(200, 200);
        let dets = vec![(BoundingBox::new(10.0, 10.0, 20.0, 40.0), 0.9, 0)];
        assert!(tracker.update(&image, dets).is_err());
    }

    #[test]
    fn extractor_wrapper_confirms_track() {
        let mut tracker = DeepOcSort::new(MockExtractor, 30, 1, 0.3, 3, 0.2, 0.5, 0.2, 100);
        let image = DynamicImage::new_rgb8(200, 200);
        let dets = vec![(BoundingBox::new(10.0, 10.0, 20.0, 40.0), 0.9, 0)];
        let tracks = tracker.update(&image, dets).unwrap();
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn new_default_runs_through_wrapper() {
        let mut tracker = DeepOcSort::new_default(MockExtractor);
        let image = DynamicImage::new_rgb8(200, 200);
        let dets = vec![(BoundingBox::new(10.0, 10.0, 20.0, 40.0), 0.9, 0)];
        // n_init default is 3, so confirmation needs three matched frames.
        for _ in 0..2 {
            assert!(tracker.update(&image, dets.clone()).unwrap().is_empty());
        }
        assert_eq!(tracker.update(&image, dets).unwrap().len(), 1);
    }

    #[test]
    fn wrapper_applies_camera_motion() {
        let mut tracker = DeepOcSort::new(MockExtractor, 30, 1, 0.3, 3, 0.2, 0.5, 0.2, 100);
        let image = DynamicImage::new_rgb8(800, 400);
        let id = tracker
            .update(
                &image,
                vec![(BoundingBox::new(100.0, 100.0, 50.0, 100.0), 0.9, 0)],
            )
            .unwrap()[0]
            .track_id;
        let cmc = CameraMotion::new(1.0, 0.0, 200.0, 0.0, 1.0, 0.0);
        let tracks = tracker
            .update_with_camera_motion(
                &image,
                vec![(BoundingBox::new(300.0, 100.0, 50.0, 100.0), 0.9, 0)],
                &cmc,
            )
            .unwrap();
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn wrapper_handles_empty_detections() {
        let mut tracker = DeepOcSort::new_default(MockExtractor);
        let image = DynamicImage::new_rgb8(200, 200);
        assert!(tracker.update(&image, vec![]).unwrap().is_empty());
    }

    #[test]
    fn round2_rematch_and_oru_after_gap() {
        // Build a strong rightward velocity with overlapping steps, drop two frames
        // so the Kalman prediction overshoots well past the object, then re-detect at
        // the last observed position. Round 1 misses; the round-2 pass on the last
        // observation recovers the track and runs the ORU re-update.
        // Large boxes and steps so the damped Kalman velocity still drifts the
        // prediction off the object during the gap.
        let mut tracker = build_tracker(20, 1, 0.3, 3, 0.2, 0.0, 0.2, 100);
        for step in 0..8 {
            let x = step as f32 * 200.0;
            tracker.update(&[det(x, 0.0, 400.0, 100.0, 0.9)], &[]);
        }
        for _ in 0..4 {
            tracker.update(&[], &[]);
        }
        let tracks = tracker.update(&[det(1400.0, 0.0, 400.0, 100.0, 0.9)], &[]);
        assert!(!tracks.is_empty());
        assert_eq!(tracks[0].track_id, 1);
    }

    #[test]
    fn oru_runs_on_rematch_after_gap() {
        // A track misses one frame and is re-detected at the same position. Round 1
        // matches, and because it was lost the ORU re-update runs.
        let mut tracker = build_tracker(10, 1, 0.3, 3, 0.2, 0.0, 0.2, 100);
        let id = tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        tracker.update(&[], &[]);
        let tracks = tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn track_deleted_after_max_age() {
        let mut tracker = build_tracker(2, 1, 0.3, 3, 0.2, 0.0, 0.2, 100);
        tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        for _ in 0..4 {
            tracker.update(&[], &[]);
        }
        assert!(tracker.update(&[], &[]).is_empty());
    }

    #[test]
    fn unmatched_tentative_is_deleted() {
        // min_hits = 2 keeps the first track tentative; a far detection on the next
        // frame leaves it unmatched, so it is dropped and a new track starts.
        let mut tracker = build_tracker(30, 2, 0.3, 3, 0.2, 0.0, 0.2, 100);
        tracker.update(&[det(0.0, 0.0, 50.0, 100.0, 0.9)], &[]);
        tracker.update(&[det(500.0, 500.0, 50.0, 100.0, 0.9)], &[]);
        // Track 1 (at the origin) was deleted; the new track confirms here.
        let tracks = tracker.update(&[det(500.0, 500.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 2);
    }

    #[test]
    fn camera_motion_warps_prediction_for_matching() {
        // Establish a track, then pan the camera right by 200px so the same object
        // appears at x=300. The warp moves the prediction with it, keeping the id.
        let mut tracker = build_tracker(30, 1, 0.3, 3, 0.2, 0.0, 0.2, 100);
        let id = tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        let cmc = CameraMotion::new(1.0, 0.0, 200.0, 0.0, 1.0, 0.0);
        let tracks =
            tracker.update_with_camera_motion(&[det(300.0, 100.0, 50.0, 100.0, 0.9)], &[], &cmc);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn without_camera_motion_a_large_shift_starts_a_new_track() {
        let mut tracker = build_tracker(30, 1, 0.3, 3, 0.2, 0.0, 0.2, 100);
        tracker.update(&[det(100.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        let tracks = tracker.update(&[det(300.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks[0].track_id, 2);
    }

    #[test]
    fn appearance_gated_when_dissimilar() {
        // A tiny max_cosine_distance gates out a dissimilar embedding, so the match
        // falls back to motion only and the track id persists.
        let mut tracker = build_tracker(30, 1, 0.3, 3, 0.2, 0.5, 0.05, 100);
        let id = tracker.update(
            &[det(100.0, 100.0, 50.0, 100.0, 0.9)],
            &[vec![1.0, 0.0, 0.0]],
        )[0]
        .track_id;
        let tracks = tracker.update(
            &[det(103.0, 100.0, 50.0, 100.0, 0.9)],
            &[vec![0.0, 1.0, 0.0]],
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }
}
