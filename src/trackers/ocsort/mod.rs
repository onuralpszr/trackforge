#![doc = include_str!("README.md")]

use crate::trackers::common::association::{last_observation_rematch, ocm_angle_bonus};
use crate::trackers::common::{ObsTrack, TrackState};
use crate::utils::assignment::greedy_match;
use crate::utils::geometry::{iou_batch, tlwh_to_xyah};
use std::collections::HashSet;

/// Track lifecycle state for OC-SORT.
///
/// Alias of the shared [`TrackState`].
pub type OcSortTrackState = TrackState;

/// A single tracked object managed by OC-SORT.
///
/// Alias of the shared observation-centric [`ObsTrack`], which carries the Kalman
/// state and observation history that OC-SORT's velocity (OCM) and re-update (ORU)
/// logic operate on.
pub type OcSortTrack = ObsTrack;

/// Internal detection wrapper.
#[derive(Debug, Clone)]
struct Detection {
    tlwh: [f32; 4],
    score: f32,
    class_id: i64,
}

/// OC-SORT tracker.
///
/// Extends SORT with observation-centric velocity, momentum-adjusted prediction,
/// and Kalman re-update on re-association. The public API is identical to [`Sort`]:
/// call [`OcSort::update`] once per frame with a list of detections.
///
/// ## Example
///
/// ```rust
/// use trackforge::trackers::ocsort::OcSort;
///
/// // max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2
/// let mut tracker = OcSort::new(30, 3, 0.3, 3, 0.2);
///
/// let detections = vec![
///     ([100.0_f32, 100.0, 50.0, 100.0], 0.9_f32, 0_i64),
///     ([200.0_f32, 200.0, 60.0, 120.0], 0.85_f32, 0_i64),
/// ];
///
/// let tracks = tracker.update(detections);
/// for t in &tracks {
///     println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
/// }
/// ```
///
/// [`Sort`]: crate::trackers::sort::Sort
pub struct OcSort {
    tracks: Vec<OcSortTrack>,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    /// Observation window size for velocity computation.
    delta_t: usize,
    /// Momentum weight applied to the direction-consistency cost bonus.
    inertia: f32,
    kf: crate::utils::kalman::KalmanFilter,
    next_id: u64,
    frame_count: usize,
}

impl OcSort {
    /// Create a new OC-SORT tracker.
    ///
    /// # Arguments
    ///
    /// * `max_age` - Frames a lost track survives before deletion (default: 30).
    /// * `min_hits` - Consecutive matches required to confirm a track (default: 3).
    /// * `iou_threshold` - Minimum IoU to associate a detection with a track (default: 0.3).
    /// * `delta_t` - Observation window for velocity computation (default: 3).
    /// * `inertia` - Weight for the OCM direction-consistency bonus in `[0, 1]` (default: 0.2).
    pub fn new(
        max_age: usize,
        min_hits: usize,
        iou_threshold: f32,
        delta_t: usize,
        inertia: f32,
    ) -> Self {
        Self {
            tracks: Vec::new(),
            max_age,
            min_hits,
            iou_threshold,
            delta_t,
            inertia: inertia.clamp(0.0, 1.0),
            kf: crate::utils::kalman::KalmanFilter::default(),
            next_id: 1,
            frame_count: 0,
        }
    }

    /// Update the tracker with detections for the current frame.
    ///
    /// # Arguments
    ///
    /// * `detections` - `(tlwh, score, class_id)` tuples for all detections in the frame.
    ///
    /// # Returns
    ///
    /// Confirmed tracks active in this frame.
    pub fn update(&mut self, detections: Vec<([f32; 4], f32, i64)>) -> Vec<OcSortTrack> {
        self.frame_count += 1;

        let detections: Vec<Detection> = detections
            .into_iter()
            .map(|(tlwh, score, class_id)| Detection {
                tlwh,
                score,
                class_id,
            })
            .collect();

        // 1. Kalman predict all existing tracks.
        for track in &mut self.tracks {
            track.predict(&self.kf);
        }

        // 2. Match detections to tracks (OCM direction-consistency bonus + round-2 re-match).
        let (matches, unmatched_dets, unmatched_trks) = self.associate(&detections);

        // 3. Update matched tracks.
        for (det_idx, trk_idx) in &matches {
            let det = &detections[*det_idx];
            let xyah = tlwh_to_xyah(&det.tlwh);
            let track = &mut self.tracks[*trk_idx];

            // ORU: correct KF drift if the track was re-found after a gap.
            if track.time_since_update > 0 {
                track.our_re_update(&xyah, self.frame_count, &self.kf);
            }
            track.update_kf(&xyah, &self.kf);
            track.push_observation(xyah, self.frame_count, self.delta_t + 1);
            track.record_match(det.tlwh, det.score, det.class_id);
        }

        // 4. Initialise new tracks for unmatched detections.
        for det_idx in unmatched_dets {
            let det = &detections[det_idx];
            let track = OcSortTrack::new(
                det.tlwh,
                det.score,
                det.class_id,
                self.next_id,
                self.frame_count,
                None,
                &self.kf,
            );
            self.next_id += 1;
            self.tracks.push(track);
        }

        // 5. Update states; delete stale tracks.
        for track in &mut self.tracks {
            // Confirm using hit_streak (consecutive hits), matching the reference behaviour.
            if track.time_since_update == 0 && track.hit_streak >= self.min_hits {
                track.state = TrackState::Confirmed;
            }
            if track.time_since_update > self.max_age {
                track.mark_deleted();
            }
        }

        // Delete unmatched tentative tracks immediately.
        let unmatched_trks_set: HashSet<usize> = unmatched_trks.into_iter().collect();
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if unmatched_trks_set.contains(&i) && track.state == TrackState::Tentative {
                track.mark_deleted();
            }
        }

        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // 6. Return confirmed tracks matched this frame.
        self.tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .cloned()
            .collect()
    }

    /// Associate detections with tracks.
    ///
    /// Round 1 is IoU on the KF-predicted boxes plus an OCM direction-consistency
    /// bonus; round 2 rematches leftovers on each track's last observed position.
    /// Returns matches as `(detection, track)` pairs plus the unmatched detections
    /// and tracks.
    fn associate(&self, detections: &[Detection]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        let n_trks = self.tracks.len();
        let n_dets = detections.len();

        if n_trks == 0 {
            return (Vec::new(), (0..n_dets).collect(), Vec::new());
        }
        if n_dets == 0 {
            return (Vec::new(), Vec::new(), (0..n_trks).collect());
        }

        let pred_boxes: Vec<[f32; 4]> = self.tracks.iter().map(|t| t.tlwh).collect();
        let det_boxes: Vec<[f32; 4]> = detections.iter().map(|d| d.tlwh).collect();
        let det_scores: Vec<f32> = detections.iter().map(|d| d.score).collect();

        let ious = iou_batch(&pred_boxes, &det_boxes);
        let angle_diff = ocm_angle_bonus(
            &self.tracks,
            &det_boxes,
            &det_scores,
            self.delta_t,
            self.inertia,
        );

        // Round 1: IoU plus the OCM bonus.
        let cost_matrix: Vec<Vec<f32>> = (0..n_trks)
            .map(|i| {
                (0..n_dets)
                    .map(|j| 1.0 - (ious[i][j] + angle_diff[i][j]))
                    .collect()
            })
            .collect();

        let (matches_raw, mut unmatched_trks, mut unmatched_dets) =
            greedy_match(&cost_matrix, 1.0 - self.iou_threshold);
        let mut matches: Vec<(usize, usize)> = matches_raw
            .into_iter()
            .map(|(trk, det)| (det, trk))
            .collect();

        // Round 2: rematch leftovers on last observed positions.
        last_observation_rematch(
            &self.tracks,
            &det_boxes,
            &mut matches,
            &mut unmatched_dets,
            &mut unmatched_trks,
            self.iou_threshold,
        );

        (matches, unmatched_dets, unmatched_trks)
    }
}

impl Default for OcSort {
    fn default() -> Self {
        Self::new(30, 3, 0.3, 3, 0.2)
    }
}

impl crate::traits::Tracker for OcSort {
    type Track = OcSortTrack;

    fn update(&mut self, detections: Vec<crate::traits::Detection>) -> Vec<OcSortTrack> {
        self.update(detections)
    }
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed OC-SORT tracker.
#[cfg(feature = "python")]
#[pyclass(name = "OCSORT")]
pub struct PyOcSort {
    inner: OcSort,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOcSort {
    #[new]
    #[pyo3(signature = (max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2))]
    /// Initialize the OC-SORT tracker.
    ///
    /// Args:
    ///     max_age (int): Frames a lost track is kept alive. Default: 30.
    ///     min_hits (int): Consecutive matches to confirm a track. Default: 3.
    ///     iou_threshold (float): IoU threshold for matching. Default: 0.3.
    ///     delta_t (int): Observation window for velocity computation. Default: 3.
    ///     inertia (float): Velocity direction-consistency weight in [0, 1]. Default: 0.2.
    fn new(
        max_age: usize,
        min_hits: usize,
        iou_threshold: f32,
        delta_t: usize,
        inertia: f32,
    ) -> Self {
        Self {
            inner: OcSort::new(max_age, min_hits, iou_threshold, delta_t, inertia),
        }
    }

    /// Update the tracker with detections from the current frame.
    ///
    /// Args:
    ///     detections (list): List of ([x, y, w, h], score, class_id) tuples.
    ///
    /// Returns:
    ///     list: Confirmed tracks as (track_id, [x, y, w, h], score, class_id) tuples.
    fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
    ) -> PyResult<Vec<crate::trackers::common::PyTrackingResult>> {
        let tracks = self.inner.update(detections);
        Ok(tracks
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, t.score, t.class_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::kalman::KalmanFilter;

    fn det(x: f32, y: f32, w: f32, h: f32, s: f32) -> ([f32; 4], f32, i64) {
        ([x, y, w, h], s, 0)
    }

    #[test]
    fn test_ocsort_empty_detections() {
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);
        let tracks = tracker.update(vec![]);
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_ocsort_single_detection_confirmed_after_min_hits() {
        let mut tracker = OcSort::new(30, 3, 0.3, 3, 0.2);

        for _ in 0..3 {
            tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        }

        let tracks = tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 1);
    }

    #[test]
    fn test_ocsort_min_hits_one() {
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);
        let tracks = tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_ocsort_track_deleted_after_max_age() {
        let mut tracker = OcSort::new(2, 1, 0.3, 3, 0.2);

        tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);

        // Stop providing the detection; track should age out after max_age frames.
        for _ in 0..3 {
            tracker.update(vec![]);
        }

        let tracks = tracker.update(vec![]);
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_ocsort_tracker_trait() {
        use crate::traits::Tracker;
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);
        let tracks = Tracker::update(&mut tracker, vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_ocsort_two_objects_separate_ids() {
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);

        let tracks = tracker.update(vec![
            det(100.0, 100.0, 50.0, 100.0, 0.9),
            det(400.0, 400.0, 50.0, 100.0, 0.85),
        ]);

        assert_eq!(tracks.len(), 2);
        let ids: HashSet<u64> = tracks.iter().map(|t| t.track_id).collect();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_ocsort_ids_sequential() {
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);

        tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        let tracks = tracker.update(vec![
            det(100.0, 100.0, 50.0, 100.0, 0.9),
            det(400.0, 400.0, 50.0, 100.0, 0.85),
        ]);

        let mut ids: Vec<u64> = tracks.iter().map(|t| t.track_id).collect();
        ids.sort();
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_ocsort_ocv_velocity_computed() {
        let kf = KalmanFilter::default();
        let tlwh = [100.0_f32, 100.0, 50.0, 100.0];
        let mut track = OcSortTrack::new(tlwh, 0.9, 0, 1, 1, None, &kf);

        // Single observation: no direction yet.
        assert!(track.obs_direction(3).is_none());

        // Add a second observation: object moved 10px to the right.
        let tlwh2 = [110.0_f32, 100.0, 50.0, 100.0];
        track.push_observation(tlwh_to_xyah(&tlwh2), 2, 4);

        let dir = track.obs_direction(3).unwrap();
        assert!(dir[0].abs() < 0.01);
        assert!((dir[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ocsort_our_does_not_panic_on_re_association() {
        let kf = KalmanFilter::default();
        let tlwh = [100.0_f32, 100.0, 50.0, 100.0];
        let mut track = OcSortTrack::new(tlwh, 0.9, 0, 1, 1, None, &kf);

        for _ in 0..5 {
            track.predict(&kf);
        }

        track.our_re_update(&tlwh_to_xyah(&[130.0_f32, 100.0, 50.0, 100.0]), 7, &kf);
        assert!(track.tlwh.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ocsort_default_params() {
        let tracker = OcSort::default();
        assert_eq!(tracker.max_age, 30);
        assert_eq!(tracker.min_hits, 3);
        assert!((tracker.iou_threshold - 0.3).abs() < 1e-5);
        assert_eq!(tracker.delta_t, 3);
        assert!((tracker.inertia - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_ocsort_instance_isolation() {
        let mut a = OcSort::new(30, 1, 0.3, 3, 0.2);
        let mut b = OcSort::new(30, 1, 0.3, 3, 0.2);

        a.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        let tracks_b = b.update(vec![det(200.0, 200.0, 50.0, 100.0, 0.9)]);

        assert_eq!(tracks_b[0].track_id, 1);
        assert_eq!(a.frame_count, 1);
        assert_eq!(b.frame_count, 1);
    }

    #[test]
    fn test_ocsort_hit_streak_resets_on_miss() {
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);

        tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        tracker.update(vec![]);

        let tracks = tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].hit_streak, 1);
    }

    #[test]
    fn test_ocsort_second_round_rematches_after_gap() {
        // A track goes missing for one frame, then reappears at the same position.
        // Round-2 matching (using last observed position) should recover it.
        let mut tracker = OcSort::new(5, 1, 0.3, 3, 0.2);

        for _ in 0..2 {
            tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        }

        tracker.update(vec![]);

        let tracks = tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1, "Track should be re-matched after gap");
        assert_eq!(tracks[0].track_id, 1, "Should be the same track");
    }

    #[test]
    fn test_ocsort_ocm_direction_bonus_path() {
        // Two consecutive matches give the track a direction, so associate() enters
        // the OCM angle-bonus path on the third frame.
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);
        tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        let tracks = tracker.update(vec![det(105.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_ocsort_round2_observations_last_reached() {
        // Build a track with 2 observations, then supply a distant detection so
        // round 1 leaves it unmatched and the round-2 last-observation path runs.
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);
        tracker.update(vec![det(0.0, 0.0, 50.0, 100.0, 0.9)]);
        tracker.update(vec![det(0.0, 0.0, 50.0, 100.0, 0.9)]);
        tracker.update(vec![det(10000.0, 0.0, 50.0, 100.0, 0.9)]);
    }

    #[test]
    fn storage_stays_bounded_under_churn() {
        // A fresh, non-matching object each frame. Tracks must be deleted after
        // max_age misses, keeping the internal vector bounded by that window.
        let max_age = 20;
        let mut tracker = OcSort::new(max_age, 3, 0.3, 3, 0.2);
        for f in 0..3000 {
            let x = 5.0 + (f % 100) as f32 * 40.0;
            let _ = tracker.update(vec![det(x, 10.0, 20.0, 40.0, 0.9)]);
            assert!(
                tracker.tracks.len() <= max_age + 5,
                "storage grew to {} at frame {f}",
                tracker.tracks.len()
            );
        }
    }
}
