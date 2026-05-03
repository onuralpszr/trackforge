#![doc = include_str!("README.md")]

use crate::utils::kalman::{CovarianceMatrix, KalmanFilter, MeasurementVector, StateVector};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Track state
// ---------------------------------------------------------------------------

/// Track lifecycle state for OC-SORT.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum OcSortTrackState {
    /// Newly created; not yet confirmed by enough consecutive matches.
    Tentative,
    /// Confirmed active track returned to callers.
    Confirmed,
    /// Marked for removal.
    Deleted,
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// A single tracked object managed by OC-SORT.
#[derive(Debug, Clone)]
pub struct OcSortTrack {
    /// Bounding box in TLWH (top-left x, top-left y, width, height) format.
    pub tlwh: [f32; 4],
    /// Detection confidence of the most recent match.
    pub score: f32,
    /// Class label of the most recent match.
    pub class_id: i64,
    /// Unique monotonically increasing track identifier.
    pub track_id: u64,
    /// Current lifecycle state.
    pub state: OcSortTrackState,
    /// Total number of detection matches over the track lifetime.
    pub hits: usize,
    /// Consecutive detection matches without interruption (resets to 0 on any missed frame).
    pub hit_streak: usize,
    /// Frames elapsed since the last detection match.
    pub time_since_update: usize,
    /// Total frames since track creation.
    pub age: usize,

    // Kalman filter state (xyah format: cx, cy, aspect, height + velocities)
    mean: StateVector,
    covariance: CovarianceMatrix,

    // OC-SORT: circular observation history used for OCV and ORU.
    // Stored as (xyah, frame_id) in insertion order; capped at `delta_t + 1` entries.
    observations: Vec<(MeasurementVector, usize)>,
}

impl OcSortTrack {
    fn new(
        tlwh: [f32; 4],
        score: f32,
        class_id: i64,
        track_id: u64,
        frame_id: usize,
        kf: &KalmanFilter,
    ) -> Self {
        let xyah = tlwh_to_xyah(&tlwh);
        let (mean, covariance) = kf.initiate(&xyah);
        let observations = vec![(xyah, frame_id)];

        Self {
            tlwh,
            score,
            class_id,
            track_id,
            state: OcSortTrackState::Tentative,
            hits: 1,
            hit_streak: 1,
            time_since_update: 0,
            age: 1,
            mean,
            covariance,
            observations,
        }
    }

    /// Kalman-predict one step forward.
    ///
    /// Resets `hit_streak` when the track missed the previous frame, matching the
    /// reference OC-SORT behaviour where consecutive-hit count is used for confirmation.
    fn predict(&mut self, kf: &KalmanFilter) {
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = covariance;
        self.age += 1;
        self.time_since_update += 1;
        self.tlwh = xyah_to_tlwh(&self.mean);
    }

    /// Standard Kalman update with a new matched detection.
    fn update_kf(&mut self, xyah: &MeasurementVector, kf: &KalmanFilter) {
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, xyah);
        self.mean = mean;
        self.covariance = covariance;
        self.tlwh = xyah_to_tlwh(&self.mean);
    }

    /// OCV: compute normalised 2-D velocity direction `[dy, dx]` over the last
    /// `delta_t` frames.
    ///
    /// Returns `None` when fewer than two observations are available.
    /// The direction vector is normalised to unit length (L2 + 1e-6 epsilon).
    fn obs_direction(&self, delta_t: usize) -> Option<[f32; 2]> {
        let n = self.observations.len();
        if n < 2 {
            return None;
        }
        let anchor_idx = n.saturating_sub(delta_t + 1);
        let (obs_old, _) = &self.observations[anchor_idx];
        let (obs_new, _) = &self.observations[n - 1];
        // xyah[0] = cx, xyah[1] = cy
        let dy = obs_new[1] - obs_old[1];
        let dx = obs_new[0] - obs_old[0];
        let norm = (dy * dy + dx * dx).sqrt() + 1e-6;
        Some([dy / norm, dx / norm])
    }

    /// ORU: replay interpolated observations to correct KF drift after re-association.
    ///
    /// Linearly interpolates between the last recorded observation and the current
    /// detection in TLWH space (matching the reference implementation), converts each
    /// virtual observation to xyah, then replays predict → update through the KF so
    /// that future predictions start from a corrected state.
    fn our_re_update(
        &mut self,
        current_xyah: &MeasurementVector,
        current_frame: usize,
        kf: &KalmanFilter,
    ) {
        let n = self.observations.len();
        if n == 0 {
            return;
        }
        let (last_obs, last_frame) = &self.observations[n - 1];
        let gap = (current_frame as isize - *last_frame as isize).max(1) as usize;

        if gap <= 1 {
            return;
        }

        // Interpolate in TLWH space (reference interpolates in (x,y,w,h), not xyah).
        let last_tlwh = xyah4_to_tlwh(last_obs);
        let current_tlwh = xyah4_to_tlwh(current_xyah);

        let (mut mean, mut covariance) = kf.initiate(last_obs);

        for step in 1..=gap {
            let t = step as f32 / gap as f32;
            let virtual_tlwh = [
                last_tlwh[0] + (current_tlwh[0] - last_tlwh[0]) * t,
                last_tlwh[1] + (current_tlwh[1] - last_tlwh[1]) * t,
                last_tlwh[2] + (current_tlwh[2] - last_tlwh[2]) * t,
                last_tlwh[3] + (current_tlwh[3] - last_tlwh[3]) * t,
            ];
            let virtual_xyah = tlwh_to_xyah(&virtual_tlwh);
            let (pm, pc) = kf.predict(&mean, &covariance);
            mean = pm;
            covariance = pc;
            let (um, uc) = kf.update(&mean, &covariance, &virtual_xyah);
            mean = um;
            covariance = uc;
        }

        self.mean = mean;
        self.covariance = covariance;
        self.tlwh = xyah_to_tlwh(&self.mean);
    }

    /// Record a new observation, keeping the history bounded to `max_obs` entries.
    fn push_observation(&mut self, xyah: MeasurementVector, frame_id: usize, max_obs: usize) {
        self.observations.push((xyah, frame_id));
        if self.observations.len() > max_obs {
            self.observations.remove(0);
        }
    }

    fn mark_deleted(&mut self) {
        self.state = OcSortTrackState::Deleted;
    }

    fn is_confirmed(&self) -> bool {
        self.state == OcSortTrackState::Confirmed
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

fn tlwh_to_xyah(tlwh: &[f32; 4]) -> MeasurementVector {
    let cx = tlwh[0] + tlwh[2] / 2.0;
    let cy = tlwh[1] + tlwh[3] / 2.0;
    let a = tlwh[2] / tlwh[3].max(1e-6);
    let h = tlwh[3];
    MeasurementVector::from_vec(vec![cx, cy, a, h])
}

fn xyah_to_tlwh(state: &StateVector) -> [f32; 4] {
    let w = state[2] * state[3];
    let h = state[3];
    let x = state[0] - w / 2.0;
    let y = state[1] - h / 2.0;
    [x, y, w, h]
}

/// Convert a 4-element xyah measurement vector (cx, cy, aspect, height) to TLWH.
fn xyah4_to_tlwh(xyah: &MeasurementVector) -> [f32; 4] {
    let w = xyah[2] * xyah[3];
    let h = xyah[3];
    let x = xyah[0] - w / 2.0;
    let y = xyah[1] - h / 2.0;
    [x, y, w, h]
}

// ---------------------------------------------------------------------------
// Internal detection wrapper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Detection {
    tlwh: [f32; 4],
    score: f32,
    class_id: i64,
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

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
    kf: KalmanFilter,
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
            kf: KalmanFilter::default(),
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

        // 2. Match detections to tracks (OCM direction-consistency bonus + second-round re-match).
        let (matches, unmatched_dets, unmatched_trks) = self.associate(&detections);

        // 3. Update matched tracks.
        for (det_idx, trk_idx) in &matches {
            let det = &detections[*det_idx];
            let xyah = tlwh_to_xyah(&det.tlwh);
            let track = &mut self.tracks[*trk_idx];

            let was_lost = track.time_since_update > 0;

            // ORU: correct KF drift if the track was re-found after a gap.
            if was_lost {
                track.our_re_update(&xyah, self.frame_count, &self.kf);
            }

            track.update_kf(&xyah, &self.kf);
            track.push_observation(xyah, self.frame_count, self.delta_t + 1);

            track.tlwh = det.tlwh;
            track.score = det.score;
            track.class_id = det.class_id;
            track.hits += 1;
            track.hit_streak += 1;
            track.time_since_update = 0;
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
                &self.kf,
            );
            self.next_id += 1;
            self.tracks.push(track);
        }

        // 5. Update states; delete stale tracks.
        for track in &mut self.tracks {
            // Confirm using hit_streak (consecutive hits), matching the reference behaviour.
            if track.time_since_update == 0 && track.hit_streak >= self.min_hits {
                track.state = OcSortTrackState::Confirmed;
            }
            if track.time_since_update > self.max_age {
                track.mark_deleted();
            }
        }

        // Delete unmatched tentative tracks immediately.
        let unmatched_trks_set: HashSet<usize> = unmatched_trks.into_iter().collect();
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if unmatched_trks_set.contains(&i) && track.state == OcSortTrackState::Tentative {
                track.mark_deleted();
            }
        }

        self.tracks.retain(|t| t.state != OcSortTrackState::Deleted);

        // 6. Return confirmed tracks matched this frame.
        self.tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .cloned()
            .collect()
    }

    /// Associate detections with tracks.
    ///
    /// Round 1 IoU on KF-predicted boxes plus an OCM direction-consistency bonus:
    /// for each (track, detection) pair where the track has a stored velocity direction,
    /// a bonus proportional to `cos_similarity * inertia * det_score` is added to the
    /// IoU value before assignment.
    ///
    /// Round 2 for still-unmatched pairs, a second IoU pass is run using each
    /// track's last *observed* position (not the KF prediction), matching the reference
    /// OC-SORT second-round re-matching step.
    fn associate(&self, detections: &[Detection]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        let n_trks = self.tracks.len();
        let n_dets = detections.len();

        if n_trks == 0 {
            return (Vec::new(), (0..n_dets).collect(), Vec::new());
        }
        if n_dets == 0 {
            return (Vec::new(), Vec::new(), (0..n_trks).collect());
        }

        // IoU matrix: rows = tracks, cols = detections.
        let pred_boxes: Vec<[f32; 4]> = self.tracks.iter().map(|t| t.tlwh).collect();
        let det_boxes: Vec<[f32; 4]> = detections.iter().map(|d| d.tlwh).collect();
        let ious = crate::utils::geometry::iou_batch(&pred_boxes, &det_boxes);

        // OCM direction-consistency cost: angle bonus added to IoU before assignment.
        // For track i and detection j:
        //   candidate_dir = normalised vector from track's last observation to detection centre
        //   cos_sim       = dot(stored_velocity_dir, candidate_dir)
        //   bonus         = (pi/2 - |arccos(cos_sim)|) / pi * inertia * det_score
        let mut angle_diff: Vec<Vec<f32>> = vec![vec![0.0_f32; n_dets]; n_trks];

        for (i, track) in self.tracks.iter().enumerate() {
            let vel_dir = match track.obs_direction(self.delta_t) {
                Some(v) => v,
                None => continue,
            };

            let (last_xyah, _) = track.observations.last().unwrap();
            let last_cx = last_xyah[0];
            let last_cy = last_xyah[1];

            for (j, det) in detections.iter().enumerate() {
                let det_cx = det.tlwh[0] + det.tlwh[2] / 2.0;
                let det_cy = det.tlwh[1] + det.tlwh[3] / 2.0;
                let dy = det_cy - last_cy;
                let dx = det_cx - last_cx;
                let norm = (dy * dy + dx * dx).sqrt() + 1e-6;
                let cand_dy = dy / norm;
                let cand_dx = dx / norm;

                let dot = (vel_dir[0] * cand_dy + vel_dir[1] * cand_dx).clamp(-1.0, 1.0);
                let angle = dot.acos();
                let normalized = (std::f32::consts::FRAC_PI_2 - angle.abs()) / std::f32::consts::PI;
                angle_diff[i][j] = (normalized * self.inertia * det.score).max(0.0);
            }
        }

        // Build combined cost matrix and run round-1 assignment.
        let cost_matrix: Vec<Vec<f32>> = (0..n_trks)
            .map(|i| {
                (0..n_dets)
                    .map(|j| 1.0 - (ious[i][j] + angle_diff[i][j]))
                    .collect()
            })
            .collect();

        let (mut matches, mut unmatched_dets, mut unmatched_trks) =
            greedy_match(&cost_matrix, 1.0 - self.iou_threshold);

        // Round 2: re-match using last observed positions for tracks not matched in round 1.
        if !unmatched_dets.is_empty() && !unmatched_trks.is_empty() {
            let left_det_boxes: Vec<[f32; 4]> = unmatched_dets
                .iter()
                .map(|&di| detections[di].tlwh)
                .collect();
            let left_trk_obs: Vec<[f32; 4]> = unmatched_trks
                .iter()
                .map(|&ti| xyah4_to_tlwh(&self.tracks[ti].observations.last().unwrap().0))
                .collect();

            let iou_left = crate::utils::geometry::iou_batch(&left_trk_obs, &left_det_boxes);

            let max_iou = iou_left
                .iter()
                .flat_map(|r| r.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            if max_iou > self.iou_threshold {
                let cost_left: Vec<Vec<f32>> = iou_left
                    .iter()
                    .map(|row| row.iter().map(|&v| 1.0 - v).collect())
                    .collect();
                let (r2_matches, r2_ud, r2_ut) = greedy_match(&cost_left, 1.0 - self.iou_threshold);

                for (det_local, trk_local) in r2_matches {
                    matches.push((unmatched_dets[det_local], unmatched_trks[trk_local]));
                }
                unmatched_dets = r2_ud.into_iter().map(|di| unmatched_dets[di]).collect();
                unmatched_trks = r2_ut.into_iter().map(|ti| unmatched_trks[ti]).collect();
            }
        }

        (matches, unmatched_dets, unmatched_trks)
    }
}

impl Default for OcSort {
    fn default() -> Self {
        Self::new(30, 3, 0.3, 3, 0.2)
    }
}

// ---------------------------------------------------------------------------
// Greedy matching
// ---------------------------------------------------------------------------

fn greedy_match(
    cost_matrix: &[Vec<f32>],
    threshold: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if cost_matrix.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let rows = cost_matrix.len();
    let cols = cost_matrix[0].len();

    let mut matches = Vec::new();
    let mut unmatched_rows: HashSet<usize> = (0..rows).collect();
    let mut unmatched_cols: HashSet<usize> = (0..cols).collect();

    let mut costs: Vec<(f32, usize, usize)> = Vec::new();
    for (r, row) in cost_matrix.iter().enumerate() {
        for (c, &cost) in row.iter().enumerate() {
            costs.push((cost, r, c));
        }
    }
    costs.sort_by(|a, b| a.0.total_cmp(&b.0));

    for (cost, trk, det) in costs {
        if cost > threshold {
            break;
        }
        if unmatched_rows.contains(&trk) && unmatched_cols.contains(&det) {
            matches.push((det, trk));
            unmatched_rows.remove(&trk);
            unmatched_cols.remove(&det);
        }
    }

    (
        matches,
        unmatched_cols.into_iter().collect(),
        unmatched_rows.into_iter().collect(),
    )
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
type PyTrackingResult = (u64, [f32; 4], f32, i64);

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
    fn update(&mut self, detections: Vec<([f32; 4], f32, i64)>) -> PyResult<Vec<PyTrackingResult>> {
        let tracks = self.inner.update(detections);
        Ok(tracks
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, t.score, t.class_id))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_ocsort_two_objects_separate_ids() {
        let mut tracker = OcSort::new(30, 1, 0.3, 3, 0.2);

        let tracks = tracker.update(vec![
            det(100.0, 100.0, 50.0, 100.0, 0.9),
            det(400.0, 400.0, 50.0, 100.0, 0.85),
        ]);

        assert_eq!(tracks.len(), 2);
        let ids: std::collections::HashSet<u64> = tracks.iter().map(|t| t.track_id).collect();
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
        // cx=125, cy=150
        let mut track = OcSortTrack::new(tlwh, 0.9, 0, 1, 1, &kf);

        // Single observation: no direction yet.
        assert!(track.obs_direction(3).is_none());

        // Add a second observation: object moved 10px to the right.
        let tlwh2 = [110.0_f32, 100.0, 50.0, 100.0];
        // cx2=135, cy2=150 → dx=10, dy=0 → direction=[0.0, 1.0]
        let xyah2 = tlwh_to_xyah(&tlwh2);
        track.push_observation(xyah2, 2, 4);

        let dir = track.obs_direction(3);
        assert!(dir.is_some());
        let d = dir.unwrap();
        assert!(
            d[0].abs() < 0.01,
            "Expected dy direction ~0.0, got {}",
            d[0]
        );
        assert!(
            (d[1] - 1.0).abs() < 0.01,
            "Expected dx direction ~1.0, got {}",
            d[1]
        );
    }

    #[test]
    fn test_ocsort_our_does_not_panic_on_re_association() {
        let kf = KalmanFilter::default();
        let tlwh = [100.0_f32, 100.0, 50.0, 100.0];
        let mut track = OcSortTrack::new(tlwh, 0.9, 0, 1, 1, &kf);

        // Simulate 5 frames of predict (no observations).
        for _ in 0..5 {
            track.predict(&kf);
        }

        // Re-associate at frame 7.
        let xyah_new = tlwh_to_xyah(&[130.0_f32, 100.0, 50.0, 100.0]);
        track.our_re_update(&xyah_new, 7, &kf);

        // Check tlwh is finite.
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

        // Frame 1: confirm track (min_hits=1).
        tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);

        // Frame 2: no detection — track predicts, hit_streak should reset.
        tracker.update(vec![]);

        // Frame 3: detection re-appears; hit_streak restarts from 1.
        let tracks = tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].hit_streak, 1);
    }

    #[test]
    fn test_ocsort_second_round_rematches_after_gap() {
        // A track goes missing for one frame, then reappears at the same position.
        // Round-2 matching (using last observed position) should recover it.
        let mut tracker = OcSort::new(5, 1, 0.3, 3, 0.2);

        // Establish a confirmed track.
        for _ in 0..2 {
            tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        }

        // One frame gap.
        tracker.update(vec![]);

        // Reappear at same location — should re-match via round-2 (last observed pos).
        let tracks = tracker.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)]);
        assert_eq!(tracks.len(), 1, "Track should be re-matched after gap");
        assert_eq!(tracks[0].track_id, 1, "Should be the same track");
    }
}

#[cfg(all(test, feature = "python"))]
mod python_tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, s: f32) -> ([f32; 4], f32, i64) {
        ([x, y, w, h], s, 0)
    }

    #[test]
    fn test_py_ocsort_new() {
        pyo3::prepare_freethreaded_python();
        let tracker = PyOcSort::new(30, 3, 0.3, 3, 0.2);
        drop(tracker);
    }

    #[test]
    fn test_py_ocsort_update_empty() {
        pyo3::prepare_freethreaded_python();
        let mut tracker = PyOcSort::new(30, 1, 0.3, 3, 0.2);
        let result = tracker.update(vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_py_ocsort_update_returns_confirmed_tracks() {
        pyo3::prepare_freethreaded_python();
        let mut tracker = PyOcSort::new(30, 1, 0.3, 3, 0.2);
        let dets = vec![det(100.0, 100.0, 50.0, 100.0, 0.9)];
        let result = tracker.update(dets).unwrap();
        assert_eq!(result.len(), 1);
        let (track_id, tlwh, score, class_id) = result[0];
        assert_eq!(track_id, 1);
        assert!(score > 0.0);
        assert_eq!(class_id, 0);
        assert_eq!(tlwh.len(), 4);
    }

    #[test]
    fn test_py_ocsort_default_params() {
        pyo3::prepare_freethreaded_python();
        let mut tracker = PyOcSort::new(30, 3, 0.3, 3, 0.2);
        for _ in 0..3 {
            tracker
                .update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)])
                .unwrap();
        }
        let result = tracker
            .update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)])
            .unwrap();
        assert_eq!(result.len(), 1);
    }
}
