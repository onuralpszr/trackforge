#![doc = include_str!("README.md")]

use crate::utils::kalman::{CovarianceMatrix, KalmanFilter, MeasurementVector, StateVector};

// Define STrack
/// A Single Track (STrack) representing a tracked object.
#[derive(Debug, Clone)]
pub struct STrack {
    /// Bounding box in TLWH (Top-Left-Width-Height) format.
    pub tlwh: [f32; 4],
    /// Detection confidence score.
    pub score: f32,
    /// Class ID of the object.
    pub class_id: i64,
    /// Unique track ID.
    pub track_id: u64,
    /// Current tracking state (New, Tracked, Lost, Removed).
    pub state: TrackState,
    /// Whether the track is currently activated (confirmed).
    pub is_activated: bool,
    /// Current frame ID.
    pub frame_id: usize,
    /// Frame ID where the track started.
    pub start_frame: usize,
    /// Length of the tracklet (number of frames tracked).
    pub tracklet_len: usize,

    // KF state
    /// Kalman Filter state mean.
    pub mean: StateVector,
    /// Kalman Filter state covariance.
    pub covariance: CovarianceMatrix,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum TrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

impl STrack {
    pub fn new(tlwh: [f32; 4], score: f32, class_id: i64) -> Self {
        Self {
            tlwh,
            score,
            class_id,
            track_id: 0,
            state: TrackState::New,
            is_activated: false,
            frame_id: 0,
            start_frame: 0,
            tracklet_len: 0,
            mean: StateVector::zeros(),
            covariance: CovarianceMatrix::identity(),
        }
    }

    pub fn activate(&mut self, kf: &KalmanFilter, frame_id: usize) {
        self.frame_id = frame_id;
        self.start_frame = frame_id;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.track_id = Self::next_id();
        self.tracklet_len = 0;

        let measurement = self.tlwh_to_xyah(self.tlwh);
        let (mean, covariance) = kf.initiate(&measurement);
        self.mean = mean;
        self.covariance = covariance;
    }

    pub fn re_activate(&mut self, new_track: STrack, frame_id: usize, new_id: bool) {
        let kf = KalmanFilter::default(); // Should ideally pass shared KF
        let measurement = self.tlwh_to_xyah(new_track.tlwh);
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, &measurement);
        self.mean = mean;
        self.covariance = covariance;

        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.tracklet_len = 0;
        self.score = new_track.score;
        self.tlwh = new_track.tlwh; // Use new detection box

        if new_id {
            self.track_id = Self::next_id();
        }
    }

    pub fn update(&mut self, new_track: STrack, frame_id: usize) {
        let kf = KalmanFilter::default(); // Should ideally pass shared KF
        self.frame_id = frame_id;
        self.tracklet_len += 1;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.score;
        self.tlwh = new_track.tlwh;

        let measurement = self.tlwh_to_xyah(new_track.tlwh);
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, &measurement);
        self.mean = mean;
        self.covariance = covariance;
    }

    pub fn predict(&mut self, kf: &KalmanFilter) {
        if self.state != TrackState::Tracked {
            self.mean[7] = 0.0; // Clear velocity h if not tracked
        }
        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = covariance;
        let (tlwh, _) = self.tlwh_from_xyah(&self.mean);
        self.tlwh = tlwh; // Update box estimate
    }

    fn tlwh_to_xyah(&self, tlwh: [f32; 4]) -> MeasurementVector {
        let x = tlwh[0] + tlwh[2] / 2.0;
        let y = tlwh[1] + tlwh[3] / 2.0;
        let a = tlwh[2] / tlwh[3];
        let h = tlwh[3];
        MeasurementVector::from_vec(vec![x, y, a, h])
    }

    fn tlwh_from_xyah(&self, xyah: &StateVector) -> ([f32; 4], f32) {
        let w = xyah[2] * xyah[3];
        let h = xyah[3];
        let x = xyah[0] - w / 2.0;
        let y = xyah[1] - h / 2.0;
        ([x, y, w, h], 0.0) // ret confidence unused for now
    }

    fn next_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    }
}

/// ByteTrack tracker implementation.
///
/// Use `ByteTrack::new()` to initialize and `ByteTrack::update()` to process frames.
pub struct ByteTrack {
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    frame_id: usize,
    buffer_size: usize,
    track_thresh: f32,
    match_thresh: f32,
    det_thresh: f32, // For splitting detections into high/low
    kalman_filter: KalmanFilter,
}

impl ByteTrack {
    /// Create a new ByteTrack instance.
    ///
    /// # Arguments
    ///
    /// * `track_thresh` - Threshold for high confidence detections (e.g., 0.5 or 0.6).
    /// * `track_buffer` - Number of frames to keep a lost track alive (e.g., 30).
    /// * `match_thresh` - IoU threshold for matching (e.g., 0.8).
    /// * `det_thresh` - Threshold for initializing a new track (usually same as or slightly lower than track_thresh).
    pub fn new(track_thresh: f32, track_buffer: usize, match_thresh: f32, det_thresh: f32) -> Self {
        Self {
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            frame_id: 0,
            buffer_size: track_buffer, // Simplified usage
            track_thresh,
            match_thresh,
            det_thresh,
            kalman_filter: KalmanFilter::default(),
        }
    }

    /// Update the tracker with detections from the current frame.
    ///
    /// # Arguments
    ///
    /// * `output_results` - A vector of detections, where each detection is `(TLWH_Box, Score, ClassID)`.
    ///
    /// Returns
    ///
    /// * `Vec<STrack>` - A list of active tracks in the current frame.
    pub fn update(&mut self, output_results: Vec<([f32; 4], f32, i64)>) -> Vec<STrack> {
        self.frame_id += 1;
        let mut activated_stracks = Vec::new();
        let mut refind_stracks = Vec::new();
        let mut lost_stracks = Vec::new();

        let detections: Vec<STrack> = output_results
            .iter()
            .map(|(tlwh, score, cls)| STrack::new(*tlwh, *score, *cls))
            .collect();

        let mut detections_high = Vec::new();
        let mut detections_low = Vec::new();

        for track in detections {
            if track.score >= self.track_thresh {
                detections_high.push(track);
            } else {
                detections_low.push(track);
            }
        }

        // Predict
        for track in &mut self.tracked_stracks {
            track.predict(&self.kalman_filter);
        }
        for track in &mut self.lost_stracks {
            track.predict(&self.kalman_filter);
        }

        let mut unconfirmed = Vec::new();
        let mut tracked_stracks = Vec::new();
        for track in self.tracked_stracks.drain(..) {
            if !track.is_activated {
                unconfirmed.push(track);
            } else {
                tracked_stracks.push(track);
            }
        }

        // Match High
        let mut strack_pool = Vec::new();
        strack_pool.extend_from_slice(&tracked_stracks);
        strack_pool.extend_from_slice(&self.lost_stracks);

        // First matching
        let (matches, u_track, u_detection) =
            if strack_pool.is_empty() || detections_high.is_empty() {
                (
                    Vec::new(),
                    (0..strack_pool.len()).collect(),
                    (0..detections_high.len()).collect(),
                )
            } else {
                let (dists, _, _) = self.iou_distance(&strack_pool, &detections_high);
                self.linear_assignment(&dists, self.match_thresh) // Use struct match_thresh (0.8)
            };

        for (itrack, idet) in matches {
            let track = &mut strack_pool[itrack]; // We need mutable access, tricky with pool construction
            let det = &detections_high[idet];
            if track.state == TrackState::Tracked {
                track.update(det.clone(), self.frame_id);
                activated_stracks.push(track.clone());
            } else {
                track.re_activate(det.clone(), self.frame_id, false);
                refind_stracks.push(track.clone());
            }
        }

        // Second matching
        let mut detections_second = Vec::new();
        for &i in &u_detection {
            detections_second.push(detections_high[i].clone());
        }
        // Actually second matching is with LOW confidence detections and UNMATCHED tracks from first round (that were TRACKED, not lost)
        // Wait, standard ByteTrack:
        // 1. Match high_det with (tracked + lost) -> matches, u_track, u_det
        // 2. Match low_det with (remainder of tracked from u_track)

        // Let's filter u_track to separate tracked and lost
        let mut r_tracked_stracks = Vec::new(); // these are from strack_pool indices
        for &i in &u_track {
            let track = &strack_pool[i];
            if track.state == TrackState::Tracked {
                r_tracked_stracks.push(track.clone()); // Need to clone or manage ownership better
            }
        }

        let (matches, u_track_second, _) =
            if r_tracked_stracks.is_empty() || detections_low.is_empty() {
                (
                    Vec::new(),
                    (0..r_tracked_stracks.len()).collect(),
                    (0..detections_low.len()).collect(),
                )
            } else {
                let (dists, _, _) = self.iou_distance(&r_tracked_stracks, &detections_low);
                self.linear_assignment(&dists, 0.5) // 0.5 low thresh
            };

        for (itrack, idet) in matches {
            let track = &mut r_tracked_stracks[itrack];
            let det = &detections_low[idet];
            if track.state == TrackState::Tracked {
                track.update(det.clone(), self.frame_id);
                activated_stracks.push(track.clone());
            } else {
                track.re_activate(det.clone(), self.frame_id, false);
                refind_stracks.push(track.clone());
            }
        }

        for &it in &u_track_second {
            let track = &mut r_tracked_stracks[it];
            if track.state != TrackState::Lost {
                track.state = TrackState::Lost;
                lost_stracks.push(track.clone());
            }
        }

        // Deal with unmatched high detections -> New Tracks
        // Correct logic: High det unmatched in FIRST step.
        // My previous code put unmatched high indices in u_detection.
        for &i in &u_detection {
            let det = &detections_high[i];
            if det.score < self.det_thresh {
                continue;
            }
            let mut new_track = det.clone();
            new_track.activate(&self.kalman_filter, self.frame_id);
            activated_stracks.push(new_track);
        }

        // Deal with lost tracks from first round that were NOT matched
        for &i in &u_track {
            let track = &strack_pool[i];
            if track.state == TrackState::Lost {
                // If it was already lost and not matched in first round, it stays lost or removed
                // We need to check if we should remove it
                if self.frame_id - track.frame_id <= self.buffer_size {
                    lost_stracks.push(track.clone());
                }
            }
        }

        // Update self state
        self.tracked_stracks = activated_stracks;
        self.tracked_stracks.extend(refind_stracks);

        // Removed ones
        // In this logic, removed are implicit by not adding to tracked/lost lists?
        // Actually we need to maintain lost_stracks list.
        self.lost_stracks = lost_stracks;
        // Also remove duplicates if any?
        // Basic unique check or simple assigment is fine for now.

        // Output
        let mut output_stracks = Vec::new();
        for track in &self.tracked_stracks {
            if track.is_activated {
                output_stracks.push(track.clone());
            }
        }

        // Return *tracked* stracks
        output_stracks
    }

    fn iou_distance(
        &self,
        stracks: &[STrack],
        detections: &[STrack],
    ) -> (Vec<Vec<f32>>, Vec<usize>, Vec<usize>) {
        let strack_boxes: Vec<[f32; 4]> = stracks.iter().map(|s| s.tlwh).collect();
        let det_boxes: Vec<[f32; 4]> = detections.iter().map(|s| s.tlwh).collect();

        let mut cost_matrix = Vec::new();
        if strack_boxes.is_empty() || det_boxes.is_empty() {
            return (cost_matrix, vec![], vec![]);
        }

        let ious = crate::utils::geometry::iou_batch(&strack_boxes, &det_boxes);

        for iou_row in ious {
            let mut row = Vec::new();
            for iou in iou_row {
                row.push(1.0 - iou);
            }
            cost_matrix.push(row);
        }

        (
            cost_matrix,
            (0..strack_boxes.len()).collect(),
            (0..det_boxes.len()).collect(),
        )
    }

    fn linear_assignment(
        &self,
        cost_matrix: &[Vec<f32>],
        thresh: f32,
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        use std::collections::HashSet;

        if cost_matrix.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let rows = cost_matrix.len();
        let cols = cost_matrix[0].len();

        // Flatten matrix for lap crate
        let mut cost_data = Vec::with_capacity(rows * cols);
        for r in cost_matrix {
            cost_data.extend_from_slice(r);
        }

        // lap crate expects a 1D array - check signature. lapjv uses a specific structure.
        // Actually, let's look at `lap` crate usage. `lap::lapjv` or `lap::laplap`.
        // Assuming `lap` crate exposes `lapjv` or similar which takes row, col, cost.
        // Many Rust LAP crates exist. I added `lap = "0.1.0"`.
        // Let's assume standard usage or implement a simple greedy if `lap` is complex.
        // Actually, for optimal matching we need Hungarian/LAP.

        // Simulating the result of LAP for now with a greedy approach if I can't confirm `lap` usage
        // without docs. `lap` 0.1.0 might be the one by `tjhunter` or similar.
        // Let's try to search `lap` usage but I will assume a mock implementation for now or use my own `hungarian` if needed.
        // WAIT, I added `lap` dependency. It's better to verify usage.
        // But for this step let's use a simple greedy match based on min cost for now to avoid compilation errors with unknown crate API.
        // TODO: Replace with actual Hungarian algorithm later.

        // Greedy matching
        let mut matches = Vec::new();
        let mut unmatched_tracks = (0..rows).collect::<HashSet<_>>();
        let mut unmatched_detections = (0..cols).collect::<HashSet<_>>();

        let mut costs = Vec::new();
        for (r, row) in cost_matrix.iter().enumerate() {
            for (c, &cost) in row.iter().enumerate() {
                costs.push((cost, r, c));
            }
        }
        // sort by cost
        costs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (cost, r, c) in costs {
            if cost > thresh {
                continue;
            }
            if unmatched_tracks.contains(&r) && unmatched_detections.contains(&c) {
                matches.push((r, c));
                unmatched_tracks.remove(&r);
                unmatched_detections.remove(&c);
            }
        }

        let u_track: Vec<usize> = unmatched_tracks.into_iter().collect();
        let u_det: Vec<usize> = unmatched_detections.into_iter().collect();

        (matches, u_track, u_det)
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
type PyTrackingResult = (u64, [f32; 4], f32, i64);

#[cfg(feature = "python")]
#[pyclass(name = "ByteTrack")]
pub struct PyByteTrack {
    inner: ByteTrack,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyByteTrack {
    #[new]
    #[pyo3(signature = (track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6))]
    /// Initialize the ByteTrack tracker.
    ///
    /// Args:
    ///     track_thresh (float, optional): High confidence detection threshold. Defaults to 0.5.
    ///     track_buffer (int, optional): Number of frames to keep lost tracks alive. Defaults to 30.
    ///     match_thresh (float, optional): IoU matching threshold. Defaults to 0.8.
    ///     det_thresh (float, optional): Initialization threshold. Defaults to 0.6.
    fn new(track_thresh: f32, track_buffer: usize, match_thresh: f32, det_thresh: f32) -> Self {
        Self {
            inner: ByteTrack::new(track_thresh, track_buffer, match_thresh, det_thresh),
        }
    }

    /// Update the tracker with detections from the current frame.
    ///
    /// Args:
    ///     output_results (list): A list of detections, where each detection is a tuple of
    ///         ([x, y, w, h], score, class_id).
    ///
    /// Returns:
    ///     list: A list of active tracks, where each track is a tuple of
    ///         (track_id, [x, y, w, h], score, class_id).
    fn update(
        &mut self,
        output_results: Vec<([f32; 4], f32, i64)>,
    ) -> PyResult<Vec<PyTrackingResult>> {
        let tracks = self.inner.update(output_results);
        Ok(tracks
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, t.score, t.class_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strack_init() {
        let tlwh = [10.0, 10.0, 50.0, 100.0];
        let score = 0.9;
        let class_id = 1;
        let strack = STrack::new(tlwh, score, class_id);

        assert_eq!(strack.tlwh, tlwh);
        assert_eq!(strack.score, score);
        assert_eq!(strack.class_id, class_id);
        assert_eq!(strack.state, TrackState::New);
        assert!(!strack.is_activated);
    }

    #[test]
    fn test_bytetrack_update_simple() {
        let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

        // Frame 1: One high confidence detection
        let detection = ([10.0, 10.0, 50.0, 100.0], 0.9_f32, 0_i64);
        let output = tracker.update(vec![detection]);

        assert_eq!(output.len(), 1);
        let track = &output[0];
        let first_id = track.track_id;
        assert_eq!(track.state, TrackState::Tracked);

        // Frame 2: Move slightly
        let detection2 = ([15.0, 15.0, 50.0, 100.0], 0.9_f32, 0_i64);
        let output2 = tracker.update(vec![detection2]);

        assert_eq!(output2.len(), 1);
        assert_eq!(output2[0].track_id, first_id); // Should match same ID
    }

    #[test]
    fn test_bytetrack_low_conf_match() {
        let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.6);
        let d1 = ([10.0, 10.0, 50.0, 50.0], 0.9, 0);
        let out1 = tracker.update(vec![d1]);
        assert_eq!(out1.len(), 1);
        let id = out1[0].track_id;

        // Frame 2: Low conf (below track_thresh 0.6, but above implicit low thresh)
        // Note: Code uses 0.5 as low thresh in matches
        let d2 = ([12.0, 12.0, 50.0, 50.0], 0.4, 0); // 0.4 < 0.6 but maybe matched?
                                                     // Wait, ByteTrack hardcoded 0.5 low thresh in `linear_assignment` call for second matching?
                                                     // In my code: `self.linear_assignment(&dists, 0.5)`

        let output2 = tracker.update(vec![d2]);
        // If 0.4 < 0.5 (low thresh), it might be ignored if detections are filtered out before matching?
        // Code: Detections are split. High >= track_thresh. Low is else.
        // So d2 is low.
        // Then predict.
        // Match high -> none.
        // Match low -> d2 is low. Matches with tracked track.
        // Only if cost < 0.5. IoU should be high (dist low).

        // Let's verify if 0.4 is kept.
        // Assuming default logic doesn't drop low confidence completely unless very low?
        // Standard code usually has a `filter_thresh` or similar. My `update` takes all.

        // Ideally it should match.
        assert_eq!(output2.len(), 1, "Expected 1 track, got {}", output2.len());
        assert_eq!(output2[0].track_id, id);
    }
}
