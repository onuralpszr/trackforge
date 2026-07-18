#![doc = include_str!("README.md")]

use crate::trackers::common::KalmanTrack;
use crate::utils::geometry::tlwh_to_xyah;
use crate::utils::kalman::{CovarianceMatrix, KalmanFilter, StateVector};

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

    /// Shared per-track Kalman state and predict/update mechanics.
    kalman: KalmanTrack,
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
            kalman: KalmanTrack {
                mean: StateVector::zeros(),
                covariance: CovarianceMatrix::identity(),
            },
        }
    }

    pub fn activate(&mut self, kf: &KalmanFilter, frame_id: usize, track_id: u64) {
        self.frame_id = frame_id;
        self.start_frame = frame_id;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.track_id = track_id;
        self.tracklet_len = 0;

        self.kalman = KalmanTrack::initiate(&tlwh_to_xyah(&self.tlwh), kf);
    }

    pub fn re_activate(
        &mut self,
        new_track: STrack,
        frame_id: usize,
        new_track_id: Option<u64>,
        kf: &KalmanFilter,
    ) {
        self.kalman.update(&tlwh_to_xyah(&new_track.tlwh), kf);

        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.tracklet_len = 0;
        self.score = new_track.score;
        self.tlwh = new_track.tlwh; // Use new detection box

        if let Some(id) = new_track_id {
            self.track_id = id;
        }
    }

    pub fn update(&mut self, new_track: STrack, frame_id: usize, kf: &KalmanFilter) {
        self.frame_id = frame_id;
        self.tracklet_len += 1;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.score;
        self.tlwh = new_track.tlwh;

        self.kalman.update(&tlwh_to_xyah(&new_track.tlwh), kf);
    }

    pub fn predict(&mut self, kf: &KalmanFilter) {
        if self.state != TrackState::Tracked {
            self.kalman.mean[7] = 0.0; // Clear velocity h if not tracked
        }
        self.tlwh = self.kalman.predict(kf); // Update box estimate
    }
}

/// ByteTrack tracker implementation.
///
/// **ByteTrack** is a simple, fast and strong multi-object tracker.
///
/// ## Example
///
/// ```rust
/// use trackforge::trackers::byte_track::ByteTrack;
///
/// // Initialize tracker
/// let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
///
/// // Simulated detections: (tlwh_box, score, class_id)
/// let detections = vec![
///     ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
///     ([200.0, 200.0, 60.0, 120.0], 0.85, 0),
/// ];
///
/// // Update tracker
/// let tracks = tracker.update(detections);
///
/// for track in tracks {
///     println!("Track ID: {}, Box: {:?}", track.track_id, track.tlwh);
/// }
/// ```
pub struct ByteTrack {
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    frame_id: usize,
    buffer_size: usize,
    track_thresh: f32,
    match_thresh: f32,
    det_thresh: f32, // For splitting detections into high/low
    kalman_filter: KalmanFilter,
    next_id: u64,
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
            next_id: 1,
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

        // First matching: high-confidence detections against tracked + lost tracks.
        let pool_boxes: Vec<[f32; 4]> = strack_pool.iter().map(|s| s.tlwh).collect();
        let high_boxes: Vec<[f32; 4]> = detections_high.iter().map(|s| s.tlwh).collect();
        let (matches, u_track, u_detection) =
            crate::utils::assignment::iou_match(&pool_boxes, &high_boxes, self.match_thresh);

        for (itrack, idet) in matches {
            let track = &mut strack_pool[itrack];
            let det = &detections_high[idet];
            if track.state == TrackState::Tracked {
                track.update(det.clone(), self.frame_id, &self.kalman_filter);
                activated_stracks.push(track.clone());
            } else {
                track.re_activate(det.clone(), self.frame_id, None, &self.kalman_filter);
                refind_stracks.push(track.clone());
            }
        }

        // Second matching: low-confidence detections against the tracks that were
        // still Tracked but left unmatched by the first round.
        let mut r_tracked_stracks = Vec::new();
        for &i in &u_track {
            let track = &strack_pool[i];
            if track.state == TrackState::Tracked {
                r_tracked_stracks.push(track.clone());
            }
        }

        // Second matching: low-confidence detections against still-tracked leftovers.
        let r_tracked_boxes: Vec<[f32; 4]> = r_tracked_stracks.iter().map(|s| s.tlwh).collect();
        let low_boxes: Vec<[f32; 4]> = detections_low.iter().map(|s| s.tlwh).collect();
        let (matches, u_track_second, _) =
            crate::utils::assignment::iou_match(&r_tracked_boxes, &low_boxes, 0.5);

        for (itrack, idet) in matches {
            let track = &mut r_tracked_stracks[itrack];
            let det = &detections_low[idet];
            if track.state == TrackState::Tracked {
                track.update(det.clone(), self.frame_id, &self.kalman_filter);
                activated_stracks.push(track.clone());
            } else {
                track.re_activate(det.clone(), self.frame_id, None, &self.kalman_filter);
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

        // Unmatched high-confidence detections above det_thresh start new tracks.
        for &i in &u_detection {
            let det = &detections_high[i];
            if det.score < self.det_thresh {
                continue;
            }
            let mut new_track = det.clone();
            let id = self.next_id;
            self.next_id += 1;
            new_track.activate(&self.kalman_filter, self.frame_id, id);
            activated_stracks.push(new_track);
        }

        // Keep first-round lost tracks alive until they exceed the buffer.
        for &i in &u_track {
            let track = &strack_pool[i];
            if track.state == TrackState::Lost && self.frame_id - track.frame_id <= self.buffer_size
            {
                lost_stracks.push(track.clone());
            }
        }

        // Commit the frame state: matched and refound tracks are active, the rest lost.
        self.tracked_stracks = activated_stracks;
        self.tracked_stracks.extend(refind_stracks);
        self.lost_stracks = lost_stracks;

        // Output the activated tracks for this frame.
        self.tracked_stracks
            .iter()
            .filter(|t| t.is_activated)
            .cloned()
            .collect()
    }
}

impl crate::traits::Tracker for ByteTrack {
    type Track = STrack;

    fn update(&mut self, detections: Vec<crate::traits::Detection>) -> Vec<STrack> {
        self.update(detections)
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::trackers::common::PyTrackingResult;

#[cfg(feature = "python")]
#[pyclass(name = "BYTETRACK")]
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
        // A low-confidence detection (below track_thresh) is recovered by the
        // second association round and keeps the existing track id.
        let mut tracker = ByteTrack::new(0.6, 30, 0.8, 0.6);
        let d1 = ([10.0, 10.0, 50.0, 50.0], 0.9, 0);
        let out1 = tracker.update(vec![d1]);
        assert_eq!(out1.len(), 1);
        let id = out1[0].track_id;

        // Frame 2: same object at low confidence; second-round IoU match keeps the id.
        let d2 = ([12.0, 12.0, 50.0, 50.0], 0.4, 0);
        let output2 = tracker.update(vec![d2]);
        assert_eq!(output2.len(), 1, "Expected 1 track, got {}", output2.len());
        assert_eq!(output2[0].track_id, id);
    }

    #[test]
    fn test_bytetrack_instance_isolation() {
        let mut tracker1 = ByteTrack::new(0.5, 30, 0.8, 0.6);
        let mut tracker2 = ByteTrack::new(0.5, 30, 0.8, 0.6);

        let det1 = vec![([100.0, 100.0, 50.0, 100.0], 0.9_f32, 0_i64)];
        let tracks1 = tracker1.update(det1);
        assert_eq!(tracks1.len(), 1);
        assert_eq!(tracks1[0].track_id, 1);

        let det2 = vec![([100.0, 100.0, 50.0, 100.0], 0.9_f32, 0_i64)];
        let tracks2 = tracker2.update(det2);
        assert_eq!(tracks2.len(), 1);
        assert_eq!(tracks2[0].track_id, 1);
    }

    #[test]
    fn test_bytetrack_tracker_trait() {
        use crate::traits::Tracker;
        let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
        let tracks = Tracker::update(&mut tracker, vec![([10.0, 10.0, 50.0, 100.0], 0.9, 0)]);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_bytetrack_id_sequential() {
        let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

        let det1 = vec![([100.0, 100.0, 50.0, 100.0], 0.9_f32, 0_i64)];
        let tracks1 = tracker.update(det1);
        assert_eq!(tracks1[0].track_id, 1);

        let det2 = vec![([200.0, 200.0, 50.0, 100.0], 0.9_f32, 1_i64)];
        let tracks2 = tracker.update(det2);
        assert_eq!(tracks2[0].track_id, 2);
    }

    #[test]
    fn test_bytetrack_re_activate_lost_track() {
        // Frame 1: high-conf detection track activated (state=Tracked).
        // Frame 2: no detection track unmatched in both rounds state=Lost.
        // Frame 3: detection at same position lost track matched in round 1
        // re_activate() called instead of update().
        let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

        let d = ([10.0, 10.0, 50.0, 100.0], 0.9_f32, 0_i64);
        let out1 = tracker.update(vec![d]);
        assert_eq!(out1.len(), 1);
        let id = out1[0].track_id;

        let out2 = tracker.update(vec![]);
        assert_eq!(out2.len(), 0, "track should not appear while lost");

        let out3 = tracker.update(vec![d]);
        assert_eq!(out3.len(), 1);
        assert_eq!(
            out3[0].track_id, id,
            "re-activated track retains its original ID"
        );
    }

    #[test]
    fn test_bytetrack_lost_track_kept_within_buffer() {
        // Frame 1 activates a track. Frame 2 misses it, so it becomes Lost. Frame 3
        // misses it again while still inside the buffer, exercising the branch that
        // keeps a first-round lost track alive. Frame 4 re-detects it and recovers
        // the original id from the buffer.
        let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
        let d = ([10.0, 10.0, 50.0, 100.0], 0.9_f32, 0_i64);
        let id = tracker.update(vec![d])[0].track_id;

        assert!(tracker.update(vec![]).is_empty());
        assert!(tracker.update(vec![]).is_empty());

        let out = tracker.update(vec![d]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, id, "track recovered from the lost buffer");
    }
}
