#![doc = include_str!("README.md")]

use crate::trackers::common::{CommonParams, KalmanTrack};
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
    second_match_thresh: f32,
    kalman_filter: KalmanFilter,
    next_id: u64,
}

/// Settings for [`ByteTrack`].
///
/// The shared lifecycle fields live in [`CommonParams`]; ByteTrack maps its track
/// buffer onto `common.max_age`. Build it with [`ByteTrackParams::default`], tweak
/// what you need, then pass it to [`ByteTrack::from_params`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteTrackParams {
    /// Shared lifecycle settings. `common.max_age` is how many frames a lost track is
    /// kept alive, what ByteTrack calls the track buffer.
    pub common: CommonParams,

    /// Score above which a detection is treated as high confidence and matched first.
    /// Detections below it are held back for the second, low confidence pass.
    pub track_thresh: f32,

    /// First stage match cutoff, given as a maximum IoU distance of one minus IoU. A
    /// pair matches when its IoU distance is at or below this, so a value of 0.8 means
    /// the boxes only need an IoU of 0.2. Lower is stricter.
    pub match_thresh: f32,

    /// Smallest score an unmatched high confidence detection needs to start a brand
    /// new track. Raising it avoids spawning tracks from weak one-off detections.
    pub det_thresh: f32,

    /// Second stage match cutoff for recovering objects from low confidence
    /// detections, again a maximum IoU distance. This is ByteTrack's core recovery
    /// step, and the reference value is 0.5.
    pub second_match_thresh: f32,
}

impl Default for ByteTrackParams {
    fn default() -> Self {
        Self {
            common: CommonParams {
                max_age: 30,
                min_hits: 3,
            },
            track_thresh: 0.5,
            match_thresh: 0.8,
            det_thresh: 0.6,
            second_match_thresh: 0.5,
        }
    }
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
        Self::from_params(ByteTrackParams {
            common: CommonParams {
                max_age: track_buffer,
                min_hits: 3,
            },
            track_thresh,
            match_thresh,
            det_thresh,
            ..ByteTrackParams::default()
        })
    }

    /// Create a ByteTrack tracker from a [`ByteTrackParams`].
    ///
    /// ByteTrack activates a track on its first high confidence match, so
    /// `params.common.min_hits` has no effect here; only `common.max_age` is used.
    pub fn from_params(params: ByteTrackParams) -> Self {
        Self {
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            frame_id: 0,
            buffer_size: params.common.max_age,
            track_thresh: params.track_thresh,
            match_thresh: params.match_thresh,
            det_thresh: params.det_thresh,
            second_match_thresh: params.second_match_thresh,
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
            crate::utils::assignment::iou_match(&r_tracked_boxes, &low_boxes, self.second_match_thresh);

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
    #[pyo3(signature = (track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6, second_match_thresh=0.5))]
    /// Initialize the ByteTrack tracker.
    ///
    /// Args:
    ///     track_thresh (float, optional): Score above which a detection is high
    ///         confidence and matched first. Defaults to 0.5.
    ///     track_buffer (int, optional): Frames a lost track is kept alive so it can
    ///         be recovered. Defaults to 30.
    ///     match_thresh (float, optional): First stage match cutoff as a maximum IoU
    ///         distance of one minus IoU. Lower is stricter. Defaults to 0.8.
    ///     det_thresh (float, optional): Smallest score an unmatched high confidence
    ///         detection needs to start a new track. Defaults to 0.6.
    ///     second_match_thresh (float, optional): Second stage match cutoff for
    ///         recovering low confidence detections, a maximum IoU distance.
    ///         Defaults to 0.5.
    fn new(
        track_thresh: f32,
        track_buffer: usize,
        match_thresh: f32,
        det_thresh: f32,
        second_match_thresh: f32,
    ) -> Self {
        Self {
            inner: ByteTrack::from_params(ByteTrackParams {
                common: CommonParams {
                    max_age: track_buffer,
                    min_hits: 3,
                },
                track_thresh,
                match_thresh,
                det_thresh,
                second_match_thresh,
            }),
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

    #[test]
    fn storage_stays_bounded_under_churn() {
        // Every frame a brand-new object appears far from any previous one and then
        // vanishes, so nothing ever re-matches. Lost tracks must age out of the
        // buffer instead of piling up. The internal vectors must stay bounded by the
        // buffer window, not grow with the frame count.
        let buffer = 30;
        let mut tracker = ByteTrack::new(0.5, buffer, 0.8, 0.6);
        for f in 0..3000 {
            let x = 5.0 + (f % 100) as f32 * 40.0; // spaced apart, cycle exceeds buffer
            let _ = tracker.update(vec![([x, 10.0, 20.0, 40.0], 0.9, 0)]);
            let stored = tracker.tracked_stracks.len() + tracker.lost_stracks.len();
            assert!(
                stored <= buffer + 5,
                "storage grew to {stored} at frame {f}, buffer is {buffer}"
            );
        }
    }

    #[test]
    fn active_ids_are_unique_each_frame() {
        // Five persistent objects; every active id in a frame must be distinct.
        let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
        for f in 0..200 {
            let dets: Vec<_> = (0..5)
                .map(|i| {
                    let x = 20.0 + i as f32 * 120.0 + (f % 3) as f32;
                    ([x, 30.0, 40.0, 80.0], 0.9, i)
                })
                .collect();
            let out = tracker.update(dets);
            let mut ids: Vec<u64> = out.iter().map(|t| t.track_id).collect();
            ids.sort_unstable();
            let unique = {
                let mut u = ids.clone();
                u.dedup();
                u.len()
            };
            assert_eq!(ids.len(), unique, "duplicate id in frame {f}: {ids:?}");
        }
    }

    #[test]
    fn from_params_default_matches_new() {
        // The params path with defaults must behave exactly like the positional new.
        let mut a = ByteTrack::from_params(ByteTrackParams::default());
        let mut b = ByteTrack::new(0.5, 30, 0.8, 0.6);
        let d = vec![([10.0, 10.0, 50.0, 100.0], 0.9, 0)];
        let ra = a.update(d.clone());
        let rb = b.update(d);
        assert_eq!(ra.len(), rb.len());
        assert_eq!(ra[0].track_id, rb[0].track_id);
        assert_eq!(ra[0].tlwh, rb[0].tlwh);
    }

    #[test]
    fn second_match_thresh_controls_low_stage_recovery() {
        let high = ([10.0, 10.0, 50.0, 50.0], 0.9, 0);
        let low = ([12.0, 12.0, 50.0, 50.0], 0.4, 0); // below track_thresh, second stage

        // Impossible second stage: the low-confidence detection cannot be recovered.
        let mut strict = ByteTrack::from_params(ByteTrackParams {
            second_match_thresh: 0.0,
            ..ByteTrackParams::default()
        });
        strict.update(vec![high]);
        assert!(strict.update(vec![low]).is_empty());

        // Default second stage recovers the same object and keeps its id.
        let mut lax = ByteTrack::from_params(ByteTrackParams::default());
        let id = lax.update(vec![high])[0].track_id;
        let out = lax.update(vec![low]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, id);
    }
}
