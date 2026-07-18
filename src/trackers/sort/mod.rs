#![doc = include_str!("README.md")]

use crate::trackers::common::{CommonParams, KalmanTrack, TrackState};
use crate::utils::geometry::tlwh_to_xyah;
use crate::utils::kalman::KalmanFilter;

/// Settings for [`Sort`].
///
/// The shared lifecycle fields live in [`CommonParams`]; the field below is specific
/// to SORT. Build it with [`SortParams::default`] and tweak what you need, then pass
/// it to [`Sort::from_params`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SortParams {
    /// Shared lifecycle settings, `max_age` and `min_hits`.
    pub common: CommonParams,

    /// Smallest IoU overlap that still counts as the same object between a track and
    /// a detection. This is a minimum IoU, so higher is stricter and separates
    /// touching objects more readily, while lower tolerates loose boxes but risks
    /// swapping ids.
    pub iou_threshold: f32,
}

impl Default for SortParams {
    fn default() -> Self {
        Self {
            common: CommonParams {
                max_age: 1,
                min_hits: 3,
            },
            iou_threshold: 0.3,
        }
    }
}

/// Track state enumeration for SORT.
///
/// Alias of the shared [`TrackState`].
pub type SortTrackState = TrackState;

/// A single track in SORT.
#[derive(Debug, Clone)]
pub struct SortTrack {
    /// Bounding box in TLWH (Top-Left-Width-Height) format.
    pub tlwh: [f32; 4],
    /// Detection confidence score.
    pub score: f32,
    /// Class ID of the object.
    pub class_id: i64,
    /// Unique track ID.
    pub track_id: u64,
    /// Current tracking state.
    pub state: SortTrackState,
    /// Number of consecutive hits (matched detections).
    pub hits: usize,
    /// Number of consecutive misses (no matched detection).
    pub time_since_update: usize,
    /// Total age of the track in frames.
    pub age: usize,

    // Kalman Filter state
    kalman: KalmanTrack,
}

impl SortTrack {
    /// Create a new track from a detection.
    pub fn new(
        tlwh: [f32; 4],
        score: f32,
        class_id: i64,
        kf: &KalmanFilter,
        track_id: u64,
    ) -> Self {
        let measurement = tlwh_to_xyah(&tlwh);
        let kalman = KalmanTrack::initiate(&measurement, kf);

        Self {
            tlwh,
            score,
            class_id,
            track_id,
            state: SortTrackState::Tentative,
            hits: 1,
            time_since_update: 0,
            age: 1,
            kalman,
        }
    }

    /// Predict the next state using Kalman filter.
    pub fn predict(&mut self, kf: &KalmanFilter) {
        self.tlwh = self.kalman.predict(kf);
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Update the track with a matched detection.
    fn update(&mut self, detection: &Detection, kf: &KalmanFilter) {
        let measurement = tlwh_to_xyah(&detection.tlwh);
        self.kalman.update(&measurement, kf);

        self.tlwh = detection.tlwh;
        self.score = detection.score;
        self.class_id = detection.class_id;
        self.hits += 1;
        self.time_since_update = 0;
    }

    /// Mark track as deleted.
    pub fn mark_deleted(&mut self) {
        self.state = SortTrackState::Deleted;
    }

    /// Check if track should be confirmed based on min_hits.
    pub fn is_confirmed(&self) -> bool {
        self.state == SortTrackState::Confirmed
    }
}

/// Internal detection representation.
#[derive(Debug, Clone)]
struct Detection {
    tlwh: [f32; 4],
    score: f32,
    class_id: i64,
}

/// SORT (Simple Online and Realtime Tracking) tracker.
///
/// ## Example
///
/// ```rust
/// use trackforge::trackers::sort::Sort;
///
/// // Initialize tracker
/// let mut tracker = Sort::new(1, 3, 0.3);
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
pub struct Sort {
    tracks: Vec<SortTrack>,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    kalman_filter: KalmanFilter,
    next_id: u64,
}

impl Sort {
    /// Create a new SORT tracker instance.
    ///
    /// # Arguments
    ///
    /// * `max_age` - Maximum frames to keep a track without detection (default: 1).
    /// * `min_hits` - Minimum consecutive hits before track is confirmed (default: 3).
    /// * `iou_threshold` - Minimum IoU for matching detection to track (default: 0.3).
    pub fn new(max_age: usize, min_hits: usize, iou_threshold: f32) -> Self {
        Self::from_params(SortParams {
            common: CommonParams::new(max_age, min_hits),
            iou_threshold,
        })
    }

    /// Create a SORT tracker from a [`SortParams`].
    pub fn from_params(params: SortParams) -> Self {
        Self {
            tracks: Vec::new(),
            max_age: params.common.max_age,
            min_hits: params.common.min_hits,
            iou_threshold: params.iou_threshold,
            kalman_filter: KalmanFilter::default(),
            next_id: 1,
        }
    }

    /// Update the tracker with detections from the current frame.
    ///
    /// # Arguments
    ///
    /// * `detections` - A vector of detections, where each detection is `(TLWH_Box, Score, ClassID)`.
    ///
    /// # Returns
    ///
    /// A vector of confirmed tracks.
    pub fn update(&mut self, detections: Vec<([f32; 4], f32, i64)>) -> Vec<SortTrack> {
        // Convert input to internal Detection format
        let detections: Vec<Detection> = detections
            .into_iter()
            .map(|(tlwh, score, class_id)| Detection {
                tlwh,
                score,
                class_id,
            })
            .collect();

        // Step 1: Predict new locations of existing tracks
        for track in &mut self.tracks {
            track.predict(&self.kalman_filter);
        }

        // Step 2: Match detections to existing tracks using IoU
        let (matches, unmatched_dets, _unmatched_trks) = self.associate(&detections);

        // Step 3: Update matched tracks with detections
        for (det_idx, trk_idx) in matches {
            self.tracks[trk_idx].update(&detections[det_idx], &self.kalman_filter);
        }

        // Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets {
            let det = &detections[det_idx];
            let new_track = SortTrack::new(
                det.tlwh,
                det.score,
                det.class_id,
                &self.kalman_filter,
                self.next_id,
            );
            self.next_id += 1;
            self.tracks.push(new_track);
        }

        // Step 5: Mark unmatched tracks (already incremented time_since_update in predict)
        // No action needed here, time_since_update is already updated in predict

        // Step 6: Update track states and remove dead tracks
        for track in &mut self.tracks {
            if track.time_since_update == 0 && track.hits >= self.min_hits {
                track.state = SortTrackState::Confirmed;
            }
            if track.time_since_update > self.max_age {
                track.mark_deleted();
            }
        }

        // Remove deleted tracks
        self.tracks.retain(|t| t.state != SortTrackState::Deleted);

        // Return confirmed tracks that were updated this frame
        self.tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .cloned()
            .collect()
    }

    /// Associate detections to existing tracks using IoU.
    ///
    /// Returns matches as `(detection, track)` pairs alongside the unmatched
    /// detections and tracks.
    fn associate(&self, detections: &[Detection]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        let track_boxes: Vec<[f32; 4]> = self.tracks.iter().map(|t| t.tlwh).collect();
        let det_boxes: Vec<[f32; 4]> = detections.iter().map(|d| d.tlwh).collect();

        let (matches, unmatched_tracks, unmatched_dets) =
            crate::utils::assignment::iou_match(&track_boxes, &det_boxes, 1.0 - self.iou_threshold);
        let matches = matches.into_iter().map(|(trk, det)| (det, trk)).collect();
        (matches, unmatched_dets, unmatched_tracks)
    }
}

impl Default for Sort {
    fn default() -> Self {
        Self::new(1, 3, 0.3)
    }
}

impl crate::traits::Tracker for Sort {
    type Track = SortTrack;

    fn update(&mut self, detections: Vec<crate::traits::Detection>) -> Vec<SortTrack> {
        self.update(detections)
    }
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::trackers::common::PyTrackingResult;

#[cfg(feature = "python")]
#[pyclass(name = "SORT")]
pub struct PySort {
    inner: Sort,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySort {
    #[new]
    #[pyo3(signature = (max_age=1, min_hits=3, iou_threshold=0.3))]
    /// Initialize the SORT tracker.
    ///
    /// Args:
    ///     max_age (int, optional): Maximum frames to keep track without detection. Defaults to 1.
    ///     min_hits (int, optional): Minimum hits before track is confirmed. Defaults to 3.
    ///     iou_threshold (float, optional): IoU threshold for matching. Defaults to 0.3.
    fn new(max_age: usize, min_hits: usize, iou_threshold: f32) -> Self {
        Self {
            inner: Sort::new(max_age, min_hits, iou_threshold),
        }
    }

    /// Update the tracker with detections from the current frame.
    ///
    /// Args:
    ///     detections (list): A list of detections, where each detection is a tuple of
    ///         ([x, y, w, h], score, class_id).
    ///
    /// Returns:
    ///     list: A list of confirmed tracks, where each track is a tuple of
    ///         (track_id, [x, y, w, h], score, class_id).
    fn update(&mut self, detections: Vec<([f32; 4], f32, i64)>) -> PyResult<Vec<PyTrackingResult>> {
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

    #[test]
    fn test_sort_track_creation() {
        let kf = KalmanFilter::default();
        let track = SortTrack::new([10.0, 20.0, 30.0, 40.0], 0.9, 1, &kf, 1);

        assert_eq!(track.tlwh, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(track.score, 0.9);
        assert_eq!(track.class_id, 1);
        assert_eq!(track.state, SortTrackState::Tentative);
        assert_eq!(track.hits, 1);
        assert_eq!(track.time_since_update, 0);
    }

    #[test]
    fn test_sort_single_detection() {
        let mut tracker = Sort::new(1, 1, 0.3); // min_hits=1 for immediate confirmation

        let detections = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks = tracker.update(detections);

        assert_eq!(tracks.len(), 1);
        assert!(tracks[0].track_id > 0);
    }

    #[test]
    fn test_sort_track_continuity() {
        let mut tracker = Sort::new(1, 1, 0.3);

        // Frame 1
        let det1 = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks1 = tracker.update(det1);
        assert_eq!(tracks1.len(), 1);
        let track_id = tracks1[0].track_id;

        // Frame 2: Same object moved slightly
        let det2 = vec![([105.0, 105.0, 50.0, 100.0], 0.9, 0)];
        let tracks2 = tracker.update(det2);
        assert_eq!(tracks2.len(), 1);
        assert_eq!(tracks2[0].track_id, track_id); // Same track ID
    }

    #[test]
    fn test_sort_min_hits() {
        let mut tracker = Sort::new(1, 3, 0.3); // Require 3 hits

        // Frame 1: First detection (hits=1, tentative)
        let det = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks = tracker.update(det.clone());
        assert_eq!(tracks.len(), 0); // Not confirmed yet

        // Frame 2: Second detection (hits=2, still tentative)
        let tracks = tracker.update(det.clone());
        assert_eq!(tracks.len(), 0);

        // Frame 3: Third detection (hits=3, now confirmed)
        let tracks = tracker.update(det);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_sort_max_age() {
        let mut tracker = Sort::new(2, 1, 0.3); // max_age=2

        // Frame 1: Create track
        let det = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        tracker.update(det);

        // Frame 2: No detection (time_since_update=1)
        let tracks = tracker.update(vec![]);
        assert_eq!(tracks.len(), 0); // Not output but still alive

        // Frame 3: No detection (time_since_update=2)
        tracker.update(vec![]);

        // Frame 4: No detection (time_since_update=3 > max_age=2, deleted)
        tracker.update(vec![]);

        // Frame 5: New detection should create new track
        let det = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks = tracker.update(det);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_sort_multiple_objects() {
        let mut tracker = Sort::new(1, 1, 0.3);

        let detections = vec![
            ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
            ([300.0, 300.0, 50.0, 100.0], 0.85, 1),
        ];

        let tracks = tracker.update(detections);
        assert_eq!(tracks.len(), 2);

        // Check different track IDs
        assert_ne!(tracks[0].track_id, tracks[1].track_id);
    }

    #[test]
    fn test_sort_instance_isolation() {
        let mut tracker1 = Sort::new(1, 1, 0.3);
        let mut tracker2 = Sort::new(1, 1, 0.3);

        let det1 = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks1 = tracker1.update(det1);
        assert_eq!(tracks1.len(), 1);
        assert_eq!(tracks1[0].track_id, 1);

        let det2 = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks2 = tracker2.update(det2);
        assert_eq!(tracks2.len(), 1);
        assert_eq!(tracks2[0].track_id, 1);
    }

    #[test]
    fn test_sort_tracker_trait() {
        use crate::traits::Tracker;
        let mut tracker = Sort::new(1, 1, 0.3);
        let tracks = Tracker::update(&mut tracker, vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)]);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn test_sort_id_sequential() {
        let mut tracker = Sort::new(1, 1, 0.3);

        let det1 = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
        let tracks1 = tracker.update(det1);
        assert_eq!(tracks1[0].track_id, 1);

        let det2 = vec![([200.0, 200.0, 50.0, 100.0], 0.9, 1)];
        let tracks2 = tracker.update(det2);
        assert_eq!(tracks2[0].track_id, 2);
    }

    #[test]
    fn storage_stays_bounded_under_churn() {
        // A fresh, non-matching object each frame. Tracks must be deleted after
        // max_age misses, keeping the internal vector bounded by that window.
        let max_age = 20;
        let mut tracker = Sort::new(max_age, 3, 0.3);
        for f in 0..3000 {
            let x = 5.0 + (f % 100) as f32 * 40.0;
            let _ = tracker.update(vec![([x, 10.0, 20.0, 40.0], 0.9, 0)]);
            assert!(
                tracker.tracks.len() <= max_age + 5,
                "storage grew to {} at frame {f}",
                tracker.tracks.len()
            );
        }
    }
}
