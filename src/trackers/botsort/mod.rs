#![doc = include_str!("README.md")]

use crate::trackers::byte_track::TrackState;
use crate::trackers::common::{CameraMotion, CommonParams, KalmanTrack};
use crate::utils::assignment::{greedy_match, iou_match};
use crate::utils::features::{cosine_distance, l2_normalize};
use crate::utils::geometry::{iou_batch, tlwh_to_xyah};
use crate::utils::kalman::KalmanFilter;

/// Exponential moving-average weight for the per-track appearance feature.
const FEATURE_MOMENTUM: f32 = 0.9;

/// A detection with an optional appearance embedding.
struct Detection {
    tlwh: [f32; 4],
    score: f32,
    class_id: i64,
    feature: Option<Vec<f32>>,
}

/// A single tracked object managed by BoT-SORT.
///
/// Carries the shared [`KalmanTrack`] state, the ByteTrack-style lifecycle used by
/// the two-stage cascade, and a smoothed appearance embedding (exponential moving
/// average) used for the appearance-fused association.
#[derive(Debug, Clone)]
pub struct BotTrack {
    /// Bounding box in TLWH (top-left x, top-left y, width, height) format.
    pub tlwh: [f32; 4],
    /// Detection confidence of the most recent match.
    pub score: f32,
    /// Class label of the most recent match.
    pub class_id: i64,
    /// Unique track identifier (0 until the track is activated).
    pub track_id: u64,
    /// Lifecycle state (`New`, `Tracked`, `Lost`, `Removed`).
    pub state: TrackState,
    /// Whether the track is confirmed and returned to callers.
    pub is_activated: bool,
    /// Frame id of the most recent update.
    pub frame_id: usize,
    /// Frame id at which the track started.
    pub start_frame: usize,
    /// Number of consecutive frames the track has been followed.
    pub tracklet_len: usize,

    kalman: KalmanTrack,
    smooth_feat: Option<Vec<f32>>,
}

impl BotTrack {
    /// Build an unactivated track from a detection (Kalman state seeded from its box).
    fn from_detection(det: &Detection, kf: &KalmanFilter) -> Self {
        let kalman = KalmanTrack::initiate(&tlwh_to_xyah(&det.tlwh), kf);
        Self {
            tlwh: det.tlwh,
            score: det.score,
            class_id: det.class_id,
            track_id: 0,
            state: TrackState::New,
            is_activated: false,
            frame_id: 0,
            start_frame: 0,
            tracklet_len: 0,
            kalman,
            smooth_feat: det.feature.as_ref().map(|f| l2_normalize(f)),
        }
    }

    /// Confirm a fresh detection as a new track with the given id.
    fn activate(&mut self, frame_id: usize, track_id: u64) {
        self.track_id = track_id;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.start_frame = frame_id;
        self.tracklet_len = 0;
    }

    /// Kalman-correct with a matched detection and adopt its box, score, and class.
    fn absorb(&mut self, det: &Detection, frame_id: usize, kf: &KalmanFilter) {
        self.kalman.update(&tlwh_to_xyah(&det.tlwh), kf);
        self.tlwh = det.tlwh;
        self.score = det.score;
        self.class_id = det.class_id;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        if let Some(f) = &det.feature {
            self.update_features(f);
        }
    }

    /// Continue an already-tracked track (extends the tracklet).
    fn update(&mut self, det: &Detection, frame_id: usize, kf: &KalmanFilter) {
        self.absorb(det, frame_id, kf);
        self.tracklet_len += 1;
    }

    /// Bring a lost track back with a matched detection, keeping its id.
    fn re_activate(&mut self, det: &Detection, frame_id: usize, kf: &KalmanFilter) {
        self.absorb(det, frame_id, kf);
        self.tracklet_len = 0;
    }

    /// Kalman-predict one step forward, damping height velocity while not tracked.
    fn predict(&mut self, kf: &KalmanFilter) {
        if self.state != TrackState::Tracked {
            self.kalman.mean[7] = 0.0;
        }
        self.tlwh = self.kalman.predict(kf);
    }

    /// Warp the predicted state by a camera motion transform.
    fn apply_camera_motion(&mut self, cmc: &CameraMotion) {
        self.tlwh = self.kalman.apply_camera_motion(cmc);
    }

    /// Blend a new embedding into the smoothed feature (EMA, then renormalise).
    fn update_features(&mut self, feature: &[f32]) {
        let feature = l2_normalize(feature);
        match &mut self.smooth_feat {
            None => self.smooth_feat = Some(feature),
            Some(sf) => {
                for (s, f) in sf.iter_mut().zip(&feature) {
                    *s = FEATURE_MOMENTUM * *s + (1.0 - FEATURE_MOMENTUM) * f;
                }
                *sf = l2_normalize(sf);
            }
        }
    }
}

/// BoT-SORT tracker.
///
/// Extends ByteTrack's two-stage cascade with camera motion compensation and an
/// appearance-fused first stage. Camera motion is supplied by the caller as an
/// affine transform (see [`CameraMotion`]); appearance embeddings are optional and,
/// when present, fused into the high-confidence association by taking the smaller of
/// the IoU distance and the gated cosine distance.
///
/// ## Example
///
/// ```rust
/// use trackforge::trackers::botsort::BotSort;
///
/// // track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6,
/// // proximity_thresh=0.5, appearance_thresh=0.25
/// let mut tracker = BotSort::new(0.5, 30, 0.8, 0.6, 0.5, 0.25);
///
/// let detections = vec![([100.0, 100.0, 50.0, 100.0], 0.9, 0)];
/// let tracks = tracker.update(detections, &[]);
/// for t in &tracks {
///     println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
/// }
/// ```
pub struct BotSort {
    tracked_stracks: Vec<BotTrack>,
    lost_stracks: Vec<BotTrack>,
    frame_id: usize,
    buffer_size: usize,
    track_thresh: f32,
    match_thresh: f32,
    det_thresh: f32,
    second_match_thresh: f32,
    proximity_thresh: f32,
    appearance_thresh: f32,
    kalman_filter: KalmanFilter,
    next_id: u64,
}

/// Settings for [`BotSort`].
///
/// BoT-SORT is ByteTrack plus camera motion and an optional appearance term, so its
/// params look like ByteTrack's plus two Re-ID gates. Shared lifecycle fields live in
/// [`CommonParams`]; BoT-SORT maps its track buffer onto `common.max_age` and, like
/// ByteTrack, activates on the first match so `common.min_hits` has no effect. Build
/// it with [`BotSortParams::default`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BotSortParams {
    /// Shared lifecycle settings. `common.max_age` is the track buffer length.
    pub common: CommonParams,

    /// Score above which a detection is treated as high confidence and matched first.
    pub track_thresh: f32,

    /// First stage match cutoff, a maximum IoU distance of one minus IoU. Lower is
    /// stricter.
    pub match_thresh: f32,

    /// Smallest score an unmatched high confidence detection needs to start a new
    /// track.
    pub det_thresh: f32,

    /// Second stage match cutoff for recovering objects from low confidence
    /// detections, a maximum IoU distance. Reference value 0.5.
    pub second_match_thresh: f32,

    /// How much boxes must overlap before appearance is allowed to influence the
    /// match, as a maximum IoU distance. If a track and a detection are farther apart
    /// than this, only motion is used and appearance is ignored.
    pub proximity_thresh: f32,

    /// How close two appearance embeddings must be for Re-ID to help the match, as a
    /// maximum cosine distance. Above this the appearance term is dropped and the
    /// match falls back to motion.
    pub appearance_thresh: f32,
}

impl Default for BotSortParams {
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
            proximity_thresh: 0.5,
            appearance_thresh: 0.25,
        }
    }
}

impl BotSort {
    /// Create a new BoT-SORT tracker.
    ///
    /// # Arguments
    ///
    /// * `track_thresh` - Confidence split between high- and low-score detections (default: 0.5).
    /// * `track_buffer` - Frames a lost track is kept alive (default: 30).
    /// * `match_thresh` - Maximum cost for a first-stage match (default: 0.8).
    /// * `det_thresh` - Minimum score to start a new track (default: 0.6).
    /// * `proximity_thresh` - IoU-distance gate above which appearance is ignored (default: 0.5).
    /// * `appearance_thresh` - Cosine-distance gate above which appearance is ignored (default: 0.25).
    pub fn new(
        track_thresh: f32,
        track_buffer: usize,
        match_thresh: f32,
        det_thresh: f32,
        proximity_thresh: f32,
        appearance_thresh: f32,
    ) -> Self {
        Self::from_params(BotSortParams {
            common: CommonParams {
                max_age: track_buffer,
                min_hits: 3,
            },
            track_thresh,
            match_thresh,
            det_thresh,
            proximity_thresh,
            appearance_thresh,
            ..BotSortParams::default()
        })
    }

    /// Create a BoT-SORT tracker from a [`BotSortParams`].
    ///
    /// BoT-SORT activates a track on its first high confidence match, so
    /// `params.common.min_hits` has no effect; only `common.max_age` is used.
    pub fn from_params(params: BotSortParams) -> Self {
        Self {
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            frame_id: 0,
            buffer_size: params.common.max_age,
            track_thresh: params.track_thresh,
            match_thresh: params.match_thresh,
            det_thresh: params.det_thresh,
            second_match_thresh: params.second_match_thresh,
            proximity_thresh: params.proximity_thresh,
            appearance_thresh: params.appearance_thresh,
            kalman_filter: KalmanFilter::default(),
            next_id: 1,
        }
    }

    /// Update the tracker with the current frame's detections and embeddings.
    ///
    /// `embeddings` is parallel to `detections`; pass an empty slice to track on
    /// motion only. Returns the activated tracks for this frame.
    pub fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: &[Vec<f32>],
    ) -> Vec<BotTrack> {
        self.update_with_camera_motion(detections, embeddings, &CameraMotion::identity())
    }

    /// Update the tracker, first warping track predictions by `camera_motion`.
    ///
    /// `camera_motion` maps the previous frame's coordinates into the current frame
    /// (see [`CameraMotion`]); pass [`CameraMotion::identity`] for a static camera.
    pub fn update_with_camera_motion(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: &[Vec<f32>],
        camera_motion: &CameraMotion,
    ) -> Vec<BotTrack> {
        self.frame_id += 1;

        let use_reid = !embeddings.is_empty() && embeddings.len() == detections.len();

        // Split detections into high- and low-confidence sets, carrying embeddings.
        let mut dets_high: Vec<Detection> = Vec::new();
        let mut dets_low: Vec<Detection> = Vec::new();
        for (i, (tlwh, score, class_id)) in detections.into_iter().enumerate() {
            let feature = if use_reid {
                Some(embeddings[i].clone())
            } else {
                None
            };
            let det = Detection {
                tlwh,
                score,
                class_id,
                feature,
            };
            if det.score >= self.track_thresh {
                dets_high.push(det);
            } else {
                dets_low.push(det);
            }
        }

        // Predict, then warp by camera motion.
        let warp = !camera_motion.is_identity();
        for track in self
            .tracked_stracks
            .iter_mut()
            .chain(&mut self.lost_stracks)
        {
            track.predict(&self.kalman_filter);
            if warp {
                track.apply_camera_motion(camera_motion);
            }
        }

        // First stage: high-confidence detections against tracked + lost tracks,
        // matched on the appearance-fused cost. Tracks are activated on creation, so
        // pool[0..n_tracked] are the tracked tracks and the rest are lost.
        let mut tracked: Vec<BotTrack> = self.tracked_stracks.drain(..).collect();
        let n_tracked = tracked.len();
        let mut pool: Vec<BotTrack> = Vec::new();
        pool.append(&mut tracked);
        pool.append(&mut self.lost_stracks);

        let mut activated: Vec<BotTrack> = Vec::new();
        let mut refind: Vec<BotTrack> = Vec::new();
        let mut lost: Vec<BotTrack> = Vec::new();

        // Guard the empty case: greedy_match on an empty matrix cannot report the
        // unmatched detections, so a first frame (no tracks) would drop them.
        let (matches, u_track, u_det_high) = if pool.is_empty() || dets_high.is_empty() {
            (
                Vec::new(),
                (0..pool.len()).collect::<Vec<_>>(),
                (0..dets_high.len()).collect::<Vec<_>>(),
            )
        } else {
            let cost = self.fused_cost(&pool, &dets_high, use_reid);
            greedy_match(&cost, self.match_thresh)
        };

        for (itrack, idet) in matches {
            let det = &dets_high[idet];
            if pool[itrack].state == TrackState::Tracked {
                pool[itrack].update(det, self.frame_id, &self.kalman_filter);
                activated.push(pool[itrack].clone());
            } else {
                pool[itrack].re_activate(det, self.frame_id, &self.kalman_filter);
                refind.push(pool[itrack].clone());
            }
        }

        // Second stage: low-confidence detections against still-Tracked leftovers,
        // matched on IoU only.
        let r_tracked: Vec<usize> = u_track
            .iter()
            .copied()
            .filter(|&i| pool[i].state == TrackState::Tracked)
            .collect();
        let r_boxes: Vec<[f32; 4]> = r_tracked.iter().map(|&i| pool[i].tlwh).collect();
        let low_boxes: Vec<[f32; 4]> = dets_low.iter().map(|d| d.tlwh).collect();
        let (matches_low, u_track_low, _) =
            iou_match(&r_boxes, &low_boxes, self.second_match_thresh);

        // `r_tracked` only holds Tracked-state tracks, so a second-stage match is
        // always a plain update (never a re-activation).
        for (local, idet) in matches_low {
            let itrack = r_tracked[local];
            let det = &dets_low[idet];
            pool[itrack].update(det, self.frame_id, &self.kalman_filter);
            activated.push(pool[itrack].clone());
        }

        // Tracked leftovers from the second stage become Lost.
        for &local in &u_track_low {
            let itrack = r_tracked[local];
            if pool[itrack].state != TrackState::Lost {
                pool[itrack].state = TrackState::Lost;
                lost.push(pool[itrack].clone());
            }
        }

        // Unmatched high detections above det_thresh start new tracks.
        for &idet in &u_det_high {
            let det = &dets_high[idet];
            if det.score < self.det_thresh {
                continue;
            }
            let mut track = BotTrack::from_detection(det, &self.kalman_filter);
            track.activate(self.frame_id, self.next_id);
            self.next_id += 1;
            activated.push(track);
        }

        // Keep already-lost tracks (unmatched this frame) alive until they exceed
        // the buffer. Tracks that just became lost in the second stage are already
        // in `lost`, so restrict this to the originally-lost part of the pool.
        for &i in &u_track {
            if i >= n_tracked && self.frame_id - pool[i].frame_id <= self.buffer_size {
                lost.push(pool[i].clone());
            }
        }

        // Commit the frame state.
        self.tracked_stracks = activated;
        self.tracked_stracks.extend(refind);
        self.lost_stracks = lost;

        self.tracked_stracks
            .iter()
            .filter(|t| t.is_activated)
            .cloned()
            .collect()
    }

    /// First-stage cost matrix (`n_tracks x n_dets`).
    ///
    /// The IoU distance `1 - IoU`, fused with a gated cosine appearance distance when
    /// embeddings are present: appearance is used only when it is below
    /// `appearance_thresh` and the pair is spatially close (IoU distance below
    /// `proximity_thresh`), and the two costs are combined by taking the smaller.
    fn fused_cost(&self, tracks: &[BotTrack], dets: &[Detection], use_reid: bool) -> Vec<Vec<f32>> {
        let track_boxes: Vec<[f32; 4]> = tracks.iter().map(|t| t.tlwh).collect();
        let det_boxes: Vec<[f32; 4]> = dets.iter().map(|d| d.tlwh).collect();
        let ious = iou_batch(&track_boxes, &det_boxes);

        (0..tracks.len())
            .map(|i| {
                (0..dets.len())
                    .map(|j| {
                        let iou_dist = 1.0 - ious[i][j];
                        let emb = self.appearance_cost(use_reid, &tracks[i], &dets[j], iou_dist);
                        emb.map_or(iou_dist, |e| iou_dist.min(e))
                    })
                    .collect()
            })
            .collect()
    }

    /// Gated cosine appearance cost for one (track, detection) pair.
    ///
    /// Returns `None` when appearance is unavailable or disabled. When the pair is not
    /// spatially close or the embeddings are too dissimilar, the cost is neutralised to
    /// `1.0` so the fused cost falls back to motion.
    fn appearance_cost(
        &self,
        use_reid: bool,
        track: &BotTrack,
        det: &Detection,
        iou_dist: f32,
    ) -> Option<f32> {
        if !use_reid {
            return None;
        }
        let tf = track.smooth_feat.as_ref()?;
        let df = det.feature.as_ref()?;
        let mut emb = cosine_distance(tf, df);
        if emb > self.appearance_thresh || iou_dist > self.proximity_thresh {
            emb = 1.0;
        }
        Some(emb)
    }
}

impl Default for BotSort {
    fn default() -> Self {
        Self::new(0.5, 30, 0.8, 0.6, 0.5, 0.25)
    }
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed BoT-SORT tracker.
#[cfg(feature = "python")]
#[pyclass(name = "BOTSORT")]
pub struct PyBotSort {
    inner: BotSort,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBotSort {
    #[new]
    #[pyo3(signature = (track_thresh=0.5, track_buffer=30, match_thresh=0.8, det_thresh=0.6, second_match_thresh=0.5, proximity_thresh=0.5, appearance_thresh=0.25))]
    /// Initialize the BoT-SORT tracker.
    ///
    /// Args:
    ///     track_thresh (float): Score above which a detection is high confidence and
    ///         matched first. Default: 0.5.
    ///     track_buffer (int): Frames a lost track is kept alive. Default: 30.
    ///     match_thresh (float): First stage match cutoff as a maximum IoU distance of
    ///         one minus IoU. Lower is stricter. Default: 0.8.
    ///     det_thresh (float): Smallest score an unmatched high confidence detection
    ///         needs to start a new track. Default: 0.6.
    ///     second_match_thresh (float): Second stage match cutoff for recovering low
    ///         confidence detections, a maximum IoU distance. Default: 0.5.
    ///     proximity_thresh (float): How much boxes must overlap before appearance is
    ///         used, as a maximum IoU distance. Default: 0.5.
    ///     appearance_thresh (float): How close two embeddings must be for Re-ID to
    ///         help, as a maximum cosine distance. Default: 0.25.
    fn new(
        track_thresh: f32,
        track_buffer: usize,
        match_thresh: f32,
        det_thresh: f32,
        second_match_thresh: f32,
        proximity_thresh: f32,
        appearance_thresh: f32,
    ) -> Self {
        Self {
            inner: BotSort::from_params(BotSortParams {
                common: CommonParams {
                    max_age: track_buffer,
                    min_hits: 3,
                },
                track_thresh,
                match_thresh,
                det_thresh,
                second_match_thresh,
                proximity_thresh,
                appearance_thresh,
            }),
        }
    }

    /// Update the tracker with detections and optional appearance embeddings.
    ///
    /// Args:
    ///     detections (list): List of ([x, y, w, h], score, class_id) tuples.
    ///     embeddings (list): Appearance vectors, one per detection. Pass an empty
    ///         list to track on motion only.
    ///     camera_motion (list, optional): Six affine coefficients
    ///         [a, b, tx, c, d, ty] mapping the previous frame to the current one.
    ///         Defaults to no camera motion.
    ///
    /// Returns:
    ///     list: Active tracks as (track_id, [x, y, w, h], score, class_id) tuples.
    #[pyo3(signature = (detections, embeddings=Vec::new(), camera_motion=None))]
    fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: Vec<Vec<f32>>,
        camera_motion: Option<[f32; 6]>,
    ) -> PyResult<Vec<crate::trackers::common::PyTrackingResult>> {
        if !embeddings.is_empty() && embeddings.len() != detections.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of detections and embeddings must match",
            ));
        }
        let cmc = camera_motion
            .map(|m| CameraMotion::new(m[0], m[1], m[2], m[3], m[4], m[5]))
            .unwrap_or_default();
        let tracks = self
            .inner
            .update_with_camera_motion(detections, &embeddings, &cmc);
        Ok(tracks
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, t.score, t.class_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, s: f32) -> ([f32; 4], f32, i64) {
        ([x, y, w, h], s, 0)
    }

    #[test]
    fn default_params() {
        let t = BotSort::default();
        assert!((t.track_thresh - 0.5).abs() < 1e-6);
        assert_eq!(t.buffer_size, 30);
        assert!((t.appearance_thresh - 0.25).abs() < 1e-6);
    }

    #[test]
    fn empty_detections_return_nothing() {
        let mut t = BotSort::default();
        assert!(t.update(vec![], &[]).is_empty());
    }

    #[test]
    fn single_detection_starts_a_track() {
        let mut t = BotSort::default();
        let tracks = t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 1);
        let (_, tlwh, _, _) = (tracks[0].track_id, tracks[0].tlwh, tracks[0].score, 0);
        assert_eq!(tlwh, [100.0, 100.0, 50.0, 100.0]);
    }

    #[test]
    fn low_score_detection_does_not_start_a_track() {
        // Below det_thresh: no new track is created.
        let mut t = BotSort::default();
        assert!(
            t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.55)], &[])
                .is_empty()
        );
    }

    #[test]
    fn track_keeps_id_across_frames() {
        let mut t = BotSort::default();
        let id = t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        let tracks = t.update(vec![det(104.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn two_objects_get_distinct_ids() {
        let mut t = BotSort::default();
        let tracks = t.update(
            vec![
                det(100.0, 100.0, 50.0, 100.0, 0.9),
                det(400.0, 400.0, 50.0, 100.0, 0.85),
            ],
            &[],
        );
        assert_eq!(tracks.len(), 2);
        assert_ne!(tracks[0].track_id, tracks[1].track_id);
    }

    #[test]
    fn low_confidence_second_stage_keeps_id() {
        // A low-confidence detection is recovered by the second association stage.
        let mut t = BotSort::new(0.6, 30, 0.8, 0.6, 0.5, 0.25);
        let id = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        let tracks = t.update(vec![det(12.0, 12.0, 50.0, 100.0, 0.4)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn lost_track_recovers_from_buffer() {
        let mut t = BotSort::default();
        let id = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        // Miss it twice: it goes lost, then stays alive in the buffer.
        assert!(t.update(vec![], &[]).is_empty());
        assert!(t.update(vec![], &[]).is_empty());
        // Re-detect: the id is recovered.
        let tracks = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn lost_track_dropped_after_buffer() {
        let mut t = BotSort::new(0.5, 2, 0.8, 0.6, 0.5, 0.25);
        let id = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        for _ in 0..4 {
            t.update(vec![], &[]);
        }
        // Beyond the buffer, re-detection starts a fresh id.
        let tracks = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks.len(), 1);
        assert_ne!(tracks[0].track_id, id);
    }

    #[test]
    fn appearance_keeps_id() {
        let mut t = BotSort::default();
        let emb = vec![vec![1.0, 0.0, 0.0]];
        let id = t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)], &emb)[0].track_id;
        let tracks = t.update(vec![det(106.0, 100.0, 50.0, 100.0, 0.9)], &emb);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn dissimilar_appearance_falls_back_to_motion() {
        // A dissimilar embedding is gated out, so a spatially-close detection still
        // matches on motion and keeps the id.
        let mut t = BotSort::default();
        let id = t.update(
            vec![det(100.0, 100.0, 50.0, 100.0, 0.9)],
            &[vec![1.0, 0.0, 0.0]],
        )[0]
        .track_id;
        let tracks = t.update(
            vec![det(104.0, 100.0, 50.0, 100.0, 0.9)],
            &[vec![0.0, 1.0, 0.0]],
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn camera_motion_keeps_id() {
        // Pan the camera right by 200px; the warp moves the prediction with it.
        let mut t = BotSort::default();
        let id = t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        let cmc = CameraMotion::new(1.0, 0.0, 200.0, 0.0, 1.0, 0.0);
        let tracks =
            t.update_with_camera_motion(vec![det(300.0, 100.0, 50.0, 100.0, 0.9)], &[], &cmc);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn without_camera_motion_a_large_shift_starts_a_new_track() {
        let mut t = BotSort::default();
        t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        let tracks = t.update(vec![det(300.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(tracks[0].track_id, 2);
    }

    #[test]
    fn mismatched_embeddings_ignored() {
        // Embedding count does not match detections, so appearance is skipped and
        // tracking still runs on motion.
        let mut t = BotSort::default();
        let tracks = t.update(
            vec![det(100.0, 100.0, 50.0, 100.0, 0.9)],
            &[vec![1.0], vec![2.0]],
        );
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn feature_added_on_a_later_frame() {
        // A track created without an embedding gains one on a later matched frame,
        // exercising the first-feature path.
        let mut t = BotSort::default();
        let id = t.update(vec![det(100.0, 100.0, 50.0, 100.0, 0.9)], &[])[0].track_id;
        let tracks = t.update(
            vec![det(104.0, 100.0, 50.0, 100.0, 0.9)],
            &[vec![1.0, 0.0, 0.0]],
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn reactivated_track_updates_its_feature() {
        // A track with an embedding goes lost, then is re-detected with an embedding,
        // exercising the feature update on re-activation.
        let mut t = BotSort::default();
        let emb = [vec![1.0, 0.0, 0.0]];
        let id = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &emb)[0].track_id;
        assert!(t.update(vec![], &[]).is_empty());
        assert!(t.update(vec![], &[]).is_empty());
        let tracks = t.update(vec![det(10.0, 10.0, 50.0, 100.0, 0.9)], &emb);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, id);
    }

    #[test]
    fn storage_stays_bounded_under_churn() {
        // A fresh object each frame that never re-matches. Lost tracks must age out
        // of the buffer rather than accumulate with the frame count.
        let buffer = 30;
        let mut t = BotSort::new(0.5, buffer, 0.8, 0.6, 0.5, 0.25);
        for f in 0..3000 {
            let x = 5.0 + (f % 100) as f32 * 40.0;
            let _ = t.update(vec![det(x, 10.0, 20.0, 40.0, 0.9)], &[]);
            let stored = t.tracked_stracks.len() + t.lost_stracks.len();
            assert!(
                stored <= buffer + 5,
                "storage grew to {stored} at frame {f}, buffer is {buffer}"
            );
        }
    }
}
