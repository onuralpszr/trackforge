#![doc = include_str!("README.md")]

use crate::trackers::byte_track::TrackState;
use crate::trackers::common::{CameraMotion, CommonParams, KalmanTrack};
use crate::utils::features::{cosine_distance, l2_normalize};
use crate::utils::geometry::tlwh_to_xyah;
use crate::utils::kalman::KalmanFilter;

/// Adaptive exponential moving-average base for the per-track appearance feature.
const FEATURE_ALPHA: f32 = 0.95;
/// Cost weight of the confidence-projection term.
const CONF_WEIGHT: f32 = 0.10;
/// Cost weight of the velocity-direction term.
const ANGLE_WEIGHT: f32 = 0.05;
/// IoU below which a pair is forbidden regardless of the fused cost.
const IOU_GATE: f32 = 0.10;

/// Settings for [`TrackTrack`].
///
/// Shared lifecycle fields live in [`CommonParams`]; TrackTrack maps its lost buffer
/// onto `common.max_age` and its confirmation length onto `common.min_hits`. The rest
/// are TrackTrack specific. Build it with [`TrackTrackParams::default`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackTrackParams {
    /// Shared lifecycle settings. `common.max_age` is the lost buffer, `common.min_hits`
    /// is how many matched frames confirm a new track.
    pub common: CommonParams,

    /// Score above which a detection is high confidence and matched first. Lower scores
    /// are still offered to tracked tracks in the same pass, carrying a penalty.
    pub det_thresh: f32,

    /// Association cost gate. A track and a detection match only when their fused cost is
    /// below this. Lower is stricter.
    pub match_thresh: f32,

    /// Smallest score a leftover detection needs before it may start a new track.
    pub init_thresh: f32,

    /// Overlap gate for track-aware initialization. A leftover detection is dropped if it
    /// overlaps an existing active track, or a more confident leftover, by more than this
    /// IoU. This is a maximum IoU.
    pub tai_thresh: f32,

    /// Extra cost added to low confidence detections during association, so they only win
    /// a match when nothing better is available.
    pub penalty_low: f32,

    /// How much the association cost gate tightens on each round of the track-perspective
    /// matching loop.
    pub reduce_step: f32,
}

impl Default for TrackTrackParams {
    fn default() -> Self {
        Self {
            common: CommonParams {
                max_age: 30,
                min_hits: 3,
            },
            det_thresh: 0.6,
            match_thresh: 0.7,
            init_thresh: 0.7,
            tai_thresh: 0.55,
            penalty_low: 0.20,
            reduce_step: 0.05,
        }
    }
}

/// A detection with an optional appearance embedding.
struct Det {
    tlwh: [f32; 4],
    score: f32,
    class_id: i64,
    feature: Option<Vec<f32>>,
}

/// A single object tracked by TrackTrack.
#[derive(Debug, Clone)]
pub struct Track {
    /// Bounding box in TLWH (top-left x, top-left y, width, height) format.
    pub tlwh: [f32; 4],
    /// Detection confidence of the most recent match.
    pub score: f32,
    /// Class label of the most recent match.
    pub class_id: i64,
    /// Unique track identifier (0 until the track is activated).
    pub track_id: u64,
    /// Lifecycle state.
    pub state: TrackState,

    prev_score: f32,
    hits: usize,
    end_frame: usize,
    kalman: KalmanTrack,
    smooth_feat: Option<Vec<f32>>,
}

impl Track {
    fn from_det(det: &Det, kf: &KalmanFilter, frame_id: usize, track_id: u64) -> Self {
        Self {
            tlwh: det.tlwh,
            score: det.score,
            class_id: det.class_id,
            track_id,
            state: TrackState::New,
            prev_score: det.score,
            hits: 1,
            end_frame: frame_id,
            kalman: KalmanTrack::initiate(&tlwh_to_xyah(&det.tlwh), kf),
            smooth_feat: det.feature.as_ref().map(|f| l2_normalize(f)),
        }
    }

    fn predict(&mut self, kf: &KalmanFilter) {
        if self.state != TrackState::Tracked {
            self.kalman.mean[7] = 0.0;
        }
        self.tlwh = self.kalman.predict(kf);
    }

    fn apply_camera_motion(&mut self, cmc: &CameraMotion) {
        self.tlwh = self.kalman.apply_camera_motion(cmc);
    }

    fn update(&mut self, det: &Det, kf: &KalmanFilter, frame_id: usize, min_hits: usize) {
        self.kalman.update(&tlwh_to_xyah(&det.tlwh), kf);
        self.tlwh = det.tlwh;
        self.prev_score = self.score;
        self.score = det.score;
        self.class_id = det.class_id;
        self.hits += 1;
        self.end_frame = frame_id;
        self.state = if self.hits >= min_hits {
            TrackState::Tracked
        } else {
            TrackState::New
        };
        if let Some(f) = &det.feature {
            self.update_features(f);
        }
    }

    /// Confidence projected one step ahead, `score + (score - prev_score)`.
    fn projected_score(&self) -> f32 {
        self.score + (self.score - self.prev_score)
    }

    fn center(&self) -> (f32, f32) {
        (
            self.tlwh[0] + self.tlwh[2] * 0.5,
            self.tlwh[1] + self.tlwh[3] * 0.5,
        )
    }

    fn update_features(&mut self, feature: &[f32]) {
        let feature = l2_normalize(feature);
        match &mut self.smooth_feat {
            None => self.smooth_feat = Some(feature),
            Some(sf) => {
                let beta = FEATURE_ALPHA + (1.0 - FEATURE_ALPHA) * (1.0 - self.score);
                for (s, f) in sf.iter_mut().zip(&feature) {
                    *s = beta * *s + (1.0 - beta) * f;
                }
                *sf = l2_normalize(sf);
            }
        }
    }
}

/// Convert a TLWH box to `[x1, y1, x2, y2]`.
fn xyxy(b: [f32; 4]) -> [f32; 4] {
    [b[0], b[1], b[0] + b[2], b[1] + b[3]]
}

/// Plain IoU of two TLWH boxes.
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let a = xyxy(a);
    let b = xyxy(b);
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area_a = ((a[2] - a[0]) * (a[3] - a[1])).max(0.0);
    let area_b = ((b[2] - b[0]) * (b[3] - b[1])).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Height-modulated IoU: plain IoU scaled by the 1-D vertical overlap ratio.
fn hmiou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ax = xyxy(a);
    let bx = xyxy(b);
    let h_inter = ax[3].min(bx[3]) - ax[1].max(bx[1]);
    let h_union = ax[3].max(bx[3]) - ax[1].min(bx[1]);
    let h_iou = if h_union > 0.0 {
        h_inter / h_union
    } else {
        0.0
    };
    h_iou * iou(a, b)
}

/// Track-perspective mutual-nearest-neighbour matching under a fixed threshold.
///
/// Returns the accepted `(track, det)` pairs among the still-active rows and columns.
fn associate(
    cost: &[Vec<f32>],
    thresh: f32,
    row_active: &[bool],
    col_active: &[bool],
) -> Vec<(usize, usize)> {
    let n = row_active.len();
    let m = col_active.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }

    // Each active track's best active detection.
    let best_det: Vec<usize> = (0..n)
        .map(|i| {
            (0..m)
                .filter(|&j| col_active[j])
                .min_by(|&a, &b| cost[i][a].total_cmp(&cost[i][b]))
                .unwrap_or(0)
        })
        .collect();
    // Each active detection's best active track.
    let best_track: Vec<usize> = (0..m)
        .map(|j| {
            (0..n)
                .filter(|&i| row_active[i])
                .min_by(|&a, &b| cost[a][j].total_cmp(&cost[b][j]))
                .unwrap_or(0)
        })
        .collect();

    let mut matches = Vec::new();
    for i in 0..n {
        if !row_active[i] {
            continue;
        }
        let j = best_det[i];
        if col_active[j] && best_track[j] == i && cost[i][j] < thresh {
            matches.push((i, j));
        }
    }
    matches
}

/// The full track-perspective association: iterate [`associate`], tightening the gate by
/// `reduce_step` each round and removing matched rows and columns, until a round is empty.
fn iterative_assignment(
    cost: &[Vec<f32>],
    n_tracks: usize,
    n_dets: usize,
    match_thresh: f32,
    reduce_step: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n = n_tracks;
    let m = n_dets;
    let mut row_active = vec![true; n];
    let mut col_active = vec![true; m];
    let mut matches = Vec::new();
    let mut thresh = match_thresh;

    loop {
        let round = associate(cost, thresh, &row_active, &col_active);
        thresh -= reduce_step;
        if round.is_empty() {
            break;
        }
        for (i, j) in round {
            row_active[i] = false;
            col_active[j] = false;
            matches.push((i, j));
        }
    }

    let u_tracks = (0..n).filter(|&i| row_active[i]).collect();
    let u_dets = (0..m).filter(|&j| col_active[j]).collect();
    (matches, u_tracks, u_dets)
}

/// TrackTrack tracker.
///
/// See the module documentation for the algorithm. The two contributions are a
/// track-perspective association and a track-aware initialization. Appearance is used
/// when embeddings are supplied, otherwise the tracker runs on motion only.
pub struct TrackTrack {
    tracks: Vec<Track>,
    frame_id: usize,
    kalman_filter: KalmanFilter,
    next_id: u64,
    max_age: usize,
    min_hits: usize,
    det_thresh: f32,
    match_thresh: f32,
    init_thresh: f32,
    tai_thresh: f32,
    penalty_low: f32,
    reduce_step: f32,
}

impl TrackTrack {
    /// Create a TrackTrack tracker from a [`TrackTrackParams`].
    pub fn from_params(params: TrackTrackParams) -> Self {
        Self {
            tracks: Vec::new(),
            frame_id: 0,
            kalman_filter: KalmanFilter::default(),
            next_id: 1,
            max_age: params.common.max_age,
            min_hits: params.common.min_hits,
            det_thresh: params.det_thresh,
            match_thresh: params.match_thresh,
            init_thresh: params.init_thresh,
            tai_thresh: params.tai_thresh,
            penalty_low: params.penalty_low,
            reduce_step: params.reduce_step,
        }
    }

    /// Create a TrackTrack tracker with the default parameters.
    pub fn new() -> Self {
        Self::from_params(TrackTrackParams::default())
    }

    /// Update the tracker with the current frame's detections and optional embeddings.
    ///
    /// Pass an empty embeddings slice to track on motion only. Returns the confirmed
    /// tracks active in this frame.
    pub fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: &[Vec<f32>],
    ) -> Vec<Track> {
        self.update_with_camera_motion(detections, embeddings, &CameraMotion::identity())
    }

    /// Update the tracker, first warping track predictions by `camera_motion`.
    pub fn update_with_camera_motion(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: &[Vec<f32>],
        camera_motion: &CameraMotion,
    ) -> Vec<Track> {
        self.frame_id += 1;
        let use_reid = !embeddings.is_empty() && embeddings.len() == detections.len();

        let dets: Vec<Det> = detections
            .into_iter()
            .enumerate()
            .map(|(i, (tlwh, score, class_id))| Det {
                tlwh,
                score,
                class_id,
                feature: if use_reid {
                    Some(embeddings[i].clone())
                } else {
                    None
                },
            })
            .collect();

        let high_idx: Vec<usize> = (0..dets.len())
            .filter(|&i| dets[i].score > self.det_thresh)
            .collect();
        let low_idx: Vec<usize> = (0..dets.len())
            .filter(|&i| dets[i].score <= self.det_thresh)
            .collect();

        // Split existing tracks: tracked or lost participate in stage one, new (tentative)
        // tracks participate in stage two.
        let mut tracked_lost: Vec<Track> = Vec::new();
        let mut tentative: Vec<Track> = Vec::new();
        for t in self.tracks.drain(..) {
            match t.state {
                TrackState::New => tentative.push(t),
                TrackState::Tracked | TrackState::Lost => tracked_lost.push(t),
                TrackState::Removed => {}
            }
        }

        // Predict and warp all tracks.
        let warp = !camera_motion.is_identity();
        for t in tracked_lost.iter_mut().chain(tentative.iter_mut()) {
            t.predict(&self.kalman_filter);
            if warp {
                t.apply_camera_motion(camera_motion);
            }
        }

        // Stage one: high and low detections against tracked and lost tracks.
        let stage1_pool: Vec<usize> = high_idx.iter().chain(&low_idx).copied().collect();
        let n_high = high_idx.len();
        let cost1 = self.build_cost(&tracked_lost, &dets, &stage1_pool, n_high, use_reid);
        let (m1, u_tracks1, u_dets1) = iterative_assignment(
            &cost1,
            tracked_lost.len(),
            stage1_pool.len(),
            self.match_thresh,
            self.reduce_step,
        );

        for (t, d) in m1 {
            let det = &dets[stage1_pool[d]];
            tracked_lost[t].update(det, &self.kalman_filter, self.frame_id, self.min_hits);
        }
        for t in u_tracks1 {
            tracked_lost[t].state = TrackState::Lost;
        }

        // Detections that were high confidence and stayed unmatched.
        let high_left: Vec<usize> = u_dets1
            .iter()
            .copied()
            .filter(|&d| d < n_high)
            .map(|d| stage1_pool[d])
            .collect();

        // Stage two: leftover high detections against tentative tracks.
        let cost2 = self.build_cost(&tentative, &dets, &high_left, high_left.len(), use_reid);
        let (m2, u_tracks2, u_dets2) = iterative_assignment(
            &cost2,
            tentative.len(),
            high_left.len(),
            self.match_thresh,
            self.reduce_step,
        );

        for (t, d) in m2 {
            let det = &dets[high_left[d]];
            tentative[t].update(det, &self.kalman_filter, self.frame_id, self.min_hits);
        }
        for t in u_tracks2 {
            tentative[t].state = TrackState::Removed;
        }

        // Age out lost tracks past the buffer.
        for t in tracked_lost.iter_mut().chain(tentative.iter_mut()) {
            if self.frame_id.saturating_sub(t.end_frame) > self.max_age {
                t.state = TrackState::Removed;
            }
        }

        // Recombine surviving tracks.
        self.tracks = tracked_lost
            .into_iter()
            .chain(tentative)
            .filter(|t| t.state != TrackState::Removed)
            .collect();

        // Track-aware initialization of the still-unmatched high detections.
        let init_candidates: Vec<usize> = u_dets2.iter().map(|&d| high_left[d]).collect();
        self.init_tracks(&dets, &init_candidates);

        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Tracked)
            .cloned()
            .collect()
    }

    /// Build the fused cost matrix (`n_tracks x n_dets`) for the given detection subset,
    /// applying the low-confidence penalty to columns at or past `n_high`.
    fn build_cost(
        &self,
        tracks: &[Track],
        dets: &[Det],
        det_idx: &[usize],
        n_high: usize,
        use_reid: bool,
    ) -> Vec<Vec<f32>> {
        let (w_iou, w_cos) = if use_reid { (0.5, 0.5) } else { (1.0, 0.0) };
        tracks
            .iter()
            .map(|t| {
                det_idx
                    .iter()
                    .enumerate()
                    .map(|(col, &di)| {
                        let d = &dets[di];
                        let iou_sim = iou(t.tlwh, d.tlwh);
                        if iou_sim <= IOU_GATE {
                            return 1.0;
                        }
                        let iou_dist = 1.0 - hmiou(t.tlwh, d.tlwh);
                        let cos_dist = if use_reid {
                            match (&t.smooth_feat, &d.feature) {
                                (Some(tf), Some(df)) => cosine_distance(tf, df).clamp(0.0, 1.0),
                                _ => 1.0,
                            }
                        } else {
                            0.0
                        };
                        let conf = (t.projected_score() - d.score).abs();
                        let angle = angle_cost(t, d);
                        let mut cost = w_iou * iou_dist
                            + w_cos * cos_dist
                            + CONF_WEIGHT * conf
                            + ANGLE_WEIGHT * angle;
                        if col >= n_high {
                            cost += self.penalty_low;
                        }
                        cost.clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Track-aware initialization. A leftover high detection starts a new track only if it
    /// clears `init_thresh` and does not overlap an active track or a more confident
    /// leftover by more than `tai_thresh`.
    fn init_tracks(&mut self, dets: &[Det], candidates: &[usize]) {
        let active_boxes: Vec<[f32; 4]> = self
            .tracks
            .iter()
            .filter(|t| t.state == TrackState::Tracked || t.state == TrackState::New)
            .map(|t| t.tlwh)
            .collect();

        let mut allow: Vec<bool> = candidates
            .iter()
            .map(|&d| dets[d].score > self.init_thresh)
            .collect();

        for idx in 0..candidates.len() {
            if !allow[idx] {
                continue;
            }
            let box_idx = dets[candidates[idx]].tlwh;
            // Do not spawn a duplicate of an already active track.
            if active_boxes
                .iter()
                .any(|tb| iou(box_idx, *tb) > self.tai_thresh)
            {
                allow[idx] = false;
                continue;
            }
            // Suppress lower-score leftovers that overlap this one.
            for jdx in 0..candidates.len() {
                if idx != jdx
                    && allow[jdx]
                    && dets[candidates[idx]].score > dets[candidates[jdx]].score
                    && iou(box_idx, dets[candidates[jdx]].tlwh) > self.tai_thresh
                {
                    allow[jdx] = false;
                }
            }
        }

        for (idx, &d) in candidates.iter().enumerate() {
            if allow[idx] {
                let track =
                    Track::from_det(&dets[d], &self.kalman_filter, self.frame_id, self.next_id);
                self.next_id += 1;
                self.tracks.push(track);
            }
        }
    }
}

impl Default for TrackTrack {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed TrackTrack tracker.
#[cfg(feature = "python")]
#[pyclass(name = "TRACKTRACK")]
pub struct PyTrackTrack {
    inner: TrackTrack,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrackTrack {
    #[new]
    #[pyo3(signature = (det_thresh=0.6, match_thresh=0.7, track_buffer=30, min_hits=3, init_thresh=0.7, tai_thresh=0.55, penalty_low=0.2, reduce_step=0.05))]
    /// Initialize the TrackTrack tracker.
    ///
    /// Args:
    ///     det_thresh (float): Score above which a detection is high confidence. Default: 0.6.
    ///     match_thresh (float): Association cost gate, lower is stricter. Default: 0.7.
    ///     track_buffer (int): Frames a lost track is kept alive. Default: 30.
    ///     min_hits (int): Matched frames needed to confirm a new track. Default: 3.
    ///     init_thresh (float): Smallest score to start a new track. Default: 0.7.
    ///     tai_thresh (float): Overlap gate for track-aware initialization. Default: 0.55.
    ///     penalty_low (float): Extra cost on low confidence detections. Default: 0.2.
    ///     reduce_step (float): How much the cost gate tightens per round. Default: 0.05.
    #[allow(clippy::too_many_arguments)]
    fn new(
        det_thresh: f32,
        match_thresh: f32,
        track_buffer: usize,
        min_hits: usize,
        init_thresh: f32,
        tai_thresh: f32,
        penalty_low: f32,
        reduce_step: f32,
    ) -> Self {
        Self {
            inner: TrackTrack::from_params(TrackTrackParams {
                common: CommonParams {
                    max_age: track_buffer,
                    min_hits,
                },
                det_thresh,
                match_thresh,
                init_thresh,
                tai_thresh,
                penalty_low,
                reduce_step,
            }),
        }
    }

    /// Update the tracker with detections and optional appearance embeddings.
    ///
    /// Args:
    ///     detections (list): List of ([x, y, w, h], score, class_id) tuples.
    ///     embeddings (list): Appearance vectors, one per detection, or an empty list
    ///         to track on motion only.
    ///     camera_motion (list, optional): Six affine coefficients [a, b, tx, c, d, ty]
    ///         mapping the previous frame to the current one. Defaults to none.
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

/// Simplified velocity-direction cost: the angle between a track's center velocity and the
/// direction from the track to the detection, normalized to `[0, 1]` and scaled by the
/// detection score.
fn angle_cost(t: &Track, d: &Det) -> f32 {
    let vx = t.kalman.mean[4];
    let vy = t.kalman.mean[5];
    let (tcx, tcy) = t.center();
    let dcx = d.tlwh[0] + d.tlwh[2] * 0.5;
    let dcy = d.tlwh[1] + d.tlwh[3] * 0.5;
    let dir = (dcx - tcx, dcy - tcy);
    let nv = (vx * vx + vy * vy).sqrt();
    let nd = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
    if nv < 1e-5 || nd < 1e-5 {
        return 0.0;
    }
    let cos = ((vx * dir.0 + vy * dir.1) / (nv * nd)).clamp(-1.0, 1.0);
    (cos.acos().abs() / std::f32::consts::PI) * d.score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, s: f32) -> ([f32; 4], f32, i64) {
        ([x, y, w, h], s, 0)
    }

    #[test]
    fn confirms_after_min_hits() {
        let mut t = TrackTrack::new(); // min_hits = 3
        let d = det(100.0, 100.0, 50.0, 100.0, 0.9);
        assert!(t.update(vec![d], &[]).is_empty(), "frame 1 still tentative");
        assert!(t.update(vec![d], &[]).is_empty(), "frame 2 still tentative");
        let out = t.update(vec![d], &[]);
        assert_eq!(out.len(), 1, "confirmed on the third match");
        assert_eq!(out[0].track_id, 1);
    }

    #[test]
    fn empty_detections_return_nothing() {
        let mut t = TrackTrack::new();
        assert!(t.update(vec![], &[]).is_empty());
    }

    #[test]
    fn track_keeps_id_across_frames() {
        let mut t = TrackTrack::new();
        let base = det(100.0, 100.0, 50.0, 100.0, 0.9);
        t.update(vec![base], &[]);
        t.update(vec![base], &[]);
        let id = t.update(vec![base], &[])[0].track_id;
        let out = t.update(vec![det(104.0, 100.0, 50.0, 100.0, 0.9)], &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, id);
    }

    #[test]
    fn recovers_id_after_short_occlusion() {
        let mut t = TrackTrack::new();
        let d = det(100.0, 100.0, 50.0, 100.0, 0.9);
        for _ in 0..3 {
            t.update(vec![d], &[]);
        }
        let id = t.update(vec![d], &[])[0].track_id;
        // Miss two frames, then re-detect at the same place.
        assert!(t.update(vec![], &[]).is_empty());
        assert!(t.update(vec![], &[]).is_empty());
        let out = t.update(vec![d], &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, id, "id recovered from the lost buffer");
    }

    #[test]
    fn two_separated_objects_get_distinct_ids() {
        let mut t = TrackTrack::new();
        let dets = vec![
            det(50.0, 50.0, 40.0, 90.0, 0.9),
            det(400.0, 400.0, 40.0, 90.0, 0.9),
        ];
        for _ in 0..3 {
            t.update(dets.clone(), &[]);
        }
        let out = t.update(dets, &[]);
        assert_eq!(out.len(), 2);
        assert_ne!(out[0].track_id, out[1].track_id);
    }

    #[test]
    fn track_aware_init_suppresses_a_duplicate() {
        // Two heavily overlapping high-confidence detections on the first frame. Track-
        // aware initialization should start only one track, not two.
        let mut t = TrackTrack::new();
        let dets = vec![
            det(100.0, 100.0, 60.0, 120.0, 0.9),
            det(104.0, 102.0, 60.0, 120.0, 0.8),
        ];
        for _ in 0..3 {
            t.update(dets.clone(), &[]);
        }
        let out = t.update(dets, &[]);
        assert_eq!(out.len(), 1, "the overlapping duplicate was suppressed");
    }

    #[test]
    fn appearance_keeps_id_when_embeddings_supplied() {
        let mut t = TrackTrack::new();
        let emb = vec![vec![1.0, 0.0, 0.0]];
        let d = det(100.0, 100.0, 50.0, 100.0, 0.9);
        for _ in 0..3 {
            t.update(vec![d], &emb);
        }
        let id = t.update(vec![d], &emb)[0].track_id;
        let out = t.update(vec![det(105.0, 100.0, 50.0, 100.0, 0.9)], &emb);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, id);
    }
}
