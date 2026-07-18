//! Observation-centric Kalman track shared by OC-SORT and Deep OC-SORT.
//!
//! Both trackers keep the same per-track state: the [`KalmanTrack`] filter, the
//! usual lifecycle counters, and a bounded history of past observations used for
//! velocity direction (OCM), re-update on re-association (ORU), and round-2
//! rematching on the last observed box. Deep OC-SORT additionally buffers
//! appearance embeddings in `pending_features`; OC-SORT leaves that buffer empty.
//! Keeping one type here means the observation-centric mechanics live in a single
//! tested place.

use crate::trackers::common::{CameraMotion, KalmanTrack, TrackState};
use crate::utils::geometry::{tlwh_to_xyah, xyah_to_tlwh};
use crate::utils::kalman::{KalmanFilter, MeasurementVector};

/// A tracked object with an observation history.
///
/// Shared by OC-SORT and Deep OC-SORT (each re-exports it under its own name).
#[derive(Debug, Clone)]
pub struct ObsTrack {
    /// Bounding box in TLWH (top-left x, top-left y, width, height) format.
    pub tlwh: [f32; 4],
    /// Detection confidence of the most recent match.
    pub score: f32,
    /// Class label of the most recent match.
    pub class_id: i64,
    /// Unique monotonically increasing track identifier.
    pub track_id: u64,
    /// Current lifecycle state.
    pub state: TrackState,
    /// Total number of detection matches over the track lifetime.
    pub hits: usize,
    /// Consecutive detection matches without interruption (resets on a missed frame).
    pub hit_streak: usize,
    /// Frames elapsed since the last detection match.
    pub time_since_update: usize,
    /// Total frames since track creation.
    pub age: usize,

    kalman: KalmanTrack,
    // Bounded observation history (xyah, frame_id) in insertion order, used for OCM and ORU.
    observations: Vec<(MeasurementVector, usize)>,
    // Appearance embeddings collected since the last gallery flush (Deep OC-SORT only).
    pending_features: Vec<Vec<f32>>,
}

impl ObsTrack {
    /// Create a track from a first detection, optionally seeding an appearance feature.
    pub(crate) fn new(
        tlwh: [f32; 4],
        score: f32,
        class_id: i64,
        track_id: u64,
        frame_id: usize,
        feature: Option<Vec<f32>>,
        kf: &KalmanFilter,
    ) -> Self {
        let xyah = tlwh_to_xyah(&tlwh);
        let kalman = KalmanTrack::initiate(&xyah, kf);

        Self {
            tlwh,
            score,
            class_id,
            track_id,
            state: TrackState::Tentative,
            hits: 1,
            hit_streak: 1,
            time_since_update: 0,
            age: 1,
            kalman,
            observations: vec![(xyah, frame_id)],
            pending_features: feature.into_iter().collect(),
        }
    }

    /// Kalman-predict one step forward, resetting `hit_streak` after a missed frame.
    pub(crate) fn predict(&mut self, kf: &KalmanFilter) {
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.tlwh = self.kalman.predict(kf);
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Standard Kalman update with a matched detection in XYAH form.
    pub(crate) fn update_kf(&mut self, xyah: &MeasurementVector, kf: &KalmanFilter) {
        self.kalman.update(xyah, kf);
        self.tlwh = xyah_to_tlwh(&self.kalman.mean);
    }

    /// Copy a matched detection's box/score/class and bump the hit counters.
    pub(crate) fn record_match(&mut self, tlwh: [f32; 4], score: f32, class_id: i64) {
        self.tlwh = tlwh;
        self.score = score;
        self.class_id = class_id;
        self.hits += 1;
        self.hit_streak += 1;
        self.time_since_update = 0;
    }

    /// Warp the predicted state and observation history by a camera motion transform.
    pub(crate) fn apply_camera_motion(&mut self, cmc: &CameraMotion) {
        self.tlwh = self.kalman.apply_camera_motion(cmc);
        for (obs, _) in &mut self.observations {
            cmc.apply_observation(obs);
        }
    }

    /// OCV: normalised `[dy, dx]` velocity direction over the last `delta_t` frames.
    ///
    /// Returns `None` when fewer than two observations are available.
    pub(crate) fn obs_direction(&self, delta_t: usize) -> Option<[f32; 2]> {
        let n = self.observations.len();
        if n < 2 {
            return None;
        }
        let anchor_idx = n.saturating_sub(delta_t + 1);
        let (obs_old, _) = &self.observations[anchor_idx];
        let (obs_new, _) = &self.observations[n - 1];
        let dy = obs_new[1] - obs_old[1];
        let dx = obs_new[0] - obs_old[0];
        let norm = (dy * dy + dx * dx).sqrt() + 1e-6;
        Some([dy / norm, dx / norm])
    }

    /// Last observed centre `(cx, cy, a, h)`; used by the OCM bonus and round-2 re-match.
    pub(crate) fn last_observation(&self) -> &MeasurementVector {
        &self
            .observations
            .last()
            .expect("observations is non-empty by invariant")
            .0
    }

    /// ORU: replay interpolated observations to correct KF drift after re-association.
    pub(crate) fn our_re_update(
        &mut self,
        current_xyah: &MeasurementVector,
        current_frame: usize,
        kf: &KalmanFilter,
    ) {
        let (last_obs, last_frame) = self
            .observations
            .last()
            .expect("observations is non-empty by invariant");
        let gap = (current_frame as isize - *last_frame as isize).max(1) as usize;
        if gap <= 1 {
            return;
        }

        // Interpolate in TLWH space (matching the reference), replaying predict/update.
        let last_tlwh = xyah_to_tlwh(last_obs);
        let current_tlwh = xyah_to_tlwh(current_xyah);
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

        self.kalman.mean = mean;
        self.kalman.covariance = covariance;
        self.tlwh = xyah_to_tlwh(&self.kalman.mean);
    }

    /// Record a new observation, keeping the history bounded to `max_obs` entries.
    pub(crate) fn push_observation(
        &mut self,
        xyah: MeasurementVector,
        frame_id: usize,
        max_obs: usize,
    ) {
        self.observations.push((xyah, frame_id));
        if self.observations.len() > max_obs {
            self.observations.remove(0);
        }
    }

    /// Buffer an appearance embedding to be flushed into the gallery this frame.
    pub(crate) fn push_feature(&mut self, feature: Vec<f32>) {
        self.pending_features.push(feature);
    }

    /// Drain the buffered embeddings collected since the last flush.
    pub(crate) fn take_features(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.pending_features)
    }

    pub(crate) fn mark_deleted(&mut self) {
        self.state = TrackState::Deleted;
    }

    /// Whether the track has been confirmed and is returned to callers.
    pub fn is_confirmed(&self) -> bool {
        self.state == TrackState::Confirmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_track() -> ObsTrack {
        let kf = KalmanFilter::default();
        ObsTrack::new(
            [100.0, 100.0, 50.0, 100.0],
            0.9,
            0,
            1,
            1,
            Some(vec![1.0, 0.0]),
            &kf,
        )
    }

    #[test]
    fn obs_direction_needs_two_observations() {
        let mut track = make_track();
        assert!(track.obs_direction(3).is_none());
        track.push_observation(tlwh_to_xyah(&[110.0, 100.0, 50.0, 100.0]), 2, 4);
        let dir = track.obs_direction(3).unwrap();
        assert!(dir[0].abs() < 0.01 && (dir[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn push_observation_is_bounded() {
        let mut track = make_track();
        for frame in 2..12 {
            track.push_observation(tlwh_to_xyah(&[100.0, 100.0, 50.0, 100.0]), frame, 4);
        }
        assert!(track.obs_direction(3).is_some());
    }

    #[test]
    fn our_re_update_stays_finite_after_gap() {
        let kf = KalmanFilter::default();
        let mut track = make_track();
        for _ in 0..5 {
            track.predict(&kf);
        }
        track.our_re_update(&tlwh_to_xyah(&[130.0, 100.0, 50.0, 100.0]), 7, &kf);
        assert!(track.tlwh.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn take_features_drains_buffer() {
        let mut track = make_track();
        track.push_feature(vec![0.5, 0.5]);
        // new() seeded one feature, push_feature added another.
        assert_eq!(track.take_features().len(), 2);
        assert!(track.take_features().is_empty());
    }

    #[test]
    fn record_match_bumps_counters() {
        let mut track = make_track();
        track.predict(&KalmanFilter::default());
        track.record_match([120.0, 100.0, 50.0, 100.0], 0.8, 2);
        assert_eq!(track.tlwh, [120.0, 100.0, 50.0, 100.0]);
        assert_eq!(track.class_id, 2);
        assert_eq!(track.hits, 2);
        assert_eq!(track.time_since_update, 0);
    }
}
