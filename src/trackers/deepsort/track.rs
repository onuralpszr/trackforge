use crate::utils::kalman::{CovarianceMatrix, KalmanFilter, MeasurementVector, StateVector};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: u64,
    pub class_id: i64,
    pub hits: usize,
    pub age: usize,
    pub time_since_update: usize,
    pub state: TrackState,
    pub mean: StateVector,
    pub covariance: CovarianceMatrix,
    pub score: f32,

    /// Features (embeddings) collected during the current update cycle or while tentative.
    /// These are flushed to the metric gallery when appropriate.
    pub features: Vec<Vec<f32>>,

    _n_init: usize,
    _max_age: usize,
}

impl Track {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mean: StateVector,
        covariance: CovarianceMatrix,
        track_id: u64,
        class_id: i64,
        n_init: usize,
        max_age: usize,
        score: f32,
        feature: Vec<f32>,
    ) -> Self {
        Self {
            mean,
            covariance,
            track_id,
            class_id,
            hits: 1,
            age: 1,
            time_since_update: 0,
            state: TrackState::Tentative,
            score,
            features: vec![feature],
            _n_init: n_init,
            _max_age: max_age,
        }
    }

    /// Convert TLWH to (x, y, a, h)
    pub fn tlwh_to_xyah(tlwh: &[f32; 4]) -> MeasurementVector {
        let x = tlwh[0] + tlwh[2] / 2.0;
        let y = tlwh[1] + tlwh[3] / 2.0;
        let a = tlwh[2] / tlwh[3].max(1e-6);
        let h = tlwh[3];
        MeasurementVector::from_vec(vec![x, y, a, h])
    }

    /// Convert (x, y, a, h) to TLWH
    pub fn xyah_to_tlwh(state: &StateVector) -> [f32; 4] {
        let w = state[2] * state[3];
        let h = state[3];
        let x = state[0] - w / 2.0;
        let y = state[1] - h / 2.0;
        [x, y, w, h]
    }

    pub fn to_tlwh(&self) -> [f32; 4] {
        Self::xyah_to_tlwh(&self.mean)
    }

    pub fn predict(&mut self, kf: &KalmanFilter) {
        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = covariance;
        self.age += 1;
        self.time_since_update += 1;
    }

    pub fn update(
        &mut self,
        kf: &KalmanFilter,
        detection: &MeasurementVector,
        score: f32,
        class_id: i64,
        feature: Vec<f32>,
    ) {
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, detection);
        self.mean = mean;
        self.covariance = covariance;
        self.hits += 1;
        self.time_since_update = 0;
        self.score = score;
        self.class_id = class_id;
        self.features.push(feature);

        if self.state == TrackState::Tentative && self.hits >= self._n_init {
            self.state = TrackState::Confirmed;
        }
    }

    pub fn mark_missed(&mut self) {
        if self.state == TrackState::Tentative || self.time_since_update > self._max_age {
            self.state = TrackState::Deleted;
        }
    }

    pub fn is_confirmed(&self) -> bool {
        self.state == TrackState::Confirmed
    }

    pub fn is_tentative(&self) -> bool {
        self.state == TrackState::Tentative
    }

    pub fn is_deleted(&self) -> bool {
        self.state == TrackState::Deleted
    }
}
