//! Building blocks shared across the trackers.
//!
//! [`TrackState`] is the common confirm/delete lifecycle used by the IoU and
//! appearance trackers, and [`KalmanTrack`] wraps the per-track Kalman state with
//! the predict/update mechanics they all repeat. [`cmc`] holds the shared camera
//! motion compensation transform.

pub mod association;
pub mod cmc;
pub mod obs_track;
pub mod params;

pub use cmc::CameraMotion;
pub use obs_track::ObsTrack;
pub use params::CommonParams;

use crate::utils::geometry::xyah_to_tlwh;

/// A track as returned to Python: `(track_id, tlwh, score, class_id)`.
///
/// Every Python binding maps its confirmed tracks into this shape, so the type is
/// shared here rather than redeclared per tracker.
#[cfg(feature = "python")]
pub type PyTrackingResult = (u64, [f32; 4], f32, i64);
use crate::utils::kalman::{CovarianceMatrix, KalmanFilter, MeasurementVector, StateVector};

/// Lifecycle state of a track.
///
/// A track starts [`Tentative`](TrackState::Tentative), becomes
/// [`Confirmed`](TrackState::Confirmed) once it has accumulated enough matches, and
/// is [`Deleted`](TrackState::Deleted) when it ages out or fails confirmation.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum TrackState {
    /// Newly created; not yet confirmed by enough matches.
    Tentative,
    /// Confirmed active track returned to callers.
    Confirmed,
    /// Marked for removal.
    Deleted,
}

/// Per-track Kalman state plus the predict/update steps every tracker repeats.
///
/// Holds the 8-dimensional mean and covariance and applies the shared
/// [`KalmanFilter`]. The owning track keeps its own box, score, and lifecycle
/// fields and delegates the filtering to this type.
#[derive(Debug, Clone)]
pub struct KalmanTrack {
    /// Kalman filter state mean (`[x, y, a, h, vx, vy, va, vh]`).
    pub mean: StateVector,
    /// Kalman filter state covariance.
    pub covariance: CovarianceMatrix,
}

impl KalmanTrack {
    /// Initialise the filter from a first measurement in XYAH form.
    pub fn initiate(measurement: &MeasurementVector, kf: &KalmanFilter) -> Self {
        let (mean, covariance) = kf.initiate(measurement);
        Self { mean, covariance }
    }

    /// Run one Kalman prediction step and return the refreshed TLWH box.
    pub fn predict(&mut self, kf: &KalmanFilter) -> [f32; 4] {
        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = covariance;
        xyah_to_tlwh(&self.mean)
    }

    /// Correct the state with a matched measurement in XYAH form.
    pub fn update(&mut self, measurement: &MeasurementVector, kf: &KalmanFilter) {
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, measurement);
        self.mean = mean;
        self.covariance = covariance;
    }

    /// Warp the Kalman state by a camera motion transform and return the new TLWH box.
    pub fn apply_camera_motion(&mut self, cmc: &CameraMotion) -> [f32; 4] {
        cmc.apply_state(&mut self.mean, &mut self.covariance);
        xyah_to_tlwh(&self.mean)
    }
}
