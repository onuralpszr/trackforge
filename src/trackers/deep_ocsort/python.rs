use super::DeepOcSortTracker;
use crate::trackers::common::{CameraMotion, PyTrackingResult};
use pyo3::prelude::*;

/// Python-exposed Deep OC-SORT tracker.
#[pyclass(name = "DEEPOCSORT")]
pub struct PyDeepOcSort {
    tracker: DeepOcSortTracker,
}

#[pymethods]
impl PyDeepOcSort {
    #[new]
    #[pyo3(signature = (max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2, appearance_weight=0.5, max_cosine_distance=0.2, nn_budget=100))]
    #[allow(clippy::too_many_arguments)]
    /// Initialize the Deep OC-SORT tracker.
    ///
    /// Args:
    ///     max_age (int): Frames a lost track is kept alive. Default: 30.
    ///     min_hits (int): Consecutive matches to confirm a track. Default: 3.
    ///     iou_threshold (float): IoU threshold for matching. Default: 0.3.
    ///     delta_t (int): Observation window for velocity computation. Default: 3.
    ///     inertia (float): Velocity direction-consistency weight in [0, 1]. Default: 0.2.
    ///     appearance_weight (float): Appearance blend weight in [0, 1]. Default: 0.5.
    ///     max_cosine_distance (float): Gate for the appearance term. Default: 0.2.
    ///     nn_budget (int): Maximum appearance features stored per track. Default: 100.
    fn new(
        max_age: usize,
        min_hits: usize,
        iou_threshold: f32,
        delta_t: usize,
        inertia: f32,
        appearance_weight: f32,
        max_cosine_distance: f32,
        nn_budget: usize,
    ) -> Self {
        let tracker = super::build_tracker(
            max_age,
            min_hits,
            iou_threshold,
            delta_t,
            inertia,
            appearance_weight,
            max_cosine_distance,
            nn_budget,
        );
        Self { tracker }
    }

    /// Update the tracker with detections and optional appearance embeddings.
    ///
    /// Args:
    ///     detections (list): List of ([x, y, w, h], score, class_id) tuples.
    ///     embeddings (list): Appearance vectors, one per detection. Pass an empty
    ///         list to track on motion only.
    ///     camera_motion (list, optional): Six affine coefficients
    ///         [a, b, tx, c, d, ty] mapping the previous frame to the current one,
    ///         for moving-camera footage. Defaults to no camera motion.
    ///
    /// Returns:
    ///     list: Confirmed tracks as (track_id, [x, y, w, h], score, class_id) tuples.
    #[pyo3(signature = (detections, embeddings=Vec::new(), camera_motion=None))]
    fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: Vec<Vec<f32>>,
        camera_motion: Option<[f32; 6]>,
    ) -> PyResult<Vec<PyTrackingResult>> {
        if !embeddings.is_empty() && embeddings.len() != detections.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of detections and embeddings must match",
            ));
        }

        let cmc = camera_motion
            .map(|m| CameraMotion::new(m[0], m[1], m[2], m[3], m[4], m[5]))
            .unwrap_or_default();
        let tracks = self
            .tracker
            .update_with_camera_motion(&detections, &embeddings, &cmc);
        Ok(tracks
            .into_iter()
            .map(|t| (t.track_id, t.tlwh, t.score, t.class_id))
            .collect())
    }
}
