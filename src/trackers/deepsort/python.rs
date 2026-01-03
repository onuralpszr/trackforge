use crate::trackers::deepsort::nn_matching::{Metric, NearestNeighborDistanceMetric};
use crate::trackers::deepsort::tracker::DeepSortTracker;
use crate::types::BoundingBox;
use pyo3::prelude::*;

#[pyclass(name = "DeepSort")]
pub struct PyDeepSort {
    tracker: DeepSortTracker,
}

#[pymethods]
impl PyDeepSort {
    #[new]
    #[pyo3(signature = (max_age=70, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100))]
    pub fn new(
        max_age: usize,
        n_init: usize,
        max_iou_distance: f32,
        max_cosine_distance: f32,
        nn_budget: usize,
    ) -> Self {
        let metric = NearestNeighborDistanceMetric::new(
            Metric::Cosine,
            max_cosine_distance,
            Some(nn_budget),
        );
        let tracker = DeepSortTracker::new(metric, max_age, n_init, max_iou_distance);
        Self { tracker }
    }

    /// Update the tracker with detections and embeddings.
    ///
    /// Args:
    ///     detections (List[Tuple[List[float], float, int]]): List of (tlwh, score, class_id).
    ///     embeddings (List[List[float]]): List of appearance embeddings corresponding to detections.
    ///
    /// Returns:
    ///     List[Track]: List of active confirmed tracks.
    pub fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: Vec<Vec<f32>>,
    ) -> PyResult<Vec<PyDeepSortTrack>> {
        // Validation
        if detections.len() != embeddings.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of detections and embeddings must match",
            ));
        }

        let rust_detections: Vec<(BoundingBox, f32, i64)> = detections
            .into_iter()
            .map(|(tlwh, score, cls)| {
                (
                    BoundingBox::new(tlwh[0], tlwh[1], tlwh[2], tlwh[3]),
                    score,
                    cls,
                )
            })
            .collect();

        // Predict
        self.tracker.predict();

        // Update
        self.tracker.update(&rust_detections, &embeddings);

        // Return confirmed confirmed tracks
        let tracks: Vec<PyDeepSortTrack> = self
            .tracker
            .tracks
            .iter()
            // Filter confirmed and active (updated in this frame? Standard DeepSORT usage usually returns all confirmed)
            // But for visualizer we usually want tracks present in the current frame.
            // Let's strictly follow standard: Confirmed tracks that were updated recently.
            // If time_since_update == 0, it was just matched.
            // If time_since_update > 0, it is a prediction/missed.
            // Usually we return current matches.
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .map(|t| PyDeepSortTrack {
                track_id: t.track_id,
                tlwh: t.to_tlwh(),
                score: t.score,
                class_id: t.class_id,
            })
            .collect();

        Ok(tracks)
    }
}

/// Python-exposed Track object
#[pyclass(name = "DeepSortTrack")]
#[derive(Clone)]
pub struct PyDeepSortTrack {
    #[pyo3(get)]
    pub track_id: u64,
    #[pyo3(get)]
    pub tlwh: [f32; 4],
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub class_id: i64,
}
