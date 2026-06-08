use crate::trackers::deepsort::tracker::DeepSortTracker;
use crate::types::BoundingBox;
use pyo3::prelude::*;

type PyTrackingResult = (u64, [f32; 4], f32, i64);

#[pyclass(name = "DEEPSORT")]
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
        let tracker = super::build_tracker(
            max_age,
            n_init,
            max_iou_distance,
            max_cosine_distance,
            nn_budget,
        );
        Self { tracker }
    }

    /// Update the tracker with detections and embeddings.
    ///
    /// Args:
    ///     detections (`List[Tuple[List[float], float, int]]`): List of (tlwh, score, class_id).
    ///     embeddings (`List[List[float]]`): List of appearance embeddings corresponding to detections.
    ///
    /// Returns:
    ///     `List[Tuple[int, List[float], float, int]]`: List of (track_id, tlwh, score, class_id).
    pub fn update(
        &mut self,
        detections: Vec<([f32; 4], f32, i64)>,
        embeddings: Vec<Vec<f32>>,
    ) -> PyResult<Vec<PyTrackingResult>> {
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

        self.tracker.predict();
        self.tracker.update(&rust_detections, &embeddings);

        let tracks: Vec<PyTrackingResult> = self
            .tracker
            .tracks
            .iter()
            .filter(|t| t.is_confirmed() && t.time_since_update == 0)
            .map(|t| (t.track_id, t.to_tlwh(), t.score, t.class_id))
            .collect();

        Ok(tracks)
    }
}
