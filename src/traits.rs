use crate::types::BoundingBox;
use image::DynamicImage;
use std::error::Error;

/// A detection passed to a tracker: `(tlwh, score, class_id)`.
///
/// The box is in TLWH (top-left x, top-left y, width, height) format.
pub type Detection = ([f32; 4], f32, i64);

/// Common interface for the detection-only trackers.
///
/// SORT, ByteTrack, and OC-SORT all consume a frame's detections and return the
/// active tracks, so generic code can drive any of them through this trait.
/// DeepSORT needs the source frame to extract appearance features and therefore
/// keeps its own `update(&image, detections)` method instead of this trait.
///
/// ```
/// use trackforge::traits::{Detection, Tracker};
/// use trackforge::trackers::sort::Sort;
///
/// fn run_one<T: Tracker>(tracker: &mut T, dets: Vec<Detection>) -> Vec<T::Track> {
///     tracker.update(dets)
/// }
///
/// let mut tracker = Sort::new(1, 1, 0.3);
/// let tracks = run_one(&mut tracker, vec![([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
/// assert_eq!(tracks.len(), 1);
/// ```
pub trait Tracker {
    /// The track type this tracker yields.
    type Track;

    /// Update the tracker with the current frame's detections and return the
    /// active tracks for this frame.
    fn update(&mut self, detections: Vec<Detection>) -> Vec<Self::Track>;
}

/// Trait for extracting appearance features (embeddings) from images.
///
/// This allows decoupling the tracker logic (DeepSORT) from the model execution
/// (ONNX, PyTorch via Python, etc.).
pub trait AppearanceExtractor {
    /// Extract features for a list of bounding boxes from a given image.
    ///
    /// # Arguments
    /// * `image` - The full frame image.
    /// * `bboxes` - List of bounding boxes to extract features for.
    ///
    /// # Returns
    /// A vector of feature vectors (embeddings), one for each bounding box.
    fn extract(
        &mut self,
        image: &DynamicImage,
        bboxes: &[BoundingBox],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;
}
