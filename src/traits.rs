use crate::types::BoundingBox;
use image::DynamicImage;
use std::error::Error;

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
        &self,
        image: &DynamicImage,
        bboxes: &[BoundingBox],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;
}
