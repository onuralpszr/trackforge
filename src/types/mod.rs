//! Common types used across the Trackforge library.
//!
//! This module defines fundamental structures like `BoundingBox`.

/// Axis-aligned bounding box in TLWH (top-left, width, height) format.
///
/// All coordinates are in pixels with the origin at the top-left corner of the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Horizontal position of the top-left corner.
    pub x: f32,
    /// Vertical position of the top-left corner.
    pub y: f32,
    /// Width of the box.
    pub width: f32,
    /// Height of the box.
    pub height: f32,
}

impl BoundingBox {
    /// Create a bounding box from top-left corner coordinates plus dimensions.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}
