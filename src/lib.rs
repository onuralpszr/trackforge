//! Trackforge is a unified, high-performance computer vision tracking library, implemented in Rust and exposed as a Python package.
//!
//! It provides state-of-the-art tracking algorithms like **ByteTrack**, optimized for speed and ease of use in both Rust and Python environments.
//!
//! ## Features
//!
//! - **High Performance**: Native Rust implementation for maximum speed and memory safety.
//! - **Python Bindings**: Seamless integration with the Python ecosystem using `pyo3`.
//! - **Unified API**: Consistent interface for tracking tasks across both languages.
//! - **ByteTrack**: Robust multi-object tracking using Kalman filters and IoU matching.
//!
//! ## Usage (Rust)
//!
//! ```rust
//! use trackforge::trackers::byte_track::ByteTrack;
//!
//! // Initialize ByteTrack
//! let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);
//!
//! // Detections: Vec<([f32; 4], f32, i64)>
//! let detections = vec![
//!     ([100.0, 100.0, 50.0, 100.0], 0.9, 0),
//! ];
//!
//! // Update
//! let tracks = tracker.update(detections);
//!
//! for t in tracks {
//!     println!("ID: {}, Box: {:?}", t.track_id, t.tlwh);
//! }
//! ```

pub mod trackers;
pub mod traits;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn trackforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<trackers::byte_track::PyByteTrack>()?;
    Ok(())
}
