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
