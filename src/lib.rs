pub mod trackers;
pub mod traits;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[cfg(feature = "python")]
/// A Python module implemented in Rust.
#[pymodule]
fn trackforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
