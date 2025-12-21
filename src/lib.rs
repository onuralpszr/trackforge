pub mod trackers;
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
fn trackforge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    return Ok(());
}
