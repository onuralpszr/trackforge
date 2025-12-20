# Trackforge


> [!IMPORTANT]
> **This project is currently under active development.** APIs and features are subject to change.

**Trackforge** is a unified, high-performance computer vision tracking library, implemented in Rust and exposed as a Python package.

## Features
- 🚀 **High Performance**: Built with Rust for maximum speed and safety.
- 🐍 **Python Bindings**: Seamless integration with the Python ecosystem using PyO3.
- 🛠 **Unified API**: Consistent interface for tracking tasks across both languages.

## Installation

### From Source
Requires Rust and Cargo to be installed.

```bash
# Install maturin
pip install maturin

# Build and install locally
maturin develop
```

## Usage

### Python
```python
import trackforge
```

### Rust
Add `trackforge` to your `Cargo.toml`.

## Development

This project uses `maturin` to manage the Rust/Python interop.

- **Build**: `maturin build`
- **Test**: `cargo test` and `pytest` (once added)
