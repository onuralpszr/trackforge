<p align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-dark.png" media="(prefers-color-scheme: dark)" />
        <source srcset="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-transparent.png" media="(prefers-color-scheme: light)" />
        <img src="https://raw.githubusercontent.com/onuralpszr/trackforge/main/assets/track-forge-transparent.png" alt="Trackforge logo" width="auto" />
    </picture>
</p>

**Trackforge** is a unified, high-performance computer vision tracking library built with Rust and exposed to Python. It implements state-of-the-art algorithms like **ByteTrack** with generic Kalman Filters.

## Features

- 🚀 **High Performance**: Written in Rust for maximum speed and memory safety.
- 🐍 **Python Bindings**: Seamless integration with the Python ecosystem via PyO3.
- 👁️ **Computer Vision Ready**: Designed for real-time tracking tasks.
- 🛠️ **Unified API**: Consistent interface for various tracking algorithms.

## Quick Start

### Installation

```bash
pip install trackforge
```

### Usage

```python
import trackforge

# Example usage (update with actual API)
tracker = trackforge.ByteTrack()
```

## Documentation

- [Python API Reference](reference/python.md)
- [Rust API Reference](/api/trackforge/index.html)
