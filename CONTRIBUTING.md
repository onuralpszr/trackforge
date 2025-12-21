# Contribution guidelines

First off, thank you for considering contributing to Trackforge.

If your contribution is not straightforward, please first discuss the change you wish to make by creating a new issue before making the change, or starting a discussion on GitHub.

## AI Generated Content

We welcome high quality PRs, whether they are human generated or made with the assistance of AI tools, but we ask that you follow these guidelines:

- **Attribution**: Tell us about your use of AI tools.
- **Review**: Make sure you review every line of AI generated content for correctness and relevance.
- **Quality**: AI-generated content should meet the same quality standards as human-written content.

## Pull requests

All contributions are welcome. Please include as many details as possible in your PR description.

### Keep PRs small, intentional, and focused

- Aim for PRs under 500 lines of changes when possible.
- Separate refactoring, formatting, and functional changes into different PRs.

### Code formatting

Run `cargo fmt` before committing to ensure that code is consistently formatted.

## Implementation Guidelines

### Setup

TL;DR: Clone the repo and build it using `cargo` (for Rust) or `maturin` (for Python).

```shell
git clone https://github.com/onuralpszr/trackforge.git
cd trackforge

# Pure Rust Development
cargo build
cargo test

# Python Development
# Ensure you are in a virtual environment
maturin develop
```

### Tests

- **Rust**: Run `cargo test` to execute unit and integration tests.
- **Python**: Run `pytest` (when available) to check Python bindings.

### Continuous Integration

We use GitHub Actions for CI where we perform the following checks:

- The code should compile on stable Rust.
- The tests should pass (`cargo test`).
- The code should be formatted (`cargo fmt`).
- The code should pass hygiene checks (`cargo clippy`).

You can check these locally:

```shell
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test
```
