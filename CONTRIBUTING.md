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

### Commit messages

We use Commitizen-style Conventional Commits and require a `Signed-off-by` trailer on
every commit (`git commit -s`). The full rules, including what must never appear in commit
messages or PR descriptions, are in [docs/COMMIT_GUIDELINES.md](docs/COMMIT_GUIDELINES.md).

In short: commit with `git commit -sm "type(scope): subject"`, use a tracker or area as the
scope, and keep commit and PR text free of co-author trailers, AI-attribution lines, em
dashes, and ASCII art.

### Code formatting

Run `cargo fmt` before committing to ensure that code is consistently formatted.

### Git hooks with prek

This repo ships a [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and runs the
hooks in CI with [`prek`](https://github.com/j178/prek), a fast, dependency-free Rust
reimplementation of `pre-commit`. It reads the same config, so no extra setup file is
needed. We recommend the Rust install:

```shell
# Install prek (pick one)
cargo install --locked prek
# or, prebuilt binaries:
cargo binstall prek

# Set up the git hooks (pre-commit and commit-msg)
prek install

# Run every hook against the whole tree
prek run --all-files
```

The hooks cover formatting (`cargo fmt`, `taplo`, `prettier`), linting (`cargo clippy`,
`markdownlint`), and hygiene (`typos`, trailing whitespace, merge-conflict markers). The
classic `pre-commit run --all-files` still works if you prefer it.

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

## Adding a tracker

Every tracker ships for both Rust and Python, is tested in both, is documented, and
credits the paper and original authors it is based on. The full step-by-step process,
module layout, API conventions, and credit requirements are in
[docs/adding-a-tracker.md](docs/adding-a-tracker.md), with a fill-in-the-blanks spec in
[.github/TRACKER_TEMPLATE.md](.github/TRACKER_TEMPLATE.md).

Open a "New tracker" issue first so the algorithm, scope, and parameters can be agreed
before you write code. A complete tracker contribution includes:

- The Rust implementation under `src/trackers/<name>/`, reusing the shared
  `utils::{kalman,geometry,assignment}` and `trackers::common` building blocks.
- The PyO3 binding and its registration in `src/lib.rs`.
- Rust unit tests and a block in `tests/test_python_bindings.py`.
- A per-tracker `README.md`, a `book/src/trackers/<name>.md` guide, a Python example on
  `examples/python/common.py`, and updates to the README tables and `docs/roadmap.md`.
- Paper credit: arXiv link, original-implementation reference, and a BibTeX citation.

## Documentation

We use a hybrid documentation strategy to serve both Python and Rust API docs in a single zensical site.

### Dependencies

- **Python**: `zensical`, `mkdocstrings[python]`, `maturin`
- **Rust**: `cargo-docs-md`
- **Rust Toolchain**: `nightly` (required for JSON output)

```bash
# Install Python deps
pip install zensical "mkdocstrings[python]" maturin

# Install Rust tool
cargo install cargo-docs-md
```

### Generating the Site

1. **Build Python Package**:

   ```bash
   maturin develop
   ```

2. **Generate Rustdoc JSON**:

   ```bash
   RUSTDOCFLAGS="-Z unstable-options --output-format json" cargo +nightly doc --no-deps
   ```

3. **Convert to Markdown**:

   ```bash
   mkdir -p docs/api
   cargo docs-md -p target/doc/trackforge.json -o docs/api --full-method-docs
   ```

4. **Fix Formatting**:
   Run the cleaning script to format tables and code blocks:

   ```bash
   python3 scripts/fix_docs.py
   ```

5. **Serve**:

   ```bash
   zensical serve
   ```

   Open http://127.0.0.1:8000.
