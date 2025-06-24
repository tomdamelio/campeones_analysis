# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Initial project scaffold and reproducible environment setup.
- DVC integration with Google Drive using Service Account:
  - Documented recommended workflow: manual backup to Drive, local download, versioning with DVC, and push/pull to a separate DVC remote.
  - Added best practices: never manually upload data to DVC remote, always version and sync with DVC for reproducibility.
  - Step-by-step instructions for configuring Service Account and Drive permissions.

### Added
- Added `mnelab` and `pywavelets` dependencies for XDF file reading and signal processing
- Added `pytest-cov` for test coverage reporting
- Added `openpyxl` dependency for reading Excel files with order matrices
- Added `glhmm` dependency via pip for Gaussian-Linear Hidden Markov Models analysis
