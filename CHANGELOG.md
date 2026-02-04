# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.5] - 2025-02-04

### Added
- **JPEG support**: `prepare()` now fully supports `.jpeg` files in addition to `.jpg` and `.png`
- **Full detection configuration**: All 24 detection parameters now exposed in `detection_config.yaml` with comprehensive documentation

### Changed
- **Streamlined inference API**: `species_list` is now optional and automatically loaded from model checkpoint (still can be overridden if needed)
- **Frame-based tracking**: Standardized on `max_lost_frames` (frame-based) instead of `lost_track_seconds` for consistent behavior across different FPS
- **Refactored detection modules**: Moved all hardcoded values to `detection_config.yaml` for better configurability
  - GMM parameters (`gmm_history`, `gmm_var_threshold`)
  - Morphological filtering (`morph_kernel_size`)
  - Cohesiveness filters (`min_motion_ratio`)
  - Track consistency (`max_area_change_ratio`)
  - Path topology (`revisit_radius`)

### Fixed
- **Indentation error** in `prepare.py` file corruption detection loop

## [2.0.4] - 2025-02-02

### Added
- **Configurable inference image size**: New `img_size` parameter in `inference()` to match training size (default: 60)

### Changed
- Renamed `insect_detector.py` to `detector.py` for cleaner module naming

### Fixed
- **Critical**: Inference now uses correct image size for classification. Previously hardcoded to 768x768 → 640, which caused poor accuracy when training with smaller sizes (e.g., 60px)

## [2.0.3] - 2025-01-28

### Added
- **Gaussian blur option**: New `blur` parameter in `prepare()` to apply Gaussian blur before resizing (as fraction of image size, 0-1)
- **Skip video rendering**: New `save_video` parameter in `inference()` to skip video output and only generate CSVs (faster processing)
- **PNG support**: `prepare()` now accepts PNG images in addition to JPG/JPEG

### Changed
- Updated documentation in README and notebook with new parameters

## [2.0.2] - 2025-01-20

### Added
- **Crop export**: New `--crops` flag in inference to save cropped frames for each classified track, organized by track ID

## [2.0.1] - 2025-01-20

### Fixed
- Minor bug fixes and code cleanup

## [2.0.0] - 2025-01-06

### Added
- **Motion-informed inference**: New detection pipeline using Gaussian Mixture Model (GMM) background subtraction instead of YOLO, with path topology analysis for confirming insect-like movement
- **Validation module** (`validation.py`): New `bplusplus.validate()` function to evaluate model performance with precision, recall, and F1-score at all taxonomic levels
- **Configurable ResNet backbone**: Choose between `resnet18`, `resnet50`, or `resnet101` for training and inference
- **Custom training transforms**: New `train_transforms` parameter for custom data augmentation in `train()`
- **Validation split control**: New `valid_fraction` parameter (0-1) in `prepare()` to control train/validation split ratio
- **Detection confidence parameter**: New `conf` parameter in `prepare()` for YOLO confidence threshold
- **Class imbalance warnings**: Training now analyzes and warns about class imbalance across taxonomic levels
- **Detection configuration**: Support for YAML/JSON config files to customize motion detection parameters
- **New prepare weights**: Higher accuracy `gbif-generic` weights for data preparation

### Improved
- **Collect robustness**: Added retry logic with exponential backoff for GBIF API calls, progress tracking, and graceful handling of interruptions
- **GBIF quality filters**: Enhanced filtering options including `occurrenceStatus`, `year` range, and more

### Changed
- **Inference pipeline**: Replaced YOLO-based detection with motion-based detection using GMM
- **Output structure**: Inference now outputs to a single directory with multiple files (`_annotated.mp4`, `_debug.mp4`, `_results.csv`, `_detections.csv`)
- **Results CSV**: Now contains only aggregated results for confirmed tracks

### Removed
- **YOLO dependency for inference**: No longer requires YOLO weights for video inference
- **test.py**: Removed outdated YOLO test module

## [1.2.2] - 2024-12-19

### Added
- Enhanced GBIF occurrence filtering in `collect.py`:
  - Added `basisOfRecord` filter to include only HUMAN_OBSERVATION, LIVING_SPECIMEN, MACHINE_OBSERVATION, OBSERVATION, and OCCURRENCE records
  - Added `lifeStage` filter to include only Adult specimens
  - These changes improve the quality and relevance of collected images by filtering out juvenile specimens and unsuitable record types