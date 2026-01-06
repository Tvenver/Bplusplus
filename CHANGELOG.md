# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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