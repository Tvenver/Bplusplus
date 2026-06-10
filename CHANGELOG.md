# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2026-06-10

### Added
- **`detection_resolution` config option**: Set an explicit `[width, height]`
  to run motion detection on downscaled frames for speed. Bounding boxes are
  scaled back to native resolution before tracking, so crops, composites, and
  classification still operate at full resolution — only the GMM/morphology/
  contour stage gets cheaper. Available via the config file/dict and the new
  `--detection-resolution W H` CLI flag. `null`/omitted detects at native
  resolution (unchanged behaviour).

### Changed
- `VideoInferenceProcessor.setup()` builds the motion detector with parameters
  resolved at the detection resolution and runs detection on a resized frame
  when `detection_resolution` is set. The native-resolution config still drives
  tracking, topology, and consistency checks. `min_density` (a length-
  dimensioned ratio that `resolve_detection_params` does not scale) is scaled
  internally by the linear downscale factor so the same objects keep passing.
- Requires `bugspot >=0.4.0` (which adds `detection_resolution` to the default
  detection config).

## [2.2.0] - 2026-03-26

### Fixed
- **Python 3.13 compatibility claim corrected.** `pyproject.toml` previously
  declared `python = "^3.10"`, which implied Python 3.13 was supported.
  In practice the pinned dependency stack cannot install on CPython 3.13.
  The Python constraint is now `>=3.10,<3.12` inclusive-style (i.e.
  `>=3.10,<3.13`) and the README prerequisites state Python 3.10–3.12.
  Full 3.13 support is planned for a future release.

### Added
- **`{video_name}_tracks.csv` output**: New per-run CSV listing every track
  (confirmed *and* unconfirmed) with detection stats, raw topology
  calculations (`net_displacement`, `revisit_ratio`, `progression_ratio`,
  `directional_variance`, `total_path_length`), per-criterion pass flags
  (`path_points_pass`, `displacement_pass`, `revisit_pass`,
  `progression_pass`, `directional_variance_pass`), and the config
  thresholds used for each check. The CSV records both the originating
  fraction config value (e.g. `min_displacement_frac`,
  `revisit_radius_frac`) and the resolved pixel threshold actually used,
  for full traceability. Enables fast diagnosis of why tracks were
  rejected by the topology filter.
- `compute_full_track_metrics()` helper that always returns raw metric
  values (NaN only when genuinely undefined), regardless of path length.

### Changed
- **Pixel-scale config values are now expressed as FRACTIONS of image
  dimensions, not absolute pixels.** Affected keys: `morph_kernel_size`,
  `min_area`, `max_area`, `min_displacement`, `max_frame_jump`,
  `revisit_radius`. Lengths are fractions of the image width `W`; areas
  are fractions of `W * H`. One config now works across resolutions.
  See `detection_config.yaml` and the README for 1080 px wide reference
  values. Pre-existing configs with absolute pixel values should be
  updated — a runtime warning is emitted if a fraction value exceeds 1.0.
- `VideoInferenceProcessor` now defers motion detector creation to a new
  `setup(image_width, image_height)` call, invoked automatically by
  `process_video` after the input video is opened. Direct callers of
  `process_frame` still work — `setup` auto-runs from the first frame's
  shape if not called explicitly.
- Requires `bugspot >=0.3.2` (which exports the underlying metric helpers
  `calculate_revisit_ratio`, `calculate_progression_ratio`,
  `calculate_directional_variance`, and the new
  `resolve_detection_params`). `morph_kernel_size` stays in absolute
  NxN pixels as before; only scene-scale lengths/areas became fractions.
- `analyze_tracks`, `detection_only_results`, and `hierarchical_aggregation`
  now share a single metric-computation path for consistency.

## [2.1.0] - 2025-02-18

### Added
- **[BugSpot](https://github.com/orlandocloss/bugspot) core library**: Extracted motion detection, tracking, and path topology into standalone package (opencv + numpy + scipy only, no ML frameworks)
- **Detection-only mode**: `classify=False` skips model loading — outputs NaN for classification fields
- **Track composite images**: `track_composites=True` generates per-track temporal trail images (lighten blend on darkened background)

### Changed
- **Inference now depends on bugspot** for detection, tracking, topology analysis, crop extraction, and composite rendering
- Removed `detector.py` and `tracker.py` from bplusplus — single source of truth in bugspot
- Consolidated inference documentation in README and notebook into one clean section
- `video_path` and `output_dir` are now the first two parameters in `inference()`

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