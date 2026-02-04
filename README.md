# B++ repository

[![DOI](https://zenodo.org/badge/765250194.svg)](https://zenodo.org/badge/latestdoi/765250194) 
[![PyPi version](https://img.shields.io/pypi/v/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![Python versions](https://img.shields.io/pypi/pyversions/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![License](https://img.shields.io/pypi/l/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![Downloads](https://static.pepy.tech/badge/bplusplus)](https://pepy.tech/project/bplusplus)
[![Downloads](https://static.pepy.tech/badge/bplusplus/month)](https://pepy.tech/project/bplusplus)
[![Downloads](https://static.pepy.tech/badge/bplusplus/week)](https://pepy.tech/project/bplusplus)

This project provides a complete, end-to-end pipeline for building a custom insect classification system. The framework is designed to be **domain-agnostic**, allowing you to train a powerful detection and classification model for **any insect species** by simply providing a list of names.

Using the `Bplusplus` library, this pipeline automates the entire machine learning workflow, from data collection to video inference.

## Key Features

- **Automated Data Collection**: Downloads hundreds of images for any species from the GBIF database.
- **Intelligent Data Preparation**: Uses a pre-trained model to automatically find, crop, and resize insects from raw images, ensuring high-quality training data.
- **Hierarchical Classification**: Trains a model to identify insects at three taxonomic levels: **family, genus, and species**.
- **Video Inference & Tracking**: Processes video files to detect, classify, and track individual insects over time, providing aggregated predictions.
## Pipeline Overview

The process is broken down into five main steps, all detailed in the `full_pipeline.ipynb` notebook:

1.  **Collect Data**: Select your target species and fetch raw insect images from the web.
2.  **Prepare Data**: Filter, clean, and prepare images for training.
3.  **Train Model**: Train the hierarchical classification model.
4.  **Validate Model**: Evaluate the performance of the trained model.
5.  **Run Inference**: Run the full pipeline on a video file for real-world application.

## How to Use

### Prerequisites

- Python 3.10+

### Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install bplusplus
    ```

### Running the Pipeline

The pipeline can be run step-by-step using the functions from the `bplusplus` library. While the `full_pipeline.ipynb` notebook provides a complete, executable workflow, the core functions are described below.

#### Step 1: Collect Data
Download images for your target species from the GBIF database. You'll need to provide a list of scientific names.

```python
import bplusplus
from pathlib import Path

# Define species and directories
names = ["Vespa crabro", "Vespula vulgaris", "Dolichovespula media"]
GBIF_DATA_DIR = Path("./GBIF_data")

# Define search parameters
search = {"scientificName": names}

# Run collection
bplusplus.collect(
    group_by_key=bplusplus.Group.scientificName,
    search_parameters=search,
    images_per_group=200,  # Recommended to download more than needed
    output_directory=GBIF_DATA_DIR,
    num_threads=5
)
```

#### Step 2: Prepare Data
Process the raw images to extract, crop, and resize insects. This step uses a pre-trained model to ensure only high-quality images are used for training.

```python
PREPARED_DATA_DIR = Path("./prepared_data")

bplusplus.prepare(
    input_directory=GBIF_DATA_DIR,
    output_directory=PREPARED_DATA_DIR,
    img_size=640,        # Target image size for training
    conf=0.6,            # Detection confidence threshold (0-1)
    valid=0.1,           # Validation split ratio (0-1), set to 0 for no validation
    blur=None,           # Gaussian blur as fraction of image size (0-1), None = disabled
)
```

**Note:** The `blur` parameter applies Gaussian blur before resizing, which can help reduce noise. Values are relative to image size (e.g., `blur=0.01` means 1% of the smallest dimension). Supported image formats: JPG, JPEG, and PNG.

#### Step 3: Train Model
Train the hierarchical classification model on your prepared data. The model learns to identify family, genus, and species.

```python
TRAINED_MODEL_DIR = Path("./trained_model")

bplusplus.train(
    batch_size=4,
    epochs=30,
    patience=3,
    img_size=640,
    data_dir=PREPARED_DATA_DIR,
    output_dir=TRAINED_MODEL_DIR,
    species_list=names,
    backbone="resnet50",  # Choose: "resnet18", "resnet50", or "resnet101"
    # num_workers=0,      # Optional: force single-process loading (most stable)
    # train_transforms=custom_transforms,  # Optional: custom torchvision transforms
)
```

**Note:** The `num_workers` parameter controls DataLoader multiprocessing (defaults to 0 for stability). The `backbone` parameter allows you to choose between different ResNet architectures—use `resnet18` for faster training or `resnet101` for potentially better accuracy.

#### Step 4: Validate Model
Evaluate the trained model on a held-out validation set. This calculates precision, recall, and F1-score at all taxonomic levels.

```python
HIERARCHICAL_MODEL_PATH = TRAINED_MODEL_DIR / "best_multitask.pt"

results = bplusplus.validate(
    species_list=names,
    validation_dir=PREPARED_DATA_DIR / "valid",
    hierarchical_weights=HIERARCHICAL_MODEL_PATH,
    img_size=640,           # Must match training
    batch_size=32,
    backbone="resnet50",    # Must match training
)
```

#### Step 5: Run Inference on Video
Process a video file to detect, classify, and track insects using motion-based detection. The pipeline uses background subtraction (GMM) to detect moving insects, tracks them across frames, and classifies confirmed tracks.

**Note:** The species list and taxonomy are automatically loaded from the model checkpoint, so you don't need to provide them again.

**Output files generated in `output_dir`:**
- `{video}_annotated.mp4` - Video showing confirmed tracks with classifications
- `{video}_debug.mp4` - Debug video with motion mask and all detections
- `{video}_results.csv` - Aggregated results per confirmed track
- `{video}_detections.csv` - Frame-by-frame detection data

```python
VIDEO_INPUT_PATH = Path("my_video.mp4")
OUTPUT_DIR = Path("./output")
HIERARCHICAL_MODEL_PATH = TRAINED_MODEL_DIR / "best_multitask.pt"

results = bplusplus.inference(
    hierarchical_model_path=HIERARCHICAL_MODEL_PATH,
    video_path=VIDEO_INPUT_PATH,
    output_dir=OUTPUT_DIR,
    # species_list=names,   # Optional: override species from checkpoint
    fps=None,               # None = process all frames
    backbone="resnet50",    # Must match training
    save_video=True,        # Set to False to skip video rendering (only CSV output)
    img_size=60,            # Must match training
)

print(f"Detected {results['tracks']} tracks ({results['confirmed_tracks']} confirmed)")
```

**Note:** Set `save_video=False` to skip generating the annotated and debug videos, which speeds up processing when you only need the CSV detection data.

**Custom Detection Configuration:**

For advanced control over detection parameters, provide a YAML config file:

```python
results = bplusplus.inference(
    ...,
    config="detection_config.yaml"
)
```

Download a template config from the [releases page](https://github.com/Tvenver/Bplusplus/releases).

<details>
<summary><b>Full Configuration Parameters</b> (click to expand)</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **GMM Background Subtractor** | | *Motion detection model* |
| `gmm_history` | 500 | Frames to build background model |
| `gmm_var_threshold` | 16 | Variance threshold for foreground detection |
| **Morphological Filtering** | | *Noise removal* |
| `morph_kernel_size` | 3 | Morphological kernel size (NxN) |
| **Cohesiveness** | | *Filters scattered motion (plants) vs compact motion (insects)* |
| `min_largest_blob_ratio` | 0.80 | Min ratio of largest blob to total motion |
| `max_num_blobs` | 5 | Max separate blobs allowed in detection |
| `min_motion_ratio` | 0.15 | Min ratio of motion pixels to bbox area |
| **Shape** | | *Filters by contour properties* |
| `min_area` | 200 | Min detection area (px²) |
| `max_area` | 40000 | Max detection area (px²) |
| `min_density` | 3.0 | Min area/perimeter ratio |
| `min_solidity` | 0.55 | Min convex hull fill ratio |
| **Tracking** | | *Controls track behavior* |
| `min_displacement` | 50 | Min net movement for confirmation (px) |
| `min_path_points` | 10 | Min points before path analysis |
| `max_frame_jump` | 100 | Max jump between frames (px) |
| `max_lost_frames` | 45 | Frames before lost track deleted (e.g., 45 @ 30fps = 1.5s) |
| `max_area_change_ratio` | 3.0 | Max area change ratio between frames |
| **Tracker Matching** | | *Hungarian algorithm cost function* |
| `tracker_w_dist` | 0.6 | Weight for distance cost (0-1) |
| `tracker_w_area` | 0.4 | Weight for area cost (0-1) |
| `tracker_cost_threshold` | 0.3 | Max cost for valid match (0-1) |
| **Path Topology** | | *Confirms insect-like movement patterns* |
| `max_revisit_ratio` | 0.30 | Max ratio of revisited positions |
| `min_progression_ratio` | 0.70 | Min forward progression |
| `max_directional_variance` | 0.90 | Max heading variance |
| `revisit_radius` | 50 | Radius (px) for revisit detection |

</details>

### Customization

To train the model on your own set of insect species, you only need to change the `names` list in **Step 1**. The pipeline will automatically handle the rest.

```python
# To use your own species, change the names in this list
names = [
    "Vespa crabro",
    "Vespula vulgaris",
    "Dolichovespula media",
    # Add your species here
]
```

#### Handling an "Unknown" Class
To train a model that can recognize an "unknown" class for insects that don't belong to your target species, add `"unknown"` to your `species_list`. You must also provide a corresponding `unknown` folder containing images of various other insects in your data directories (e.g., `prepared_data/train/unknown`).

```python
# Example with an unknown class
names_with_unknown = [
    "Vespa crabro",
    "Vespula vulgaris",
    "unknown"
]
```

## Directory Structure

The pipeline will create the following directories to store artifacts:

- `GBIF_data/`: Stores the raw images downloaded from GBIF.
- `prepared_data/`: Contains the cleaned, cropped, and resized images ready for training (`train/` and optionally `valid/` subdirectories).
- `trained_model/`: Saves the trained model weights (`best_multitask.pt`).
- `output/`: Inference results including annotated videos and CSV files.

# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
