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

The process is broken down into six main steps, all detailed in the `full_pipeline.ipynb` notebook:

1.  **Collect Data**: Select your target species and fetch raw insect images from the web.
2.  **Prepare Data**: Filter, clean, and prepare images for training.
3.  **Train Model**: Train the hierarchical classification model.
4.  **Download Weights**: Fetch pre-trained weights for the detection model.
5.  **Test Model**: Evaluate the performance of the trained model.
6.  **Run Inference**: Run the full pipeline on a video file for real-world application.

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
    img_size=640  # Target image size for training
)
```

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
    species_list=names
    # num_workers=0  # Optional: force single-process loading (most stable)
)
```

**Note:** The `num_workers` parameter controls DataLoader multiprocessing (defaults to 0 for stability). You can increase it for potentially faster data loading.

#### Step 4: Download Detection Weights
The inference pipeline uses a separate, pre-trained YOLO model for initial insect detection. You need to download its weights manually.

You can download the weights file from [this link](https://github.com/Tvenver/Bplusplus/releases/download/v1.2.3/v11small-generic.pt).

Place it in the `trained_model` directory and ensure it is named `yolo_weights.pt`.

#### Step 5: Run Inference on Video
Process a video file to detect, classify, and track insects. The final output is an annotated video and a CSV file with aggregated results for each tracked insect.

```python
VIDEO_INPUT_PATH = Path("my_video.mp4")
VIDEO_OUTPUT_PATH = Path("my_video_annotated.mp4")
HIERARCHICAL_MODEL_PATH = TRAINED_MODEL_DIR / "best_multitask.pt"
YOLO_WEIGHTS_PATH = TRAINED_MODEL_DIR / "yolo_weights.pt"

bplusplus.inference(
    species_list=names,
    yolo_model_path=YOLO_WEIGHTS_PATH,
    hierarchical_model_path=HIERARCHICAL_MODEL_PATH,
    confidence_threshold=0.35,
    video_path=VIDEO_INPUT_PATH,
    output_path=VIDEO_OUTPUT_PATH,
    tracker_max_frames=60,
    fps=15  # Optional: set processing FPS
)
```

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
- `prepared_data/`: Contains the cleaned, cropped, and resized images ready for training.
- `trained_model/`: Saves the trained model weights (`best_multitask.pt`) and pre-trained detection weights.

# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
