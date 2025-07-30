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

- Python 3.8+
- `venv` for creating a virtual environment (recommended)

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

The entire workflow is contained within **`full_pipeline.ipynb`**. Open it with a Jupyter Notebook or JupyterLab environment and run the cells sequentially to execute the full pipeline.

### Customization

To train the model on different insect species, simply modify the `names` list in **Step 1** of the notebook:

```python
# a/full_pipeline.ipynb

# To use your own species, change the names in this list
names = [
    "Vespa crabro", "Vespula vulgaris", "Dolichovespula media"
]
```

The pipeline will automatically handle the rest, from data collection to training, for your new set of species.

# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
