# B++ repository

[![DOI](https://zenodo.org/badge/765250194.svg)](https://zenodo.org/badge/latestdoi/) 
[![PyPI version](https://badge.fury.io/py/bplusplus.svg)](https://badge.fury.io/py/bplusplus)

This repo can be used to quickly generate YOLOv8 models for biodiversity monitoring, relying on Ultralytics and a GBIF dataset.

All code is tested on macOS and Python 3.12, without GPU. GPU would obviously accelerate the below steps, Ultralytics should automatically select the available GPU if there is any.

## GitHub
https://github.com/Tvenver/Bplusplus

## PyPI
https://pypi.org/project/bplusplus/

# How does it work?

![Figure 9](https://github.com/user-attachments/assets/a01f513b-0609-412d-a633-3aee1e5dded6)

1. Select scientific names you want to train your model on. For now, only scientific names are supported as training categories.
2. Select the parameters you want to use to filter your dataset (using the [parameters available in the GBIF Occurrence Search API](https://techdocs.gbif.org/en/openapi/v1/occurrence)).
3. Decide how many images you want to use for training and validation per category.
4. Select a directory to output the model information.
5. Pass the above information to the `build_model` function.

You have created a YOLOv8 model for bug classification.

The training and validation is done using Ultralytics. Please visit the Ultralytics YOLOv8 documentation for more information.

# Pretrained Model

There is also a pretrained YOLOv8 classification model, containing 2584 species, included in this repo under B++ CV Model. The included species are listed in a separate file.
1. Download the pretrained model from the Google Drive link listed in the folder B++ CV Model
2. Take the notebooks/run_model.py script, specify the path to the downloaded .pt file, and run the model.

# Example Usage
## Using search options
```python
import os
import bplusplus
from typing import Any

names = [
    "Nabis rugosus", 
    "Forficula auricularia",
    "Calosoma inquisitor",
    "Bombus veteranus",
    "Glyphotaelius pellucidus",
    "Notoxus monoceros",
    "Cacoxenus indagator",
    "Chorthippus mollis",
    "Trioza remota"
]

search: dict[str, Any] = {
    "scientificName": names,
    "country": ["US", "NL"]
}

bplusplus.build_model(
    group_by_key=bplusplus.Group.scientificName,
    search_parameters=search, 
    images_per_group=150,
    model_output_folder=os.path.join('model')
)
```

# Pending Improvements

* The Ultralytics parameters should be surfaced to the user of the package so they have more control over the training process.
* The GBIF API documentation claims that you can filter on a dataset in your search, however it does not work in my current testing. This would be nice to allow users to create datasets on the GBIF website then pass that DOI directly here, so may warrant a closer look.


# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
