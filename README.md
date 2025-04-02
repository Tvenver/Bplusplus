# B++ repository

[![DOI](https://zenodo.org/badge/765250194.svg)](https://zenodo.org/badge/latestdoi/765250194) 
[![PyPi version](https://img.shields.io/pypi/v/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![Python versions](https://img.shields.io/pypi/pyversions/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![License](https://img.shields.io/pypi/l/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![Downloads](https://static.pepy.tech/badge/bplusplus)](https://pepy.tech/project/bplusplus)
[![Downloads](https://static.pepy.tech/badge/bplusplus/month)](https://pepy.tech/project/bplusplus)
[![Downloads](https://static.pepy.tech/badge/bplusplus/week)](https://pepy.tech/project/bplusplus)

This repo can be used to quickly generate models for biodiversity monitoring, relying on the GBIF dataset.

# Three pipeline options

## One stage YOLO

For the one stage pipeline, we first collect `collect()` the data from GBIF, then prepare the data for training by running the `prepare()` function, which adds bounding boxes to the images using a pretrained YOLO model.  We then train the model with YOLOv8 using the `train()` function. 

## Two stage YOLO/Resnet

For the two stage pipeline, we first collect `collect()` the data from GBIF, then prepare `prepare()` this (classification) data for training by either size filtering (recommended "large") which also splits the data into train and valid. We then train the model with resnet using the `train_resnet()` function. The trained model is a resnet classification model which will then be paired with a pretrained YOLOv8 insect detection model (hence two stage). 

## Two stage YOLO/Multitask-Resnet

For the two stage pipeline, we first collect `collect()` the data from GBIF, then prepare `prepare()` this (classification) data for training by either size filtering (recommended "large") which also splits the data into train and valid. We then train the model with resnet using the `train_multitask()` function. The difference here is that it is training for species, order and family simultaneously. The trained model is a resnet classification model which will then be paired with a pretrained YOLOv8 insect detection model (hence two stage). 

### Install package

```python
pip install bplusplus
```

### bplusplus.collect() (All pipelines)

This function takes three arguments: 
- **search_parameters: dict[str, Any]** - List of scientific names of the species you want to collect from the GBIF database 
- **images_per_group: int** - Number of images per species collected for training. Max 9000. 
- **output_directory: str** - Directory to store collected images
- **num_threads: int** - Number of threads you want to run for collecting images. We recommend using a moderate number (3-5) to avoid overwhelming tha API server.  

Example run: 
```python
import bplusplus

species_list=[ "Vanessa atalanta", "Gonepteryx rhamni", "Bombus hortorum"] 
# convert to dict
search: dict[str, any] = {
    "scientificName": species_list
}

images_per_group=20 
output_directory="/dataset/selected-species"
num_threads=2


# Collect data from GBIF
bplusplus.collect(
  search_parameters=search,
  images_per_group=images_per_group,
  output_directory=output_directory,
  group_by_key=bplusplus.Group.scientificName,
  num_threads=num_threads
)
```

### bplusplus.prepare() (All pipelines)

Prepares the dataset for training by performing the following steps:
  1. Copies images from the input directory to a temporary directory.
  2. Deletes corrupted images.
  3. Downloads YOLOv5 weights for *insect detection* if not already present.
  4. Runs YOLOv5 inference to generate labels for the images.
  5. Deletes orphaned images and inferences.
  6. Updates labels based on class mapping.
  7. Splits the data into train, test, and validation sets.
  8. Counts the total number of images across all splits.
  9. Makes a YAML configuration file for YOLOv8.

This function takes three arguments: 
- **input_directory: str** - The path to the input directory containing the images.
- **output_directory: str** - The path to the output directory where the prepared dataset will be saved.
- **with_background: bool = False** - Set to False if you don't want to include/download background images
- **one_stage: bool = False** - Set to True if you want to train a one stage model
- **size_filter: bool = False** - Set to True if you want to filter by size of insect 
- **sizes: list = None** - List of sizes to filter by. If None, all sizes will be used, ["large", "medium", "small"].

```python
# Prepare data 
bplusplus.prepare(
    input_directory='/dataset/selected-species',
    output_directory='/dataset/prepared-data',
    with_background=False,
    one_stage=False,
    size_filter=True,
    sizes=["large"]
)
```

### bplusplus.train() (One stage pipeline)

This function takes five arguments: 
- **input_yaml: str** - yaml file created to train the model
- **output_directory: str**
- **epochs: int = 30** - Number of epochs to train the model
- **imgsz: int = 640** - Image size 
- **batch: int = 16** - Batch size for training

```python
# Train model
model = bplusplus.train(
  input_yaml="/dataset/prepared-data/dataset.yaml", # Make sure to add the correct path
  output_directory="trained-model",
  epochs=30, 
  batch=16 
)
```

### bplusplus.train_resnet() (Two stage (standard resnet) pipeline)

This function takes eight arguments: 
- **species_list: list** - List of species to train the model on
- **model_type: str** - The type of resnet model to train. Options are "resnet50", "resnet152"
- **batch_size: int** - The batch size for training
- **num_epochs: int** - The number of epochs to train the model
- **patience: int** - The number of epochs to wait before early stopping
- **output_dir: str** - The path to the output directory where the trained model will be saved
- **data_dir: str** - The path to the directory containing the prepared data
- **img_size: int** - The size of the images to train the model on

```python
# Train resnet model
bplusplus.train_resnet(
  species_list=["Vanessa atalanta", "Gonepteryx rhamni", "Bombus hortorum"],
  model_type="resnet50",
  batch_size=16,
  num_epochs=30,
  patience=5,
  output_dir="trained-model",
  data_dir="prepared-data",
  img_size=256
)
```

### bplusplus.train_multitask() (Two stage (multitask resnet) pipeline)

This function takes seven arguments: 
- **batch_size: int** - The batch size for training
- **epochs: int** - The number of epochs to train the model
- **patience: int** - The number of epochs to wait before early stopping
- **img_size: int** - The size of the images to train the model on
- **data_dir: str** - The path to the directory containing the prepared data
- **output_dir: str** - The path to the output directory where the trained model will be saved
- **species_list: list** - List of species to train the model on

```python
# Train multitask model
bplusplus.train_multitask(
  batch_size=16,
  epochs=30,
  patience=5,
  img_size=256,
  data_dir="prepared-data",
  output_dir="trained-model",
  species_list=["Vanessa atalanta", "Gonepteryx rhamni", "Bombus hortorum"]
)
```


### bplusplus.validate() (One stage pipeline)

This function takes two arguments: 
- **model** - The trained YOLO model
- **Path to yaml file** 

```python
metrics = bplusplus.validate(model, '/dataset/prepared-data/dataset.yaml')
print(metrics)
```

### bplusplus.test_resnet() (Two stage (standard resnet) pipeline)

This function takes six arguments: 
- **data_path: str** - The path to the directory containing the test data
- **yolo_weights: str** - The path to the YOLO weights
- **resnet_weights: str** - The path to the resnet weights
- **model: str** - The type of resnet model to use
- **species_names: list** - The list of species names
- **output_dir: str** - The path to the output directory where the test results will be saved

```python

bplusplus.test_resnet(
    data_path=TEST_DATA_DIR,
    yolo_weights=YOLO_WEIGHTS,
    resnet_weights=RESNET_WEIGHTS,
    model="resnet50",
    species_names=species_list,
    output_dir=TRAINED_MODEL_DIR
)
```

### bplusplus.test_multitask() (Two stage (multitask resnet) pipeline)

This function takes five arguments: 
- **species_list: list** - List of species to test the model on
- **test_set: str** - The path to the directory containing the test data
- **yolo_weights: str** - The path to the YOLO weights
- **hierarchical_weights: str** - The path to the hierarchical weights
- **output_dir: str** - The path to the output directory where the test results will be saved


```python
bplusplus.test_multitask(
    species_list,
    test_set=TEST_DATA_DIR,
    yolo_weights=YOLO_WEIGHTS,
    hierarchical_weights=RESNET_MULTITASK_WEIGHTS,
    output_dir=TRAINED_MODEL_DIR
)
```
# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
