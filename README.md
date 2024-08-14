# B++ repository

[![DOI](https://zenodo.org/badge/765250194.svg)](https://zenodo.org/badge/latestdoi/765250194) 
[![PyPi version](https://img.shields.io/pypi/v/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![Python versions](https://img.shields.io/pypi/pyversions/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![License](https://img.shields.io/pypi/l/bplusplus.svg)](https://pypi.org/project/bplusplus/)
[![Downloads](https://static.pepy.tech/badge/bplusplus)](https://pepy.tech/project/bplusplus)
[![Downloads](https://static.pepy.tech/badge/bplusplus/month)](https://pepy.tech/project/bplusplus)
[![Downloads](https://static.pepy.tech/badge/bplusplus/week)](https://pepy.tech/project/bplusplus)

This repo can be used to quickly generate YOLOv8 models for biodiversity monitoring, relying on Ultralytics and a GBIF dataset.
All code is tested on Windows 10 and Python 3.11, without GPU. GPU would obviously accelerate the below steps, Ultralytics should automatically select the available GPU if there is any.

# New release
We have released a new version here: [github.com/Tvenver/Bplusplus/tree/package](https://github.com/Tvenver/Bplusplus/tree/package)
We also launched a package, which can be installed directly: [https://github.com/Tvenver/Bplusplus/tree/package](https://pypi.org/project/bplusplus/)

# How does it work?

To create your own custom CV model:
1. Input names (scientific names) in the names.csv file, in the data folder
2. Download the GBIF repository of your choosing, or download a prepared dataset linking to 16M images of many insect species: https://doi.org/10.15468/dl.dk9czq
3. Update the path in collect_images.py on line 36 and line 54, to route to the unzipped GBIF downloaded files.
4. In collect_images.py, consider activating the sampling function, to reduce the number of images to download per species - in the case of many insect species, the download will take longer.
5. run collect_images.py, this fetches the names, iterates through them, and attempts to download images from a GBIF data repository.
6. As an example, for about 8 insect species, ending up with 4000 images, the entire operation might take +-20 minutes, depending on your internet speed and hardware.
7. run train_validate.py, this shuffles the images into a train and validation set, and Ultralytics takes care of the training.
8. You can tweak various parameters for the training, if you want to, please visit the Ultralytics YOLOv8 documentation for more information.

You have created a YOLOv8 model for image classification.

![Figure 9](https://github.com/user-attachments/assets/a01f513b-0609-412d-a633-3aee1e5dded6)

To use the pretrained model:
There is also a pretrained YOLOv8 classification model, containing 2584 species, included in this repo under B++ CV Model. The included species are listed in a separate file.
1. Download the pretrained model from the Google Drive link listed in the folder B++ CV Model
2. Take the run_model.py script, specify the path to the downloaded .pt file, and run the model.

# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
