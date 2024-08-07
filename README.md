# B++ repository

This repo can be used to quickly generate YOLOv8 models for biodiversity monitoring, relying on Ultralytics and a GBIF dataset.

All code is tested on Windows 10 and Python 3.11, without GPU. GPU would obviously accelerate the below steps, Ultralytics should automatically select the available GPU if there is any.

# How does it work?
1. Input names (scientific names) in the names.csv file, in the data folder
2. Download the GBIF repository of your choosing, or download a prepared dataset linking to 16M images of many insect species: https://doi.org/10.15468/dl.dk9czq
3. Update the path in collect_images.py on line 35 and line 53, to route to the unzipped GBIF downloaded files.
4. In collect_images.py, consider activating the sampling function, to reduce the number of images to download per species - in the case of many insect species, the download will take longer.
5. run collect_images.py, this fetches the names, iterates through them, and attempts to download images from a GBIF data repository.
6. As an example, for about 8 insect species, ending up with 4000 images, the entire operation might take +-20 minutes, depending on your internet speed and hardware.
7. run train_validate.py, this shuffles the images into a train and validation set, and Ultralytics takes care of the training.
8. You can tweak various parameters for the training, if you want to, please visit the Ultralytics YOLOv8 documentation for more information.

You have created a YOLOv8 model for image classification.

![Figure 9](https://github.com/user-attachments/assets/a01f513b-0609-412d-a633-3aee1e5dded6)

There is also a pretrained YOLOv8 classification model, containing 2584 species, included in this repo under B++ CV Model.
![Fig2](https://github.com/user-attachments/assets/8b0e73db-c942-41fd-88bc-a1c370f5bd18)

# Citation

All information in this GitHub is available under MIT license, as long as credit is given to the authors.

**Venverloo, T., Duarte, F., B++: Towards Real-Time Monitoring of Insect Species. MIT Senseable City Laboratory, AMS Institute.**
