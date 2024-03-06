**B++ repository**

This repo can be used to quickly generate YOLOv8 models for biodiversity monitoring, relying on Ultralytics and a GBIF dataset.

**How does it work?**
1. Input names (scientific name) in the names.csv
2. Download the GBIF repository of your choosing, or download a prepared dataset linking to 16M images of many insect species: https://doi.org/10.15468/dl.dk9czq
3. Update the path in collect_images.py on line 35 and line 53, to route to the unzipped GBIF downloaded files.
4. run collect_images.py, this fetches the names, iterates through them, and attempt to download images from a GBIF data repository.
5. This might take +-20 minutes, depending on your internet speed and hardware.
6. run train_validate.py, this shuffels the images into a train and validation set, and Ultralytics takes care of the training.
7. You can tweek various parameters for the training, if you want to, please visit the Ultralytics YOLOv8 documentation for more information.

You have created a YOLOv8 model for image classification.

