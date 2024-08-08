import os
import random
import shutil

from ultralytics import YOLO


def train_validate(scientificNames: list[str], dataset_path: str, output_directory: str):
# def train_validate(data_directory: str, dataset_path: str, train_path: str, val_path: str):
    # Define the ratio for splitting the dataset
    split_ratio = 0.8  # 80% for training, 20% for validation

    train_path = os.path.join(output_directory, 'train')  # Path to the training folder
    val_path = os.path.join(output_directory, 'val')   # Path to the validation folder


    # Create training and validation directories if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Walk through the dataset directory
    # for root, dirs, files in os.walk(dataset_path):
    for species in scientificNames:
        dataset_folder = os.path.join(dataset_path, species)
        images = __files_in_folder(folder=dataset_folder)

        # Shuffle the images
        random.shuffle(images)

        # Calculate the split index
        split_index = int(len(images) * split_ratio)

        # Split the images into training and validation sets
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create destination folders if they don't exist
        train_label_path = os.path.join(train_path, species)
        val_label_path = os.path.join(val_path, species)
        os.makedirs(train_label_path, exist_ok=True)
        os.makedirs(val_label_path, exist_ok=True)

        # Move images to the appropriate folders
        for image in train_images:
            src = os.path.join(dataset_folder, image)
            dst = os.path.join(train_label_path, image)
            shutil.move(src, dst)

        for image in val_images:
            src = os.path.join(dataset_folder, image)
            dst = os.path.join(val_label_path, image)
            shutil.move(src, dst)

    print("Dataset splitting completed successfully.")

    # Create a new YOLO model from scratch
    model = YOLO(os.path.join(output_directory,'yolov8n-cls.pt'))
    #
    #define parameters for YOLO training, be aware of epoch, batch, and imgsz, to not exceed system requirements (memory, CPU, GPU...)
    #Folder for training *bplusplus/data/train
    #Folder for validation *bplusplus/data/val
    #Specify path to folder where the val and train folder is located
    data = output_directory
    results = model.train(data=data, epochs=5, batch=16, imgsz=224, save_dir=output_directory)

    #batch is adjusted to 1 to prevent a resizing bug - in training this bug doesnt emerge. A work around for larger batch size could be a resizing step in advance.
    model.val(batch=1)



def __files_in_folder(folder: str) -> list[str]:
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
