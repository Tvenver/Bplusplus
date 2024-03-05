from ultralytics import YOLO
import os
import shutil
import random

# Define paths
dataset_path = os.path.join('data', 'dataset')  # Path to your dataset
train_path = os.path.join('data', 'train')  # Path to the training folder
val_path = os.path.join('data', 'val')   # Path to the validation folder

# Define the ratio for splitting the dataset
split_ratio = 0.8  # 80% for training, 20% for validation

# Create training and validation directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Walk through the dataset directory
for root, dirs, files in os.walk(dataset_path):
    for label in dirs:
        label_path = os.path.join(root, label)
        images = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]

        # Shuffle the images
        random.shuffle(images)

        # Calculate the split index
        split_index = int(len(images) * split_ratio)

        # Split the images into training and validation sets
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create destination folders if they don't exist
        train_label_path = os.path.join(train_path, label)
        val_label_path = os.path.join(val_path, label)
        os.makedirs(train_label_path, exist_ok=True)
        os.makedirs(val_label_path, exist_ok=True)

        # Move images to the appropriate folders
        for image in train_images:
            src = os.path.join(label_path, image)
            dst = os.path.join(train_label_path, image)
            shutil.move(src, dst)

        for image in val_images:
            src = os.path.join(label_path, image)
            dst = os.path.join(val_label_path, image)
            shutil.move(src, dst)

print("Dataset splitting completed successfully.")

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')
#
#define parameters for YOLO training, be aware of epoch, batch, and imgsz, to not exceed system requirements (memory, CPU, GPU...)
#Folder for training *bplusplus/data/train
#Folder for validation *bplusplus/data/val
#
data = "C:/Users/titusvenverloo/Documents/GitHub/Bplusplus/data/"
results = model.train(data=data, epochs=5, batch=16, imgsz=224)

model.val()

