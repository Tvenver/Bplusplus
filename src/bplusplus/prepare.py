import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import requests
import torch
from PIL import Image
from torch import serialization
from torch.nn import Module, ModuleDict, ModuleList
from torch.nn.modules.activation import LeakyReLU, ReLU, SiLU
# Add more modules to prevent further errors
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF, Bottleneck, C2f, Concat, Detect
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel


def prepare(input_directory: str, output_directory: str, img_size: int = 40):
    """
    Prepares a YOLO classification dataset by performing the following steps:
    1. Copies images from input directory to temporary directory and creates class mapping.
    2. Deletes corrupted images and downloads YOLO model weights if not present.
    3. Runs YOLO inference to generate detection labels (bounding boxes) for the images.
    4. Cleans up orphaned images, invalid labels, and updates labels with class indices.
    5. Crops detected objects from images based on bounding boxes and resizes them.
    6. Splits data into train/valid sets with classification folder structure (train/class_name/image.jpg).

    Args:
        input_directory (str): The path to the input directory containing the images.
        output_directory (str): The path to the output directory where the prepared classification dataset will be saved.
        img_size (int, optional): The target size for the smallest dimension of cropped images. Defaults to 40.
    """
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    print("="*60)
    print("STARTING BPLUSPLUS DATASET PREPARATION")
    print("="*60)
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Target image size: {img_size}px (smallest dimension)")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        print(f"Using temporary directory: {temp_dir_path}")
        print()
        
        # Step 1: Setup directories and copy images
        print("STEP 1: Setting up directories and copying images...")
        print("-" * 50)
        class_mapping, original_image_count = _setup_directories_and_copy_images(
            input_directory, temp_dir_path
        )
        print(f"✓ Step 1 completed: {original_image_count} images copied from {len(class_mapping)} classes")
        print()
        
        # Step 2-3: Clean images and setup model
        print("STEP 2: Cleaning images and setting up YOLO model...")
        print("-" * 50)
        weights_path = _prepare_model_and_clean_images(temp_dir_path)
        print(f"✓ Step 2 completed: Model ready at {weights_path}")
        print()
        
        # Step 4: Run YOLO inference
        print("STEP 3: Running YOLO inference to detect objects...")
        print("-" * 50)
        labels_path = _run_yolo_inference(temp_dir_path, weights_path)
        print(f"✓ Step 3 completed: Labels generated at {labels_path}")
        print()
        
        # Step 5-6: Clean up labels and update class mapping
        print("STEP 4: Cleaning up orphaned files and processing labels...")
        print("-" * 50)
        class_idxs = _cleanup_and_process_labels(
            temp_dir_path, labels_path, class_mapping
        )
        print(f"✓ Step 4 completed: Processed {len(class_idxs)} classes")
        print()
        
        # Step 7-9: Finalize dataset
        print("STEP 5: Creating classification dataset with cropped images...")
        print("-" * 50)
        _finalize_dataset(
            class_mapping, temp_dir_path, output_directory, 
            class_idxs, original_image_count, img_size
        )
        print("✓ Step 5 completed: Classification dataset ready!")
        print()
        
    print("="*60)
    print("BPLUSPLUS DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*60)

def _setup_directories_and_copy_images(input_directory: Path, temp_dir_path: Path):
    """
    Sets up temporary directories and copies images from input directory.
    
    Returns:
        tuple: (class_mapping dict, original_image_count int)
    """
    images_path = temp_dir_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    print(f"  Created temporary images directory: {images_path}")
    
    class_mapping = {}
    total_copied = 0
    
    print("  Scanning input directory for class folders...")
    class_folders = [d for d in input_directory.iterdir() if d.is_dir()]
    print(f"  Found {len(class_folders)} class folders")
    
    for folder_directory in class_folders:
        images_names = []
        if folder_directory.is_dir():
            folder_name = folder_directory.name
            image_files = list(folder_directory.glob("*.jpg"))
            print(f"  Copying {len(image_files)} images from class '{folder_name}'...")
            
            for image_file in image_files:
                shutil.copy(image_file, images_path)
                image_name = image_file.name
                images_names.append(image_name)
                total_copied += 1
            
            class_mapping[folder_name] = images_names
            print(f"    ✓ {len(images_names)} images copied for class '{folder_name}'")
    
    original_image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.jpeg")))
    print(f"  Total images in temporary directory: {original_image_count}")
    
    return class_mapping, original_image_count

def _prepare_model_and_clean_images(temp_dir_path: Path):
    """
    Cleans corrupted images and downloads/prepares the YOLO model.
    
    Returns:
        Path: weights_path for the YOLO model
    """
    images_path = temp_dir_path / "images"
    
    # Clean corrupted images
    print("  Checking for corrupted images...")
    images_before = len(list(images_path.glob("*.jpg")))
    __delete_corrupted_images(images_path)
    images_after = len(list(images_path.glob("*.jpg")))
    deleted_count = images_before - images_after
    print(f"  ✓ Cleaned {deleted_count} corrupted images ({images_after} images remain)")
    
    # Setup model weights
    current_dir = Path(__file__).resolve().parent
    weights_path = current_dir / 'v11small-generic.pt'
    github_release_url = 'https://github.com/Tvenver/Bplusplus/releases/download/v1.2.3/v11small-generic.pt'
    
    print(f"  Checking for YOLO model weights at: {weights_path}")
    if not weights_path.exists():
        print("  Model weights not found, downloading from GitHub...")
        __download_file_from_github_release(github_release_url, weights_path)
        print(f"  ✓ Model weights downloaded successfully")
    else:
        print("  ✓ Model weights already exist")
    
    # Add all required classes to safe globals
    if hasattr(serialization, 'add_safe_globals'):
        serialization.add_safe_globals([
            DetectionModel, Sequential, Conv, Conv2d, BatchNorm2d, 
            SiLU, ReLU, LeakyReLU, MaxPool2d, Linear, Dropout, Upsample,
            Module, ModuleList, ModuleDict,
            Bottleneck, C2f, SPPF, Detect, Concat, DFL,
            # Add torch internal classes
            torch.nn.parameter.Parameter,
            torch.Tensor,
            torch._utils._rebuild_tensor_v2,
            torch._utils._rebuild_parameter
        ])
    
    return weights_path

def _run_yolo_inference(temp_dir_path: Path, weights_path: Path):
    """
    Runs YOLO inference on all images to generate labels.
    
    Returns:
        Path: labels_path where the generated labels are stored
    """
    images_path = temp_dir_path / "images"
    labels_path = temp_dir_path / "predict" / "labels"
    
    try:
        print(f"  Loading YOLO model from: {weights_path}")
        model = YOLO(weights_path)
        print("  ✓ YOLO model loaded successfully")
        
        # Get list of all image files
        image_files = list(images_path.glob('*.jpg'))
        print(f"  Found {len(image_files)} images to process with YOLO")
        
        # Ensure predict directory exists
        predict_dir = temp_dir_path / "predict"
        predict_dir.mkdir(exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created prediction output directory: {predict_dir}")
        
        result_count = 0
        error_count = 0
        
        print("  Starting YOLO inference...")
        print(f"  Progress: 0/{len(image_files)} images processed", end="", flush=True)
        
        for i, img_path in enumerate(image_files, 1):
            try:
                results = model.predict(
                    source=str(img_path),
                    conf=0.35,
                    save=True,
                    save_txt=True,
                    project=temp_dir_path,
                    name="predict",
                    exist_ok=True,
                    verbose=False  # Set to False to reduce YOLO's own output
                )
                
                result_count += 1
                
                # Update progress every 10% or every 100 images, whichever is smaller
                update_interval = max(1, min(100, len(image_files) // 10))
                if i % update_interval == 0 or i == len(image_files):
                    print(f"\r  Progress: {i}/{len(image_files)} images processed", end="", flush=True)
                
            except Exception as e:
                error_count += 1
                print(f"\n  Error processing {img_path.name}: {e}")
                continue
        
        print()  # New line after progress
        print(f"  ✓ YOLO inference completed: {result_count} successful, {error_count} failed")
        
        # Verify labels were created
        label_files = list(labels_path.glob("*.txt"))
        print(f"  Generated {len(label_files)} label files")
        
        if len(label_files) == 0:
            print("WARNING: No label files were created by the model prediction!")
            
    except Exception as e:
        print(f"Error during model prediction setup: {e}")
        import traceback
        traceback.print_exc()
    
    return labels_path

def _cleanup_and_process_labels(temp_dir_path: Path, labels_path: Path, class_mapping: dict):
    """
    Cleans up orphaned images and invalid labels, then creates class index mapping.
    
    Returns:
        dict: class_idxs mapping class indices to class names
    """
    images_path = temp_dir_path / "images"
    
    print("  Cleaning up orphaned images and labels...")
    images_before = len(list(images_path.glob("*.jpg")))
    labels_before = len(list(labels_path.glob("*.txt")))
    
    __delete_orphaned_images_and_inferences(images_path, labels_path)
    __delete_invalid_txt_files(images_path, labels_path)
    
    images_after = len(list(images_path.glob("*.jpg")))
    labels_after = len(list(labels_path.glob("*.txt")))
    
    deleted_images = images_before - images_after
    deleted_labels = labels_before - labels_after
    print(f"  ✓ Cleaned up {deleted_images} orphaned images and {deleted_labels} invalid labels")
    print(f"  Final counts: {images_after} images, {labels_after} valid labels")
    
    # Create class index mapping for classification
    class_idxs = {}
    for idx, class_name in enumerate(class_mapping.keys()):
        class_idxs[idx] = class_name
    
    print(f"  Created class mapping for {len(class_idxs)} classes: {list(class_idxs.values())}")
    
    return class_idxs

def _finalize_dataset(class_mapping: dict, temp_dir_path: Path, output_directory: Path, 
                     class_idxs: dict, original_image_count: int, img_size: int):
    """
    Finalizes the dataset by creating cropped classification images and splitting into train/valid sets.
    """
    # Split data into train/valid with cropped classification images
    __classification_split(class_mapping, temp_dir_path, output_directory, img_size)
    
    # Generate final report
    print("  Generating final statistics...")
    final_image_count = count_images_across_splits(output_directory)
    print(f"  Dataset Statistics:")
    print(f"    - Original images: {original_image_count}")
    print(f"    - Final cropped images: {final_image_count}")
    print(f"    - Success rate: {final_image_count/original_image_count*100:.1f}%")
    print(f"    - Output directory: {output_directory}")

def __delete_corrupted_images(images_path: Path):
     
    """
    Deletes corrupted images from the specified directory.

    Args:
        images_path (Path): The path to the directory containing images.

    This function iterates through all the image files in the specified directory
    and attempts to open each one. If an image file is found to be corrupted (i.e.,
    it cannot be opened), the function deletes the corrupted image file.
    """

    for image_file in images_path.glob("*.jpg"):
        try:
            Image.open(image_file)
        except IOError:
            image_file.unlink()

def __download_file_from_github_release(url, dest_path):

    """
    Downloads a file from a given GitHub release URL and saves it to the specified destination path,
    with a progress bar displayed in the terminal.

    Args:
        url (str): The URL of the file to download.
        dest_path (Path): The destination path where the file will be saved.

    Raises:
        Exception: If the file download fails.
    """

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    else:
        progress_bar.close()
        raise Exception(f"Failed to download file from {url}")

def __delete_orphaned_images_and_inferences(images_path: Path, labels_path: Path):
    
    """
    Deletes orphaned images and their corresponding inference files if they do not have a label file.

    Args:
        images_path (Path): The path to the directory containing images.
        inference_path (Path): The path to the directory containing inference files.
        labels_path (Path): The path to the directory containing label files.

    This function iterates through all the image files in the specified directory
    and checks if there is a corresponding label file. If an image file does not
    have a corresponding label file, the function deletes the orphaned image file
    and its corresponding inference file.
    """

    for txt_file in labels_path.glob("*.txt"):
        image_file_jpg = images_path / (txt_file.stem + ".jpg")
        image_file_jpeg = images_path / (txt_file.stem + ".jpeg")

        if not (image_file_jpg.exists() or image_file_jpeg.exists()):
            # print(f"Deleting {txt_file.name} - No corresponding image file")
            txt_file.unlink()
            
    label_stems = {txt_file.stem for txt_file in labels_path.glob("*.txt")}
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg"))

    for image_file in image_files:
        if image_file.stem not in label_stems:
            # print(f"Deleting orphaned image: {image_file.name}")
            image_file.unlink()



def __delete_invalid_txt_files(images_path: Path, labels_path: Path):

    """
    Deletes invalid text files and their corresponding image and inference files.

    Args:
        images_path (Path): The path to the directory containing images.
        inference_path (Path): The path to the directory containing inference files.
        labels_path (Path): The path to the directory containing label files.

    This function iterates through all the text files in the specified directory
    and checks if they have 0 or more than one detections. If a text file is invalid,
    the function deletes the invalid text file and its corresponding image and inference files.
    """

    for txt_file in labels_path.glob("*.txt"):
        with open(txt_file, 'r') as file:
            lines = file.readlines()

        if len(lines) == 0 or len(lines) > 1:
            # print(f"Deleting {txt_file.name} - Invalid file")
            txt_file.unlink()

            image_file_jpg = images_path / (txt_file.stem + ".jpg")
            image_file_jpeg = images_path / (txt_file.stem + ".jpeg")

            if image_file_jpg.exists():
                image_file_jpg.unlink()
                # print(f"Deleted corresponding image file: {image_file_jpg.name}")
            elif image_file_jpeg.exists():
                image_file_jpeg.unlink()
                # print(f"Deleted corresponding image file: {image_file_jpeg.name}")




def __classification_split(class_mapping: dict, temp_dir_path: Path, output_directory: Path, img_size: int):
    """
    Splits the data into train and validation sets for classification tasks,
    cropping images according to their YOLO labels but preserving original class structure.
    
    Args:
        class_mapping (dict): A dictionary mapping class names to image file names.
        temp_dir_path (Path): The path to the temporary directory containing the images.
        output_directory (Path): The path to the output directory where train and valid splits will be created.
        img_size (int): The target size for the smallest dimension of cropped images.
    """
    images_dir = temp_dir_path / "images"
    labels_dir = temp_dir_path / "predict" / "labels"
    
    # Create train and valid directories
    train_dir = output_directory / 'train'
    valid_dir = output_directory / 'valid'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories based on class_mapping
    print(f"  Creating train and validation directories for {len(class_mapping)} classes...")
    for class_name in class_mapping:
        (train_dir / class_name).mkdir(exist_ok=True)
        (valid_dir / class_name).mkdir(exist_ok=True)
        print(f"    ✓ Created directories for class: {class_name}")
    
    # Process each class folder and its images
    valid_images = []
    
    # First, collect all valid label files
    valid_label_stems = {label_file.stem for label_file in labels_dir.glob("*.txt") 
                        if label_file.exists() and os.path.getsize(label_file) > 0}
    
    print(f"  Found {len(valid_label_stems)} valid label files for cropping")
    
    print("  Starting image cropping and resizing...")
    total_processed = 0
    total_valid = 0
    
    for class_name, image_names in class_mapping.items():
        print(f"  Processing class '{class_name}' ({len(image_names)} images)...")
        class_processed = 0
        class_valid = 0
        
        for image_name in image_names:
            # Check if the image exists in the images directory
            image_path = images_dir / image_name
            class_processed += 1
            total_processed += 1
            
            if not image_path.exists():
                continue
                
            # Skip images that don't have a valid label
            if image_path.stem not in valid_label_stems:
                continue
                
            label_file = labels_dir / (image_path.stem + '.txt')
            
            try:
                img = Image.open(image_path)
                
                if label_file.exists():
                    # If label exists, crop the image
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            parts = lines[0].strip().split()
                            if len(parts) >= 5:
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                img_width, img_height = img.size
                                x_min = int((x_center - width/2) * img_width)
                                y_min = int((y_center - height/2) * img_height)
                                x_max = int((x_center + width/2) * img_width)
                                y_max = int((y_center + height/2) * img_height)
                                
                                x_min = max(0, x_min)
                                y_min = max(0, y_min)
                                x_max = min(img_width, x_max)
                                y_max = min(img_height, y_max)
                                
                                img = img.crop((x_min, y_min, x_max, y_max))
                
                img_width, img_height = img.size
                if img_width < img_height:
                    # Width is smaller, set to img_size
                    new_width = img_size
                    new_height = int((img_height / img_width) * img_size)
                else:
                    # Height is smaller, set to img_size
                    new_height = img_size
                    new_width = int((img_width / img_height) * img_size)
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                valid_images.append((image_path, img, class_name))
                class_valid += 1
                total_valid += 1
            except Exception as e:
                print(f"    Error processing {image_path}: {e}")
        
        print(f"    ✓ Class '{class_name}': {class_valid} valid images from {class_processed} processed")
    
    print(f"  ✓ Successfully processed {total_valid} valid images from {total_processed} total images")
    
    # Shuffle and split images
    print("  Shuffling and splitting images into train/validation sets...")
    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * 0.9)
    train_images = valid_images[:split_idx]
    valid_images_split = valid_images[split_idx:]
    
    print(f"  Split: {len(train_images)} training images, {len(valid_images_split)} validation images")
    
    # Save images to train/valid directories
    print("  Saving cropped and resized images...")
    saved_train = 0
    saved_valid = 0
    
    for image_set, dest_dir, split_name in [(train_images, train_dir, "train"), (valid_images_split, valid_dir, "valid")]:
        print(f"    Saving {len(image_set)} images to {split_name} set...")
        for orig_file, img, class_name in image_set:
            output_path = dest_dir / class_name / (orig_file.stem + '.jpg')
            
            # Convert any non-RGB mode to RGB before saving
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            img.save(output_path, format='JPEG', quality=95)
            
            if split_name == "train":
                saved_train += 1
            else:
                saved_valid += 1
    
    print(f"    ✓ Saved {saved_train} train images and {saved_valid} validation images")
    
    # Print detailed summary table
    print(f"  Final dataset summary:")
    print()
    
    # Calculate column widths for proper alignment
    max_class_name_length = max(len(class_name) for class_name in class_mapping.keys())
    class_col_width = max(max_class_name_length, len("Class"))
    
    # Print table header
    print(f"  {'Class':<{class_col_width}} | {'Train':<7} | {'Valid':<7} | {'Total':<7}")
    print(f"  {'-' * class_col_width}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")
    
    # Print data for each class and calculate totals
    total_train = 0
    total_valid = 0
    total_overall = 0
    
    for class_name in sorted(class_mapping.keys()):  # Sort for consistent output
        train_count = len(list((train_dir / class_name).glob('*.*')))
        valid_count = len(list((valid_dir / class_name).glob('*.*')))
        class_total = train_count + valid_count
        
        print(f"  {class_name:<{class_col_width}} | {train_count:<7} | {valid_count:<7} | {class_total:<7}")
        
        total_train += train_count
        total_valid += valid_count
        total_overall += class_total
    
    # Print totals row
    print(f"  {'-' * class_col_width}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")
    print(f"  {'TOTAL':<{class_col_width}} | {total_train:<7} | {total_valid:<7} | {total_overall:<7}")
    print()
    
    print(f"  ✓ Classification dataset created successfully at: {output_directory}")

def count_images_across_splits(output_directory: Path) -> int:
    """
    Counts the total number of images across train and validation splits for classification dataset.

    Args:
        output_directory (Path): The path to the output directory containing the split data.

    Returns:
        int: The total number of images across all splits.
    """
    total_images = 0
    for split in ['train', 'valid']:
        split_dir = output_directory / split
        if split_dir.exists():
            # Count all images in all class subdirectories
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    total_images += len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg")))

    return total_images