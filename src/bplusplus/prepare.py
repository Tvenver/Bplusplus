import os
import random
from typing import Any, Optional
import requests
import tempfile
from .collect import Group, collect
from pathlib import Path
from .yolov5detect.detect import run
import shutil
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
import yaml

def prepare(input_directory: str, output_directory: str, with_background: bool = False):

    """
    Prepares the dataset for training by performing the following steps:
    1. Copies images from the input directory to a temporary directory.
    2. Deletes corrupted images.
    3. Downloads YOLOv5 weights if not already present.
    4. Runs YOLOv5 inference to generate labels for the images.
    5. Deletes orphaned images and inferences.
    6. Updates labels based on class mapping.
    7. Splits the data into train, test, and validation sets.
    8. Counts the total number of images across all splits.
    9. Makes a YAML configuration file for YOLOv8.

    Args:
        input_directory (str): The path to the input directory containing the images.
        output_directory (str): The path to the output directory where the prepared dataset will be saved.
    """

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    class_mapping={}

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir_path = Path(temp_dir)
        images_path = temp_dir_path / "images"
        inference_path = temp_dir_path / "inference"
        labels_path = temp_dir_path / "labels"

        images_path.mkdir(parents=True, exist_ok=True)
        inference_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        for folder_directory in input_directory.iterdir():
            images_names = []
            if folder_directory.is_dir():
                folder_name = folder_directory.name
                for image_file in folder_directory.glob("*.jpg"):
                    shutil.copy(image_file, images_path)
                    image_name = image_file.name
                    images_names.append(image_name)

                class_mapping[folder_name] = images_names   

        original_image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.jpeg")))

        __delete_corrupted_images(images_path)

        current_dir = Path(__file__).resolve().parent
        yaml_path = current_dir / 'yolov5detect' / 'insect.yaml'
        weights_path = current_dir / 'yolov5detect' / 'acc94.pt'

        github_release_url = 'https://github.com/Tvenver/Bplusplus/releases/download/v0.1.2/acc94.pt'

        if not weights_path.exists():
            __download_file_from_github_release(github_release_url, weights_path)

        run(source=images_path, data=yaml_path, weights=weights_path, save_txt=True, project=temp_dir_path)

        __delete_orphaned_images_and_inferences(images_path, inference_path, labels_path)
        __delete_invalid_txt_files(images_path, inference_path, labels_path)
        class_idxs = update_labels(class_mapping, labels_path)
        __split_data(class_mapping, temp_dir_path, output_directory)

        # __save_class_idx_to_file(class_idxs, output_directory)
        final_image_count = count_images_across_splits(output_directory)
        print(f"\nOut of {original_image_count} input images, {final_image_count} are eligible for detection. \nThese are saved across train, test and valid split in {output_directory}.")
        __generate_sample_images_with_detections(output_directory, class_idxs)

        if with_background:
            print("\nCollecting and splitting background images.")

            bg_images=int(final_image_count*0.06)

            search: dict[str, Any] = {
                "scientificName": ["Plantae"]
            }

            collect(
                group_by_key=Group.scientificName,
                search_parameters=search, 
                images_per_group=bg_images,
                output_directory=temp_dir_path
            )

            __delete_corrupted_images(temp_dir_path / "Plantae")

            __split_background_images(temp_dir_path / "Plantae", output_directory)

        __count_classes_and_output_table(output_directory, class_idxs)

        __make_yaml_file(output_directory, class_idxs)

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

def __delete_orphaned_images_and_inferences(images_path: Path, inference_path: Path, labels_path: Path):
    
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
        inference_file_jpg = inference_path / (txt_file.stem + ".jpg")
        inference_file_jpeg = inference_path / (txt_file.stem + ".jpeg")

        if not (image_file_jpg.exists() or image_file_jpeg.exists()):
            print(f"Deleting {txt_file.name} - No corresponding image file")
            txt_file.unlink()
        elif not (inference_file_jpg.exists() or inference_file_jpeg.exists()):
            print(f"Deleting {txt_file.name} - No corresponding inference file")
            txt_file.unlink()
            
    label_stems = {txt_file.stem for txt_file in labels_path.glob("*.txt")}
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg"))

    for image_file in image_files:
        if image_file.stem not in label_stems:
            print(f"Deleting orphaned image: {image_file.name}")
            image_file.unlink()

            inference_file_jpg = inference_path / (image_file.stem + ".jpg")
            inference_file_jpeg = inference_path / (image_file.stem + ".jpeg")

            if inference_file_jpg.exists():
                inference_file_jpg.unlink()
                print(f"Deleted corresponding inference file: {inference_file_jpg.name}")
            elif inference_file_jpeg.exists():
                inference_file_jpeg.unlink()
                print(f"Deleted corresponding inference file: {inference_file_jpeg.name}")

    print("Orphaned images and inference files without corresponding labels have been deleted.")

def __delete_invalid_txt_files(images_path: Path, inference_path: Path, labels_path: Path):

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
            print(f"Deleting {txt_file.name} - Invalid file")
            txt_file.unlink()

            image_file_jpg = images_path / (txt_file.stem + ".jpg")
            image_file_jpeg = images_path / (txt_file.stem + ".jpeg")
            inference_file_jpg = inference_path / (txt_file.stem + ".jpg")
            inference_file_jpeg = inference_path / (txt_file.stem + ".jpeg")

            if image_file_jpg.exists():
                image_file_jpg.unlink()
                print(f"Deleted corresponding image file: {image_file_jpg.name}")
            elif image_file_jpeg.exists():
                image_file_jpeg.unlink()
                print(f"Deleted corresponding image file: {image_file_jpeg.name}")

            if inference_file_jpg.exists():
                inference_file_jpg.unlink()
                print(f"Deleted corresponding inference file: {inference_file_jpg.name}")
            elif inference_file_jpeg.exists():
                inference_file_jpeg.unlink()
                print(f"Deleted corresponding inference file: {inference_file_jpeg.name}")

    print("Invalid text files and their corresponding images and inference files have been deleted.")


def __split_data(class_mapping: dict, temp_dir_path: Path, output_directory: Path):
    """
    Splits the data into train, test, and validation sets.

    Args:
        class_mapping (dict): A dictionary mapping class names to image file names.
        temp_dir_path (Path): The path to the temporary directory containing the images.
        output_directory (Path): The path to the output directory where the split data will be saved.
    """
    images_dir = temp_dir_path / "images"
    labels_dir = temp_dir_path / "labels"

    def create_dirs(split):
        (output_directory / split).mkdir(parents=True, exist_ok=True)
        (output_directory / split / "images").mkdir(parents=True, exist_ok=True)
        (output_directory / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, split):
        for image_file in file_list:
            image_file_path = images_dir / image_file

            if not image_file_path.exists():
                continue  

            shutil.copy(image_file_path, output_directory / split / "images" / image_file_path.name)

            label_file = labels_dir / (image_file_path.stem + ".txt")
            if label_file.exists():
                shutil.copy(label_file, output_directory / split / "labels" / label_file.name)

    for split in ["train", "test", "valid"]:
        create_dirs(split)

    for _, files in class_mapping.items():
        random.shuffle(files) 
        num_files = len(files)

        train_count = int(0.8 * num_files)
        test_count = int(0.1 * num_files)
        valid_count = num_files - train_count - test_count

        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        valid_files = files[train_count + test_count:]

        copy_files(train_files, "train")
        copy_files(test_files, "test")
        copy_files(valid_files, "valid")

    print("Data has been split into train, test, and valid.")

def __save_class_idx_to_file(class_idxs: dict, output_directory: Path):
    """
    Saves the class indices to a file.

    Args:
        class_idxs (dict): A dictionary mapping class names to class indices.
        output_directory (Path): The path to the output directory where the class index file will be saved.
    """
    class_idx_file = output_directory / "class_idx.txt"
    with open(class_idx_file, 'w') as f:
        for class_name, idx in class_idxs.items():
            f.write(f"{class_name}: {idx}\n")
    print(f"Class indices have been saved to {class_idx_file}")

def __generate_sample_images_with_detections(main_dir: Path, class_idxs: dict):

    """
    Generates one sample image with multiple detections for each of train, test, valid, combining up to 6 images in one output.

    Args:
        main_dir (str): The main directory containing the train, test, and valid splits.
    """

    def resize_and_contain(image, target_size):
        image.thumbnail(target_size, Image.LANCZOS)
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        new_image.paste(image, ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2))
        return new_image

    def draw_bounding_boxes(image, labels_path, class_mapping, color_map):
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        if labels_path.exists():
            with open(labels_path, 'r') as label_file:
                for line in label_file.readlines():
                    parts = line.strip().split()
                    class_idx = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])
                    x_min = int((center_x - width / 2) * img_width)
                    y_min = int((center_y - height / 2) * img_height)
                    x_max = int((center_x + width / 2) * img_width)
                    y_max = int((center_y + height / 2) * img_height)
                    class_name = class_mapping.get(class_idx, str(class_idx))
                    color = color_map[class_idx]
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                    draw.text((x_min, y_min - 20), class_name, fill=color, font=font)
        return image

    def combine_images(images, grid_size=(3, 2), target_size=(416, 416)):
        resized_images = [resize_and_contain(img, target_size) for img in images]
        width, height = target_size
        combined_image = Image.new('RGB', (width * grid_size[0], height * grid_size[1]))

        for i, img in enumerate(resized_images):
            row = i // grid_size[0]
            col = i % grid_size[0]
            combined_image.paste(img, (col * width, row * height))
        
        return combined_image

    def generate_color_map(class_mapping):
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan', 'magenta']
        color_map = {idx: random.choice(colors) for idx in class_mapping.keys()}
        return color_map

    splits = ['train', 'test', 'valid']
    class_mapping = class_idxs
    color_map = generate_color_map(class_mapping)

    for split in splits:
        images_dir = Path(main_dir) / split / 'images'
        labels_dir = Path(main_dir) / split / 'labels'
        image_files = list(images_dir.glob("*.jpg"))
        if not image_files:
            continue
        
        sample_images = []
        for image_file in image_files[:6]:
            label_file = labels_dir / (image_file.stem + '.txt')
            image = Image.open(image_file)
            image_with_boxes = draw_bounding_boxes(image, label_file, class_mapping, color_map)
            sample_images.append(image_with_boxes)
        
        if sample_images:
            combined_image = combine_images(sample_images, grid_size=(3, 2), target_size=(416, 416))
            combined_image_path = Path(main_dir) / split / f"{split}_sample_with_detections.jpg"
            combined_image.save(combined_image_path)
    

def __split_background_images(background_dir: Path, output_directory: Path):
    """
    Splits the background images into train, test, and validation sets.

    Args:
        temp_dir_path (Path): The path to the temporary directory containing the background images.
        output_directory (Path): The path to the output directory where the split background images will be saved.
    """

    image_files = list(Path(background_dir).glob("*.jpg"))
    random.shuffle(image_files)

    num_images = len(image_files)
    train_split = int(0.8 * num_images)
    valid_split = int(0.1 * num_images)

    train_files = image_files[:train_split]
    valid_files = image_files[train_split:train_split + valid_split]
    test_files = image_files[train_split + valid_split:]

    def copy_files(image_list, split):
        for image_file in image_list:
            shutil.copy(image_file, Path(output_directory) / split / 'images' / image_file.name)

            label_file = Path(output_directory) / split / 'labels' / (image_file.stem + ".txt")
            label_file.touch()  

    copy_files(train_files, 'train')
    copy_files(valid_files, 'valid')
    copy_files(test_files, 'test')

    print(f"Background data has been split: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")
    

def __count_classes_and_output_table(output_directory: Path, class_idxs: dict):
    """
    Counts the number of images per class and outputs a table.

    Args:
        output_directory (Path): The path to the output directory containing the split data.
        class_idxs (dict): A dictionary mapping class indices to class names.
    """

    def count_classes_in_split(labels_dir):
        class_counts = defaultdict(int)
        for label_file in os.listdir(labels_dir):
            if label_file.endswith(".txt"):
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        # Count empty files as 'null' class (background images)
                        class_counts['null'] += 1
                    else:
                        for line in lines:
                            class_index = int(line.split()[0])
                            class_counts[class_index] += 1
        return class_counts

    splits = ['train', 'test', 'valid']
    total_counts = defaultdict(int)

    table = PrettyTable()
    table.field_names = ["Class", "Class Index", "Train Count", "Test Count", "Valid Count", "Total"]

    split_counts = {split: defaultdict(int) for split in splits}

    for split in splits:
        labels_dir = output_directory / split / 'labels'
        if not os.path.exists(labels_dir):
            print(f"Warning: {labels_dir} does not exist, skipping {split}.")
            continue

        class_counts = count_classes_in_split(labels_dir)
        for class_index, count in class_counts.items():
            split_counts[split][class_index] = count
            total_counts[class_index] += count

    for class_index, total in total_counts.items():
        class_name = class_idxs.get(class_index, "Background" if class_index == 'null' else f"Class {class_index}")
        train_count = split_counts['train'].get(class_index, 0)
        test_count = split_counts['test'].get(class_index, 0)
        valid_count = split_counts['valid'].get(class_index, 0)
        table.add_row([class_name, class_index, train_count, test_count, valid_count, total])

    print(table)
def update_labels(class_mapping: dict, labels_path: Path) -> dict:
    """
    Updates the labels based on the class mapping.

    Args:
        class_mapping (dict): A dictionary mapping class names to image file names.
        labels_path (Path): The path to the directory containing the label files.

    Returns:
        dict: A dictionary mapping class names to class indices.
    """
    class_index_mapping = {}
    class_index_definition = {}

    for idx, (class_name, images) in enumerate(class_mapping.items()):
        class_index_definition[idx] = class_name
        for image_name in images:
            class_index_mapping[image_name] = idx

    for txt_file in labels_path.glob("*.txt"):
        image_name_jpg = txt_file.stem + ".jpg"
        image_name_jpeg = txt_file.stem + ".jpeg"

        if image_name_jpg in class_index_mapping:
            class_index = class_index_mapping[image_name_jpg]
        elif image_name_jpeg in class_index_mapping:
            class_index = class_index_mapping[image_name_jpeg]
        else:
            print(f"Warning: No corresponding image found for {txt_file.name}")
            continue

        with open(txt_file, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) > 0:
                parts[0] = str(class_index)
                updated_lines.append(" ".join(parts))

        with open(txt_file, 'w') as file:
            file.write("\n".join(updated_lines))

    print(f"Labels updated successfully")
    return class_index_definition

def count_images_across_splits(output_directory: Path) -> int:
    """
    Counts the total number of images across train, test, and validation splits.

    Args:
        output_directory (Path): The path to the output directory containing the split data.

    Returns:
        int: The total number of images across all splits.
    """
    total_images = 0
    for split in ['train', 'test', 'valid']:
        split_dir = output_directory / split / 'images'
        total_images += len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.jpeg")))

    return total_images

def __make_yaml_file(output_directory: Path, class_idxs: dict):
    """
    Creates a YAML configuration file for YOLOv8.

    Args:
        output_directory (Path): The path to the output directory where the YAML file will be saved.
        class_idxs (dict): A dictionary mapping class indices to class names.
    """

    # Define the structure of the YAML file
    yaml_content = {
        'path': str(output_directory.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {idx: name for idx, name in class_idxs.items()}
    }

    # Write the YAML content to a file
    yaml_file_path = output_directory / 'dataset.yaml'
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"YOLOv8 YAML file created at {yaml_file_path}")