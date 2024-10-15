import os
import random
from typing import Any, Optional
import pygbif
import requests
import validators
from .collect_images import collect_images, Group
import tempfile
from pathlib import Path
from src.yolov5detect import detect
import shutil
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def collect_and_prepare(group_by_key: Group, search_parameters: dict[str, Any], images_per_group: int, output_directory: str):

    groups: list[str] = search_parameters[group_by_key.value]

    class_mapping={}

    output_directory = Path(output_directory)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir_path = Path(temp_dir)
        images_path = temp_dir_path / "images"
        inference_path = temp_dir_path / "inference"
        labels_path = temp_dir_path / "labels"
        
        images_path.mkdir(parents=True, exist_ok=True)
        inference_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        print("Beginning to collect images from GBIF...")
        for group in groups:
            print(f"Collecting images for {group}...")
            occurrences_json = _fetch_occurrences(group_key=group_by_key, group_value=group, parameters=search_parameters, totalLimit=10000)
            optional_occurrences = map(lambda x: __parse_occurrence(x), occurrences_json)
            occurrences = list(filter(None, optional_occurrences))

            print(f"{group} : {len(occurrences)} parseable occurrences fetched, will sample for {images_per_group}")

            random.seed(42) # for reproducibility
            sampled_occurrences = random.sample(occurrences, min(images_per_group, len(occurrences)))

            print(f"Downloading {len(sampled_occurrences)} images into the {group} folder...")
            image_names=[]
            for occurrence in sampled_occurrences:
                #image_url = occurrence.image_url.replace("original", "large") # hack to get max 1024px image

                image_name= down_image(
                    url=occurrence.image_url,
                    group=group,
                    ID_name=occurrence.key,
                    folder=images_path
                )

                image_names.append(image_name)
            class_mapping[group] = image_names

        original_image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.jpeg")))

        __delete_corrupted_images(images_path)

        yaml_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5detect/insect.yaml')))
        weights_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5detect/acc94.pt')))

        detect.run(source=images_path, data=yaml_path, weights=weights_path, save_txt=True, project=temp_dir_path)

        __delete_orphaned_images_and_inferences(images_path, inference_path, labels_path)
        __delete_invalid_txt_files(images_path, inference_path, labels_path)
        class_idxs = update_labels(class_mapping, labels_path)
        __split_data(class_mapping, temp_dir_path, output_directory)

        __save_class_idx_to_file(class_idxs, output_directory)
        print('\n')
        final_image_count = count_images_across_splits(output_directory)
        print(f"\nOut of {original_image_count} input images, {final_image_count} are eligible for detection. \nThese are saved across train, test and valid split in {output_directory}.")
        __generate_sample_images_with_detections(output_directory)

        print("\nCollecting and splitting background images.")

        bg_images=int(final_image_count*0.06)

        search: dict[str, Any] = {
            "scientificName": ["Plantae"]
        }

        collect_images(
            group_by_key=Group.scientificName,
            search_parameters=search, 
            images_per_group=bg_images,
            output_directory=temp_dir_path
        )

        __delete_corrupted_images(temp_dir_path / "Plantae")

        __split_background_images(temp_dir_path / "Plantae", output_directory)

        __count_classes_and_output_table(output_directory, output_directory / 'class_idx.txt' )
    

def _fetch_occurrences(group_key: str, group_value: str, parameters: dict[str, Any], totalLimit: int) -> list[dict[str, Any]]:
    parameters[group_key] = group_value
    return __next_batch(
        parameters=parameters,
        total_limit=totalLimit,
        offset=0,
        current=[]
    ) 

def __next_batch(parameters: dict[str, Any], total_limit: int, offset: int, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parameters["limit"] = total_limit
        parameters["offset"] = offset
        parameters["mediaType"] = ["StillImage"]
        search = pygbif.occurrences.search(**parameters)
        occurrences = search["results"]
        if search["endOfRecords"] or len(current) >= total_limit:
            return current + occurrences
        else:
            offset = search["offset"]
            count = search["limit"] # this seems to be returning the count, and `count` appears to be returning the total number of results returned by the search
            return __next_batch(
                parameters=parameters,
                total_limit=total_limit,
                offset=offset + count,
                current=current + occurrences
            )
 
#function to download insect images
def down_image(url: str, group: str, ID_name: str, folder: str):
    directory = folder
    image_response = requests.get(url)
    image_name = f"{group}{ID_name}.jpg"  # You can modify the naming convention as per your requirements
    image_path = os.path.join(directory, image_name)
    with open(image_path, "wb") as f:
        f.write(image_response.content)
    return(image_name)
    # print(f"{image_name} downloaded successfully.")

def __delete_corrupted_images(images_folder):
    '''
    Takes a folder of images and deletes any corrupted images that cannot be opened.

    '''
    images_folder = Path(images_folder)
    
    for image_file in images_folder.iterdir():
        if image_file.is_file() and image_file.suffix in ['.jpg', '.jpeg', '.png']:  
            try:
                with Image.open(image_file) as img:
                    img.verify() 
            except (IOError, SyntaxError) as e:

                print(f"Deleting corrupted image: {image_file}")
                image_file.unlink()  

    print("Corrupted images (if any) have been deleted.")

def __delete_orphaned_images_and_inferences(images_path, inference_path, labels_path):

    ''' 
    Delete any image file that does not have a label file due to transfer errors.
    '''

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
    
    print(f"Orphaned images and inference files without corresponding labels have been deleted")

def __delete_invalid_txt_files(images_path, inference_path, labels_path):

    ''' 
    Delete images that have 0 or more than one detections.
    These types are more prone to inaccurate data.
    '''

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

    print(f"Invalid txt files and corresponding images/inference files have been deleted")

def update_labels(class_mapping, labels_path):

    ''' 
    Change every detection label to the corresponding class index.
    '''

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

def __split_data(class_mapping, temp_dir, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    ''' 
    Split data per class and copy to train, test, valid directories.
    
    '''

    images_dir = temp_dir / "images"
    labels_dir = temp_dir / "labels"

    def create_dirs(split):
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, split):
        for image_file in file_list:
            image_file_path = images_dir / image_file

            if not image_file_path.exists():
                continue  

            shutil.copy(image_file_path, output_dir / split / "images" / image_file_path.name)

            label_file = labels_dir / (image_file_path.stem + ".txt")
            if label_file.exists():
                shutil.copy(label_file, output_dir / split / "labels" / label_file.name)

    for split in ["train", "test", "valid"]:
        create_dirs(split)

    for _, files in class_mapping.items():
        random.shuffle(files) 
        num_files = len(files)


        train_count = int(split_ratios[0] * num_files)
        test_count = int(split_ratios[1] * num_files)
        valid_count = num_files - train_count - test_count

        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        valid_files = files[train_count + test_count:]

        copy_files(train_files, "train")
        copy_files(test_files, "test")
        copy_files(valid_files, "valid")

    print(f"Data has been split into train, test, and valid.")


def __save_class_idx_to_file(class_idx, output_dir, filename="class_idx.txt"):

    ''' 
    Convert dictionary to txt file and save in output directory.
    '''
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename

    with open(file_path, 'w') as file:
        for idx, class_name in class_idx.items():
            file.write(f"{idx}: {class_name}\n")

    print(f"class_idx has been saved to {file_path}")

def count_images_across_splits(output_dir):

    ''' 
    Count the total number of images across train, test, valid. 
    '''

    total_count = 0
    splits = ["train", "test", "valid"]
    
    for split in splits:
        images_path = output_dir / split / "images"
        image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.jpeg")))
        print(f"Number of images in {split}: {image_count}")
        total_count += image_count
    
    return total_count

def __count_classes_and_output_table(main_dir, class_ids_file):
    """
    Counts the class occurrences in train, test, and valid splits and outputs the results in a pretty table.
    Background images with empty txt files are counted as 'null' class.
    """

    def read_class_ids(class_ids_file):
        class_ids = {}
        with open(class_ids_file, 'r') as f:
            for line in f:
                class_index, class_name = line.strip().split(': ')
                class_ids[int(class_index)] = class_name
        return class_ids

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
    class_ids = read_class_ids(class_ids_file)
    total_counts = defaultdict(int)

    table = PrettyTable()
    table.field_names = ["Class", "Class Index", "Train Count", "Test Count", "Valid Count", "Total"]

    split_counts = {split: defaultdict(int) for split in splits}

    for split in splits:
        labels_dir = os.path.join(main_dir, split, 'labels')
        if not os.path.exists(labels_dir):
            print(f"Warning: {labels_dir} does not exist, skipping {split}.")
            continue

        class_counts = count_classes_in_split(labels_dir)
        for class_index, count in class_counts.items():
            split_counts[split][class_index] = count
            total_counts[class_index] += count

    for class_index, total in total_counts.items():
        class_name = class_ids.get(class_index, "Background" if class_index == 'null' else f"Class {class_index}")
        train_count = split_counts['train'].get(class_index, 0)
        test_count = split_counts['test'].get(class_index, 0)
        valid_count = split_counts['valid'].get(class_index, 0)
        table.add_row([class_name, class_index, train_count, test_count, valid_count, total])

    print(table)

# def __count_classes_and_output_table(main_dir, class_ids_file):
#     """
#     Counts the class occurrences in train, test, and valid splits and outputs the results in a pretty table.
#     """

#     def read_class_ids(class_ids_file):
#         class_ids = {}
#         with open(class_ids_file, 'r') as f:
#             for line in f:
#                 class_index, class_name = line.strip().split(': ')
#                 class_ids[int(class_index)] = class_name
#         return class_ids

#     def count_classes_in_split(labels_dir):
#         class_counts = defaultdict(int)
#         for label_file in os.listdir(labels_dir):
#             if label_file.endswith(".txt"):
#                 with open(os.path.join(labels_dir, label_file), 'r') as f:
#                     for line in f:
#                         class_index = int(line.split()[0])
#                         class_counts[class_index] += 1
#         return class_counts

#     splits = ['train', 'test', 'valid']
#     class_ids = read_class_ids(class_ids_file)
#     total_counts = defaultdict(int)

#     table = PrettyTable()
#     table.field_names = ["Class", "Class Index", "Train Count", "Test Count", "Valid Count", "Total"]

#     split_counts = {split: defaultdict(int) for split in splits}

#     for split in splits:
#         labels_dir = os.path.join(main_dir, split, 'labels')
#         if not os.path.exists(labels_dir):
#             print(f"Warning: {labels_dir} does not exist, skipping {split}.")
#             continue
        
#         class_counts = count_classes_in_split(labels_dir)
#         for class_index, count in class_counts.items():
#             split_counts[split][class_index] = count
#             total_counts[class_index] += count

#     for class_index, total in total_counts.items():
#         class_name = class_ids.get(class_index, f"Class {class_index}")
#         train_count = split_counts['train'].get(class_index, 0)
#         test_count = split_counts['test'].get(class_index, 0)
#         valid_count = split_counts['valid'].get(class_index, 0)
#         table.add_row([class_name, class_index, train_count, test_count, valid_count, total])

#     print(table)

def __generate_sample_images_with_detections(main_dir):
    """
    Generates one sample image with multiple detections for each of train, test, valid, combining up to 6 images in one output.
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

    def read_class_ids(class_ids_file):
        class_mapping = {}
        with open(class_ids_file, 'r') as f:
            for line in f:
                class_idx, class_name = line.strip().split(': ')
                class_mapping[int(class_idx)] = class_name
        return class_mapping

    splits = ['train', 'test', 'valid']
    class_idx_file = os.path.join(main_dir, 'class_idx.txt')
    class_mapping = read_class_ids(class_idx_file)
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

def __split_background_images(background_dir, output_dir):

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
            shutil.copy(image_file, Path(output_dir) / split / 'images' / image_file.name)

            label_file = Path(output_dir) / split / 'labels' / (image_file.stem + ".txt")
            label_file.touch()  

    copy_files(train_files, 'train')
    copy_files(valid_files, 'valid')
    copy_files(test_files, 'test')

    print(f"Background data has been split: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")


class Occurrence:

    def __init__(self, key: str, image_url: str) -> None:
         self.key = key
         self.image_url = image_url
         

def __parse_occurrence(json: dict[str, Any]) -> Optional[Occurrence]:
    if (key := json.get("key", str)) is not None \
        and (image_url := json.get("media", {})[0].get("identifier", str)) is not None \
            and validators.url(image_url):
         
         return Occurrence(key=key, image_url=image_url)
    else:
         return None