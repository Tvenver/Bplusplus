import os
import numpy as np
from PIL import Image
import shutil
import argparse
from scipy.spatial.distance import euclidean
from pathlib import Path


def compute_center_rgb(img_path):
    """
    Compute the average RGB color of the center 40% crop of an image.

    Args:
        img_path (str): Path to the image file

    Returns:
        np.array or None: RGB color values as numpy array, or None if error
    """
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        crop_box = (w * 0.3, h * 0.3, w * 0.7, h * 0.7)
        center_crop = img.crop(crop_box)
        pixels = np.array(center_crop).reshape(-1, 3)
        if pixels.size == 0:
            print(f"Warning: No pixels found in crop for {img_path}")
            return None
        return np.mean(pixels, axis=0)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def collect_images_from_directory(directory_path):
    """
    Collect all image files from a directory and compute their center RGB colors.

    Args:
        directory_path (str): Path to directory containing images

    Returns:
        list: List of tuples (color, filename, full_path)
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = []

    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(directory_path, filename)
        color = compute_center_rgb(img_path)
        if color is not None:
            images.append((color, filename, img_path))

    return images


def compute_outlier_threshold(color_vectors, keep_percent):
    """
    Compute the distance threshold for outlier detection.

    Args:
        color_vectors (np.array): Array of RGB color vectors
        keep_percent (float): Percentage of images to keep

    Returns:
        tuple: (median_color, distance_threshold)
    """
    median_color = np.median(color_vectors, axis=0)
    distances = [euclidean(color, median_color) for color in color_vectors]
    threshold = np.percentile(distances, keep_percent)
    return median_color, threshold


def process_single_directory(subdir_path, subdir_name, output_dir, deleted_dir, keep_percent, min_images=5):
    """
    Process a single subdirectory for outlier detection and file copying.

    Args:
        subdir_path (str): Path to the subdirectory to process
        subdir_name (str): Name of the subdirectory
        output_dir (str): Base output directory for cleaned images
        deleted_dir (str): Base directory for deleted images
        keep_percent (float): Percentage of images to keep
        min_images (int): Minimum number of images required for outlier detection

    Returns:
        tuple: (processed_count, kept_count, removed_count)
    """
    # Collect all images in this subdirectory
    all_images = collect_images_from_directory(subdir_path)

    if len(all_images) == 0:
        print(f"  {subdir_name}: No valid images found")
        return 0, 0, 0

    if len(all_images) < min_images:
        print(f"  {subdir_name}: Only {len(all_images)} images, copying all (no outlier detection)")
        # Copy all images to output directory
        output_subdir = os.path.join(output_dir, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)
        for _, filename, img_path in all_images:
            shutil.copy(img_path, os.path.join(output_subdir, filename))
        return len(all_images), len(all_images), 0

    print(f"  {subdir_name}: Processing {len(all_images)} images")

    # Compute outlier threshold
    color_vectors = np.array([img[0] for img in all_images])
    median_color, threshold = compute_outlier_threshold(color_vectors, keep_percent)

    # Create output subdirectories
    output_subdir = os.path.join(output_dir, subdir_name)
    deleted_subdir = os.path.join(deleted_dir, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(deleted_subdir, exist_ok=True)

    kept_count = 0
    removed_count = 0

    # Process each image
    for color, filename, img_path in all_images:
        distance = euclidean(color, median_color)

        if distance > threshold:
            # Remove as outlier
            output_filename = f"{distance:.2f}_{filename}"
            shutil.copy(img_path, os.path.join(deleted_subdir, output_filename))
            removed_count += 1
        else:
            # Keep image
            output_filename = f"{distance:.2f}_{filename}"
            shutil.copy(img_path, os.path.join(output_subdir, output_filename))
            kept_count += 1

    print(f"    Kept: {kept_count}, Removed: {removed_count}")
    return len(all_images), kept_count, removed_count


def setup_output_directories(output_dir, deleted_dir):
    """
    Create output directories if they don't exist.

    Args:
        output_dir (str): Directory for cleaned images
        deleted_dir (str): Directory for deleted images
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(deleted_dir, exist_ok=True)


def print_summary(total_processed, total_kept, total_removed):
    """
    Print processing summary statistics.

    Args:
        total_processed (int): Total number of images processed
        total_kept (int): Number of images kept
        total_removed (int): Number of images removed
    """
    print(f"\n=== Summary ===")
    print(f"Total images processed: {total_processed}")
    print(f"Images kept: {total_kept}")
    print(f"Images removed: {total_removed}")
    if total_processed > 0:
        print(f"Removal rate: {(total_removed / total_processed) * 100:.1f}%")


def clean_dataset_by_color_outliers(input_dir, output_dir, deleted_dir="deleted", keep_percent=98):
    """
    Clean a dataset by removing color outliers from each subdirectory.

    Args:
        input_dir (str): Directory containing subdirectories with images
        output_dir (str): Directory to save cleaned images (preserves subdirectory structure)
        deleted_dir (str): Directory to save removed outlier images (preserves subdirectory structure)
        keep_percent (float): Percentage of images to keep (removes (100-keep_percent)% outliers)
    """
    # Setup directories
    setup_output_directories(output_dir, deleted_dir)

    total_processed = 0
    total_kept = 0
    total_removed = 0

    print(f"Processing directories in {input_dir}...")
    print(f"Cleaned images will go to: {output_dir}")
    print(f"Deleted images will go to: {deleted_dir}")
    print(f"Keeping {keep_percent}% of images per subdirectory")
    print()

    # Process each subdirectory
    for subdir_name in sorted(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        processed, kept, removed = process_single_directory(
            subdir_path, subdir_name, output_dir, deleted_dir, keep_percent
        )

        total_processed += processed
        total_kept += kept
        total_removed += removed

    print_summary(total_processed, total_kept, total_removed)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Clean image dataset by removing color outliers from each subdirectory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_dir",
        help="Input directory containing subdirectories with images"
    )

    parser.add_argument(
        "-o", "--output-dir",
        default="cleaned",
        help="Output directory for cleaned images"
    )

    parser.add_argument(
        "-d", "--deleted-dir",
        default="deleted",
        help="Output directory for deleted outlier images"
    )

    parser.add_argument(
        "-k", "--keep-percent",
        type=float,
        default=98.0,
        help="Percentage of images to keep per subdirectory (1-99)"
    )

    return parser.parse_args()


def main():
    """Main function for command line execution."""
    args = parse_arguments()

    # Validate arguments
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1

    if not (1 <= args.keep_percent <= 99):
        print(f"Error: keep_percent must be between 1 and 99, got {args.keep_percent}")
        return 1

    # Run the cleaning process
    try:
        clean_dataset_by_color_outliers(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            deleted_dir=args.deleted_dir,
            keep_percent=args.keep_percent
        )
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())