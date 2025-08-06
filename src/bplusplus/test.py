# pip install ultralytics torchvision pillow numpy scikit-learn tabulate tqdm requests

import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet50
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import time
import argparse
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import csv
import requests
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test(species_list, test_set, yolo_weights, hierarchical_weights, output_dir="."):
    """
    Run the two-stage classifier on a test set.
    
    Args:
        species_list (list): List of species names used for training
        test_set (str): Path to the test directory
        yolo_weights (str): Path to the YOLO model file
        hierarchical_weights (str): Path to the hierarchical classifier model file
        output_dir (str): Directory to save output CSV files (default: current directory)
    
    Returns:
        Results from the classifier
    """
    classifier = TestTwoStage(yolo_weights, hierarchical_weights, species_list, output_dir)
    results = classifier.run(test_set)
    print("Testing complete with metrics calculated at all taxonomic levels")
    return results

def cuda_cleanup():
    """Clear CUDA cache and reset device"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
def setup_gpu():
    """Set up GPU with better error handling and reporting"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available on this system")
        return torch.device("cpu")
    
    try:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA device(s)")
        
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_properties.name} with {gpu_properties.total_memory / 1e9:.2f} GB memory")
        
        device = torch.device("cuda:0")
        test_tensor = torch.ones(1, device=device)
        test_result = test_tensor * 2
        del test_tensor, test_result
        
        logger.info("CUDA initialization successful")
        return device
    except Exception as e:
        logger.error(f"CUDA initialization error: {str(e)}")
        logger.warning("Falling back to CPU")
        return torch.device("cpu")

# Add this check for backwards compatibility
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        'torch.LongTensor',
        'torch.cuda.LongTensor',
        'torch.FloatStorage',
        'torch.FloatStorage',
        'torch.cuda.FloatStorage',
    ])

class HierarchicalInsectClassifier(nn.Module):
    def __init__(self, num_classes_per_level):
        """
        Args:
            num_classes_per_level (list): Number of classes for each taxonomic level
        """
        super(HierarchicalInsectClassifier, self).__init__()
        
        self.backbone = resnet50(pretrained=True)
        backbone_output_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        
        self.branches = nn.ModuleList()
        for num_classes in num_classes_per_level:
            branch = nn.Sequential(
                nn.Linear(backbone_output_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            self.branches.append(branch)
        
        self.num_levels = len(num_classes_per_level)
        
        self.register_buffer('class_means', torch.zeros(sum(num_classes_per_level)))
        self.register_buffer('class_stds', torch.ones(sum(num_classes_per_level)))
        self.class_counts = [0] * sum(num_classes_per_level)
        self.output_history = defaultdict(list)
        
    def forward(self, x):
        R0 = self.backbone(x)
        
        outputs = []
        for branch in self.branches:
            outputs.append(branch(R0))
            
        return outputs

def get_taxonomy(species_list):
    """
    Retrieves taxonomic information for a list of species from GBIF API.
    Creates a hierarchical taxonomy dictionary with family, genus, and species relationships.
    """
    taxonomy = {1: [], 2: {}, 3: {}}
    species_to_genus = {}
    genus_to_family = {}
    
    logger.info(f"Building taxonomy from GBIF for {len(species_list)} species")
    
    print("\nTaxonomy Results:")
    print("-" * 80)
    print(f"{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)
    
    for species_name in species_list:
        url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
        try:
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') == 'ACCEPTED' or data.get('status') == 'SYNONYM':
                family = data.get('family')
                genus = data.get('genus')
                
                if family and genus:
                    status = "OK"
                    
                    print(f"{species_name:<30} {family:<20} {genus:<20} {status}")
                    
                    species_to_genus[species_name] = genus
                    genus_to_family[genus] = family
                    
                    if family not in taxonomy[1]:
                        taxonomy[1].append(family)
                    
                    taxonomy[2][genus] = family
                    taxonomy[3][species_name] = genus
                else:
                    error_msg = f"Species '{species_name}' found in GBIF but family and genus not found, could be spelling error in species, check GBIF"
                    logger.error(error_msg)
                    print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                    print(f"Error: {error_msg}")
                    sys.exit(1)  # Stop the script
            else:
                error_msg = f"Species '{species_name}' not found in GBIF, could be spelling error, check GBIF"
                logger.error(error_msg)
                print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                print(f"Error: {error_msg}")
                sys.exit(1)  # Stop the script
                
        except Exception as e:
            error_msg = f"Error retrieving data for species '{species_name}': {str(e)}"
            logger.error(error_msg)
            print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
            print(f"Error: {error_msg}")
            sys.exit(1)  # Stop the script
    
    taxonomy[1] = sorted(list(set(taxonomy[1])))
    print("-" * 80)
    
    num_families = len(taxonomy[1])
    num_genera = len(taxonomy[2])
    num_species = len(taxonomy[3])
    
    print("\nFamily indices:")
    for i, family in enumerate(taxonomy[1]):
        print(f"  {i}: {family}")
    
    print("\nGenus indices:")
    for i, genus in enumerate(taxonomy[2].keys()):
        print(f"  {i}: {genus}")
    
    print("\nSpecies indices:")
    for i, species in enumerate(species_list):
        print(f"  {i}: {species}")
    
    logger.info(f"Taxonomy built: {num_families} families, {num_genera} genera, {num_species} species")
    return taxonomy, species_to_genus, genus_to_family

def create_mappings(taxonomy):
    """Create index mappings from taxonomy"""
    level_to_idx = {}
    idx_to_level = {}

    for level, labels in taxonomy.items():
        if isinstance(labels, list):
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels)}
            idx_to_level[level] = {idx: label for idx, label in enumerate(labels)}
        else:  # Dictionary
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels.keys())}
            idx_to_level[level] = {idx: label for idx, label in enumerate(labels.keys())}
    
    return level_to_idx, idx_to_level

class TestTwoStage:
    def __init__(self, yolo_model_path, hierarchical_model_path, species_names, output_dir="."):
        cuda_cleanup()
        
        self.device = setup_gpu()
        logger.info(f"Using device: {self.device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        logger.info(f"Results will be saved to: {self.output_dir}")
            
        print(f"Using device: {self.device}")

        self.yolo_model = YOLO(yolo_model_path)
        
        self.species_names = species_names
        
        logger.info(f"Loading model from {hierarchical_model_path}")
        try:
            checkpoint = torch.load(hierarchical_model_path, map_location='cpu')
            logger.info("Model loaded to CPU successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            
            if "taxonomy" in checkpoint:
                print("Using taxonomy from saved model")
                taxonomy = checkpoint["taxonomy"]
                if "species_list" in checkpoint:
                    saved_species = checkpoint["species_list"]
                    print(f"Saved model was trained on: {', '.join(saved_species)}")
                
                # Use saved taxonomy mappings if available
                if "species_to_genus" in checkpoint and "genus_to_family" in checkpoint:
                    species_to_genus = checkpoint["species_to_genus"]
                    genus_to_family = checkpoint["genus_to_family"]
                else:
                    # Fallback: fetch from GBIF but this may cause index mismatches
                    print("Warning: No taxonomy mappings in checkpoint, fetching from GBIF")
                    _, species_to_genus, genus_to_family = get_taxonomy(species_names)
            else:
                taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
        else:
            state_dict = checkpoint
            taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
        
        level_to_idx, idx_to_level = create_mappings(taxonomy)
        
        self.level_to_idx = level_to_idx
        self.idx_to_level = idx_to_level
        
        if hasattr(taxonomy, "items"):
            num_classes_per_level = [len(classes) if isinstance(classes, list) else len(classes.keys()) 
                                    for level, classes in taxonomy.items()]
        
        print(f"Using model with class counts: {num_classes_per_level}")
        
        self.classification_model = HierarchicalInsectClassifier(num_classes_per_level)
        
        try:
            self.classification_model.load_state_dict(state_dict)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Attempting to load with strict=False...")
            self.classification_model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded with strict=False")
        
        try:
            self.classification_model.to(self.device)
            print(f"Model successfully transferred to {self.device}")
        except RuntimeError as e:
            logger.error(f"Error transferring model to {self.device}: {e}")
            print(f"Error transferring model to {self.device}, falling back to CPU")
            self.device = torch.device("cpu")
            # No need to move to CPU since it's already there
            
        self.classification_model.eval()

        self.classification_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model successfully loaded")
        print(f"Using species: {', '.join(species_names)}")
        
        self.species_to_genus = species_to_genus
        self.genus_to_family = genus_to_family

    def get_frames(self, test_dir):
        image_dir = os.path.join(test_dir, "images")
        label_dir = os.path.join(test_dir, "labels")
        
        predicted_frames = []
        predicted_family_frames = []
        predicted_genus_frames = []
        true_species_frames = []
        true_family_frames = []
        true_genus_frames = []
        image_names = []

        start_time = time.time()  # Start timing

        for image_name in tqdm(os.listdir(image_dir), desc="Processing Images", unit="image"):
            image_names.append(image_name)
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

            frame = cv2.imread(image_path)
            # Suppress print statements from YOLO model
            with torch.no_grad():
                results = self.yolo_model(frame, conf=0.3, iou=0.5, verbose=False)

            detections = results[0].boxes
            predicted_frame = []
            predicted_family_frame = []
            predicted_genus_frame = []

            if detections:
                for box in detections:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = xyxy[:4]
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    insect_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    insect_crop_rgb = cv2.cvtColor(insect_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(insect_crop_rgb)
                    input_tensor = self.classification_transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.classification_model(input_tensor)
                    
                    # Get all taxonomic level predictions
                    family_output = outputs[0]   # First output is family (level 1)
                    genus_output = outputs[1]    # Second output is genus (level 2)
                    species_output = outputs[2]  # Third output is species (level 3)
                    
                    # Get prediction indices
                    family_idx = family_output.argmax(dim=1).item()
                    genus_idx = genus_output.argmax(dim=1).item()
                    species_idx = species_output.argmax(dim=1).item()
                    
                    img_height, img_width, _ = frame.shape
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height
                    
                    # Create box coordinates in YOLO format
                    box_coords = [x_center_norm, y_center_norm, width_norm, height_norm]
                    
                    # Add predictions for each taxonomic level
                    predicted_frame.append([species_idx] + box_coords)
                    predicted_family_frame.append([family_idx] + box_coords)
                    predicted_genus_frame.append([genus_idx] + box_coords)

            predicted_frames.append(predicted_frame if predicted_frame else [])
            predicted_family_frames.append(predicted_family_frame if predicted_family_frame else [])
            predicted_genus_frames.append(predicted_genus_frame if predicted_genus_frame else [])

            true_species_frame = []
            true_family_frame = []
            true_genus_frame = []
            
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    for line in f:
                        label_line = line.strip().split()
                        species_idx = int(label_line[0])
                        box_coords = list(map(np.float32, label_line[1:]))
                        
                        true_species_frame.append([species_idx] + box_coords)
                        
                        if species_idx < len(self.species_names):
                            species_name = self.species_names[species_idx]
                            
                            if species_name in self.species_to_genus:
                                genus_name = self.species_to_genus[species_name]
                                # Get the index of the genus in the level_to_idx mapping
                                if 2 in self.level_to_idx and genus_name in self.level_to_idx[2]:
                                    genus_idx = self.level_to_idx[2][genus_name]
                                    true_genus_frame.append([genus_idx] + box_coords)
                                
                                if genus_name in self.genus_to_family:
                                    family_name = self.genus_to_family[genus_name]
                                    if 1 in self.level_to_idx and family_name in self.level_to_idx[1]:
                                        family_idx = self.level_to_idx[1][family_name]
                                        true_family_frame.append([family_idx] + box_coords)

            true_species_frames.append(true_species_frame if true_species_frame else [])
            true_family_frames.append(true_family_frame if true_family_frame else [])
            true_genus_frames.append(true_genus_frame if true_genus_frame else [])

        end_time = time.time()  # End timing
        
        # Create a more descriptive filename with timestamp
        output_file = os.path.join(self.output_dir, f"results_hierarchical_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(output_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Image Name", 
                "True Species Detections", 
                "True Genus Detections",
                "True Family Detections",
                "Species Detections", 
                "Genus Detections", 
                "Family Detections"
            ])
            
            for image_name, true_species, true_genus, true_family, species_pred, genus_pred, family_pred in zip(
                image_names, 
                true_species_frames, 
                true_genus_frames,
                true_family_frames,
                predicted_frames, 
                predicted_genus_frames, 
                predicted_family_frames
            ):
                writer.writerow([
                    image_name, 
                    true_species, 
                    true_genus,
                    true_family,
                    species_pred, 
                    genus_pred, 
                    family_pred
                ])
        
        print(f"Results saved to {output_file}")
        return predicted_frames, true_species_frames, end_time - start_time, predicted_genus_frames, predicted_family_frames, true_genus_frames, true_family_frames
    
    def run(self, test_dir):
        results = self.get_frames(test_dir)
        predicted_frames, true_species_frames, total_time = results[0], results[1], results[2]
        predicted_genus_frames = results[3]
        predicted_family_frames = results[4]
        true_genus_frames = results[5]
        true_family_frames = results[6]
        
        num_frames = len(os.listdir(os.path.join(test_dir, 'images')))
        avg_time_per_frame = total_time / num_frames

        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
        
        self.calculate_metrics(
            predicted_frames, true_species_frames, 
            predicted_genus_frames, true_genus_frames,
            predicted_family_frames, true_family_frames
        )

    def calculate_metrics(self, predicted_species_frames, true_species_frames, 
                         predicted_genus_frames, true_genus_frames, 
                         predicted_family_frames, true_family_frames):
        """Calculate metrics at all taxonomic levels"""
        # Get list of species, families and genera using the same order as model training
        species_list = self.species_names
        
        # Use the index mappings from the model to ensure consistency
        if 1 in self.idx_to_level and 2 in self.idx_to_level:
            family_list = [self.idx_to_level[1][i] for i in sorted(self.idx_to_level[1].keys())]
            genus_list = [self.idx_to_level[2][i] for i in sorted(self.idx_to_level[2].keys())]
        else:
            # Fallback to sorted lists (may cause issues)
            print("Warning: Using fallback sorted lists for taxonomy - this may cause index mismatches")
            genus_list = sorted(list(set(self.species_to_genus.values())))
            family_list = sorted(list(set(self.genus_to_family.values())))
        
        # Print the index mappings we're using for evaluation
        print("\nUsing the following index mappings for evaluation:")
        print("\nFamily indices:")
        for i, family in enumerate(family_list):
            print(f"  {i}: {family}")
        
        print("\nGenus indices:")
        for i, genus in enumerate(genus_list):
            print(f"  {i}: {genus}")
        
        print("\nSpecies indices:")
        for i, species in enumerate(species_list):
            print(f"  {i}: {species}")
        
        # Dictionary to track prediction category counts for debugging
        prediction_counts = {
            "true_species_boxes": sum(len(frame) for frame in true_species_frames),
            "true_genus_boxes": sum(len(frame) for frame in true_genus_frames),
            "true_family_boxes": sum(len(frame) for frame in true_family_frames),
            "predicted_species": sum(len(frame) for frame in predicted_species_frames),
            "predicted_genus": sum(len(frame) for frame in predicted_genus_frames),
            "predicted_family": sum(len(frame) for frame in predicted_family_frames)
        }
        
        print(f"Prediction counts: {prediction_counts}")
        
        # Calculate metrics for all three levels
        print("\n=== Species-level Metrics ===")
        self.get_metrics(predicted_species_frames, true_species_frames, species_list)
        
        print("\n=== Genus-level Metrics ===")
        self.get_metrics(predicted_genus_frames, true_genus_frames, genus_list)
        
        print("\n=== Family-level Metrics ===")
        self.get_metrics(predicted_family_frames, true_family_frames, family_list)

    def get_metrics(self, predicted_frames, true_frames, labels):
        """Calculate metrics for object detection predictions"""
        def calculate_iou(box1, box2):
            x1_min, y1_min = box1[1] - box1[3] / 2, box1[2] - box1[4] / 2
            x1_max, y1_max = box1[1] + box1[3] / 2, box1[2] + box1[4] / 2
            x2_min, y2_min = box2[1] - box2[3] / 2, box2[2] - box2[4] / 2
            x2_max, y2_max = box2[1] + box2[3] / 2, box2[2] + box2[4] / 2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)

            iou = inter_area / (box1_area + box2_area - inter_area)
            return iou

        def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
            label_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
            generic_tp = 0
            generic_fp = 0
            
            matched_true_boxes = set()
            
            for pred_box in pred_boxes:
                label_idx = pred_box[0]
                matched = False
                
                best_iou = 0
                best_match_idx = -1
                
                for i, true_box in enumerate(true_boxes):
                    if i in matched_true_boxes:
                        continue
                    
                    iou = calculate_iou(pred_box, true_box)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                if best_match_idx >= 0:
                    matched = True
                    true_box = true_boxes[best_match_idx]
                    matched_true_boxes.add(best_match_idx)
                    generic_tp += 1
                    
                    if pred_box[0] == true_box[0]:
                        label_results[label_idx]['tp'] += 1
                    else:
                        label_results[label_idx]['fp'] += 1
                        true_label_idx = true_box[0]
                        label_results[true_label_idx]['fn'] += 1
                        
                if not matched:
                    label_results[label_idx]['fp'] += 1
                    generic_fp += 1
            
            for i, true_box in enumerate(true_boxes):
                if i not in matched_true_boxes:
                    label_idx = true_box[0]
                    label_results[label_idx]['fn'] += 1
            
            generic_fn = len(true_boxes) - len(matched_true_boxes)
            
            return label_results, generic_tp, generic_fp, generic_fn

        label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
        background_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0}
        generic_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        
        for true_frame in true_frames:
            if not true_frame:  # Empty frame (background only)
                background_metrics['support'] += 1
            else:
                for true_box in true_frame:
                    label_idx = true_box[0]
                    label_metrics[label_idx]['support'] += 1  # Count each detection, not just unique labels

        for pred_frame, true_frame in zip(predicted_frames, true_frames):
            if not pred_frame and not true_frame:
                background_metrics['tp'] += 1
            elif not pred_frame:
                background_metrics['fn'] += 1
            elif not true_frame:
                background_metrics['fp'] += 1
            else:
                frame_results, g_tp, g_fp, g_fn = calculate_precision_recall(pred_frame, true_frame)
                
                for label_idx, metrics in frame_results.items():
                    label_metrics[label_idx]['tp'] += metrics['tp']
                    label_metrics[label_idx]['fp'] += metrics['fp'] 
                    label_metrics[label_idx]['fn'] += metrics['fn']
                
                generic_metrics['tp'] += g_tp
                generic_metrics['fp'] += g_fp
                generic_metrics['fn'] += g_fn

        table_data = []

        for label_idx, metrics in label_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            support = metrics['support']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            label_name = labels[label_idx] if label_idx < len(labels) else f"Label {label_idx}"
            table_data.append([label_name, f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}", f"{support}"])

        tp = background_metrics['tp']
        fp = background_metrics['fp']
        fn = background_metrics['fn']
        support = background_metrics['support']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        table_data.append(["Background", f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}", f"{support}"])

        headers = ["Label", "Precision", "Recall", "F1 Score", "Support"]
        total_tp = sum(metrics['tp'] for metrics in label_metrics.values())
        total_fp = sum(metrics['fp'] for metrics in label_metrics.values())
        total_fn = sum(metrics['fn'] for metrics in label_metrics.values())
        total_support = sum(metrics['support'] for metrics in label_metrics.values())

        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

        table_data.append(["\nTotal (excluding background)", f"{total_precision:.2f}", f"{total_recall:.2f}", f"{total_f1_score:.2f}", f"{total_support}"])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        generic_tp = generic_metrics['tp']
        generic_fp = generic_metrics['fp']
        generic_fn = generic_metrics['fn']

        generic_precision = generic_tp / (generic_tp + generic_fp) if (generic_tp + generic_fp) > 0 else 0
        generic_recall = generic_tp / (generic_tp + generic_fn) if (generic_tp + generic_fn) > 0 else 0
        generic_f1_score = 2 * (generic_precision * generic_recall) / (generic_precision + generic_recall) if (generic_precision + generic_recall) > 0 else 0

        print("\nGeneric Total", f"{generic_precision:.2f}", f"{generic_recall:.2f}", f"{generic_f1_score:.2f}")

if __name__ == "__main__":
    species_names = [
        "Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
        "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
        "Eristalis tenax"
    ]
    
    test_directory = "/mnt/nvme0n1p1/mit/two-stage-detection/bjerge-test"
    yolo_model_path = "/mnt/nvme0n1p1/mit/two-stage-detection/small-generic.pt"
    hierarchical_model_path = "/mnt/nvme0n1p1/mit/two-stage-detection/hierarchical/hierarchical-weights.pth"
    output_directory = "./output"
    
    test(species_names, test_directory, yolo_model_path, hierarchical_model_path, output_directory)