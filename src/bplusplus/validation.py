import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import requests
import logging
import sys
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate(species_list, validation_dir, hierarchical_weights, img_size=640, batch_size=32, backbone: str = "resnet50"):
    """
    Validate the hierarchical classifier on a directory of images organized by species.
    
    Args:
        species_list (list): List of species names used for training
        validation_dir (str): Path to directory containing subdirectories for each species
        hierarchical_weights (str): Path to the hierarchical classifier model file
        img_size (int): Image size for validation (should match training, default: 640)
        batch_size (int): Batch size for processing images (default: 32)
        backbone (str): ResNet backbone to use ('resnet18', 'resnet50', 'resnet101'). Default: 'resnet50'
    
    Returns:
        Dictionary containing validation results
    """
    validator = HierarchicalValidator(hierarchical_weights, species_list, img_size, batch_size, backbone)
    results = validator.run(validation_dir)
    print("\nValidation complete with metrics calculated at all taxonomic levels")
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
    def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None, backbone: str = "resnet50"):
        """
        Args:
            num_classes_per_level (list): Number of classes for each taxonomic level
            level_to_idx (dict): Mapping from level to label indices
            parent_child_relationship (dict): Parent-child relationships in taxonomy
            backbone (str): ResNet backbone to use ('resnet18', 'resnet50', 'resnet101')
        """
        super(HierarchicalInsectClassifier, self).__init__()
        
        self.backbone = self._build_backbone(backbone)
        self.backbone_name = backbone
        backbone_output_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
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
        
        # Store the taxonomy mappings
        self.level_to_idx = level_to_idx
        self.parent_child_relationship = parent_child_relationship
        
        self.register_buffer('class_means', torch.zeros(sum(num_classes_per_level)))
        self.register_buffer('class_stds', torch.ones(sum(num_classes_per_level)))
        self.class_counts = [0] * sum(num_classes_per_level)
        self.output_history = defaultdict(list)

    @staticmethod
    def _build_backbone(backbone: str):
        name = backbone.lower()
        if name == "resnet18":
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if name == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if name == "resnet101":
            return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from 'resnet18', 'resnet50', 'resnet101'.")
        
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
    
    species_list_for_gbif = [s for s in species_list if s.lower() != 'unknown']
    has_unknown = len(species_list_for_gbif) != len(species_list)
    
    logger.info(f"Building taxonomy from GBIF for {len(species_list_for_gbif)} species")
    
    print("\nTaxonomy Results:")
    print("-" * 80)
    print(f"{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)
    
    for species_name in species_list_for_gbif:
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
                    sys.exit(1)
            else:
                error_msg = f"Species '{species_name}' not found in GBIF, could be spelling error, check GBIF"
                logger.error(error_msg)
                print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                print(f"Error: {error_msg}")
                sys.exit(1)
                
        except Exception as e:
            error_msg = f"Error retrieving data for species '{species_name}': {str(e)}"
            logger.error(error_msg)
            print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
            print(f"Error: {error_msg}")
            sys.exit(1)

    if has_unknown:
        unknown_family = "Unknown"
        unknown_genus = "Unknown"
        unknown_species = "unknown"
        
        if unknown_family not in taxonomy[1]:
            taxonomy[1].append(unknown_family)
        
        taxonomy[2][unknown_genus] = unknown_family
        taxonomy[3][unknown_species] = unknown_genus
        
        print(f"{unknown_species:<30} {unknown_family:<20} {unknown_genus:<20} {'OK'}")
    
    taxonomy[1] = sorted(list(set(taxonomy[1])))
    print("-" * 80)
    
    num_families = len(taxonomy[1])
    num_genera = len(taxonomy[2])
    num_species = len(taxonomy[3])
    
    print("\nFamily indices:")
    for i, family in enumerate(taxonomy[1]):
        print(f"  {i}: {family}")
    
    print("\nGenus indices:")
    for i, genus in enumerate(sorted(taxonomy[2].keys())):
        print(f"  {i}: {genus}")
    
    print("\nSpecies indices:")
    for i, species in enumerate(species_list):
        print(f"  {i}: {species}")
    
    logger.info(f"Taxonomy built: {num_families} families, {num_genera} genera, {num_species} species")
    return taxonomy, species_to_genus, genus_to_family

def create_mappings(taxonomy, species_list=None):
    """Create index mappings from taxonomy"""
    level_to_idx = {}
    idx_to_level = {}

    for level, labels in taxonomy.items():
        if isinstance(labels, list):
            # Level 1: Family (already sorted)
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels)}
            idx_to_level[level] = {idx: label for idx, label in enumerate(labels)}
        else:  # dict for levels 2 and 3
            if level == 3 and species_list is not None:
                # For species, the order is determined by species_list
                level_to_idx[level] = {label: idx for idx, label in enumerate(species_list)}
                idx_to_level[level] = {idx: label for idx, label in enumerate(species_list)}
            else:
                # For genus, sort alphabetically
                sorted_keys = sorted(labels.keys())
                level_to_idx[level] = {label: idx for idx, label in enumerate(sorted_keys)}
                idx_to_level[level] = {idx: label for idx, label in enumerate(sorted_keys)}
    
    return level_to_idx, idx_to_level

class HierarchicalValidator:
    def __init__(self, hierarchical_model_path, species_names, img_size=640, batch_size=32, backbone: str = "resnet50"):
        cuda_cleanup()
        
        self.device = setup_gpu()
        logger.info(f"Using device: {self.device}")
        print(f"Using device: {self.device}")
        
        self.species_names = species_names
        self.img_size = img_size
        self.batch_size = batch_size
        self.backbone = backbone
        
        logger.info(f"Loading model from {hierarchical_model_path}")
        try:
            checkpoint = torch.load(hierarchical_model_path, map_location='cpu')
            logger.info("Model loaded to CPU successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Extract taxonomy and model state
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_backbone = checkpoint.get("backbone", backbone)
            self.backbone = checkpoint_backbone
            
            if "taxonomy" in checkpoint:
                print("Using taxonomy from saved model")
                taxonomy = checkpoint["taxonomy"]
                if "species_list" in checkpoint:
                    saved_species = checkpoint["species_list"]
                    print(f"Saved model was trained on: {', '.join(saved_species)}")
                
                # Construct mappings from saved taxonomy
                if "level_to_idx" in checkpoint:
                    level_to_idx = checkpoint["level_to_idx"]
                    # Create idx_to_level from level_to_idx
                    idx_to_level = {}
                    for level, label_dict in level_to_idx.items():
                        idx_to_level[level] = {idx: label for label, idx in label_dict.items()}
                    species_to_genus = {species: genus for species, genus in taxonomy[3].items()}
                    genus_to_family = {genus: family for genus, family in taxonomy[2].items()}
                else:
                    # Fallback: create mappings from taxonomy
                    print("Warning: No level_to_idx in checkpoint, creating from taxonomy")
                    level_to_idx, idx_to_level = create_mappings(taxonomy, species_names)
                    species_to_genus = {species: genus for species, genus in taxonomy[3].items()}
                    genus_to_family = {genus: family for genus, family in taxonomy[2].items()}
            else:
                # Fetch from GBIF
                print("No taxonomy in checkpoint, fetching from GBIF")
                taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
                level_to_idx, idx_to_level = create_mappings(taxonomy, species_names)
        else:
            # Old format without model_state_dict wrapper
            state_dict = checkpoint
            taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
            level_to_idx, idx_to_level = create_mappings(taxonomy, species_names)
            # keep user-provided backbone when old checkpoint is used
        
        self.level_to_idx = level_to_idx
        self.idx_to_level = idx_to_level
        self.taxonomy = taxonomy
        self.species_to_genus = species_to_genus
        self.genus_to_family = genus_to_family
        
        # Get number of classes per level
        if hasattr(taxonomy, "items"):
            num_classes_per_level = [len(classes) if isinstance(classes, list) else len(classes.keys()) 
                                    for level, classes in taxonomy.items()]
        
        print(f"Using model with class counts: {num_classes_per_level}")
        
        # Initialize and load model
        self.model = HierarchicalInsectClassifier(
            num_classes_per_level,
            level_to_idx=level_to_idx,
            parent_child_relationship=checkpoint.get("parent_child_relationship", None) if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else None,
            backbone=self.backbone
        )
        
        try:
            self.model.load_state_dict(state_dict)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Attempting to load with strict=False...")
            self.model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded with strict=False")
        
        try:
            self.model.to(self.device)
            print(f"Model successfully transferred to {self.device}")
        except RuntimeError as e:
            logger.error(f"Error transferring model to {self.device}: {e}")
            print(f"Error transferring model to {self.device}, falling back to CPU")
            self.device = torch.device("cpu")
            
        self.model.eval()

        # Transform for validation (same as training validation transform)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model successfully loaded")
        print(f"Using image size: {self.img_size}x{self.img_size}")
        print(f"Using species: {', '.join(species_names)}")

    def load_images_from_directory(self, validation_dir):
        """Load all images from subdirectories organized by species"""
        images = []
        labels = []
        image_paths = []
        
        print(f"\nLoading images from {validation_dir}")
        
        for species_name in self.species_names:
            species_dir = os.path.join(validation_dir, species_name)
            
            if not os.path.exists(species_dir):
                logger.warning(f"Directory not found for species: {species_name}")
                continue
            
            species_idx = self.species_names.index(species_name)
            
            # Get taxonomy for this species
            if species_name in self.species_to_genus:
                genus_name = self.species_to_genus[species_name]
                family_name = self.genus_to_family[genus_name]
                
                genus_idx = self.level_to_idx[2][genus_name]
                family_idx = self.level_to_idx[1][family_name]
            else:
                logger.warning(f"Taxonomy not found for species: {species_name}")
                continue
            
            # Load all images for this species
            image_files = [f for f in os.listdir(species_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(species_dir, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    labels.append([family_idx, genus_idx, species_idx])
                    image_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
                    continue
            
            print(f"  {species_name}: {len(image_files)} images")
        
        print(f"\nTotal images loaded: {len(images)}")
        return images, labels, image_paths

    def predict_images(self, images):
        """Run prediction on a list of images"""
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(images), self.batch_size), desc="Processing batches"):
            batch_images = images[i:i + self.batch_size]
            
            # Transform and batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.transform(img)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # Get predictions for each level
            for j in range(len(batch_images)):
                pred = []
                for level_output in outputs:
                    level_pred = level_output[j].argmax().item()
                    pred.append(level_pred)
                predictions.append(pred)
        
        return predictions

    def run(self, validation_dir):
        """Run validation on the dataset"""
        # Load images
        images, labels, image_paths = self.load_images_from_directory(validation_dir)
        
        if len(images) == 0:
            logger.error("No images found in validation directory")
            return None
        
        # Get predictions
        print("\nRunning predictions...")
        predictions = self.predict_images(images)
        
        # Calculate metrics
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        self.calculate_metrics(predictions, labels)
        
        return {
            'predictions': predictions,
            'labels': labels,
            'image_paths': image_paths
        }

    def calculate_metrics(self, predictions, labels):
        """Calculate metrics at all taxonomic levels"""
        # Convert to numpy arrays for easier manipulation
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        level_names = ['Family', 'Genus', 'Species']
        
        for level in range(3):
            print(f"\n{'='*80}")
            print(f"{level_names[level]}-level Metrics")
            print(f"{'='*80}")
            
            # Get predictions and labels for this level
            level_preds = predictions[:, level]
            level_labels = labels[:, level]
            
            # Get label names for this level
            if level == 0:  # Family
                label_names = [self.idx_to_level[1][i] for i in sorted(self.idx_to_level[1].keys())]
            elif level == 1:  # Genus
                label_names = [self.idx_to_level[2][i] for i in sorted(self.idx_to_level[2].keys())]
            else:  # Species
                label_names = self.species_names
            
            self.print_classification_report(level_preds, level_labels, label_names)

    def print_classification_report(self, predictions, labels, label_names):
        """Print detailed classification metrics"""
        # Calculate per-class metrics
        unique_labels = sorted(set(labels))
        
        table_data = []
        total_correct = 0
        total_samples = 0
        
        for label_idx in unique_labels:
            # Get indices for this class
            class_mask = labels == label_idx
            class_preds = predictions[class_mask]
            class_labels = labels[class_mask]
            
            # Calculate metrics
            support = len(class_labels)
            correct = (class_preds == class_labels).sum()
            
            # Calculate precision, recall, f1
            tp = correct
            fp = ((predictions == label_idx) & (labels != label_idx)).sum()
            fn = support - correct
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            label_name = label_names[label_idx] if label_idx < len(label_names) else f"Label {label_idx}"
            table_data.append([label_name, f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}", support])
            
            total_correct += correct
            total_samples += support
        
        # Calculate overall metrics
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Calculate macro and weighted averages
        macro_precision = np.mean([float(row[1]) for row in table_data])
        macro_recall = np.mean([float(row[2]) for row in table_data])
        macro_f1 = np.mean([float(row[3]) for row in table_data])
        
        weighted_precision = np.sum([float(row[1]) * row[4] for row in table_data]) / total_samples
        weighted_recall = np.sum([float(row[2]) * row[4] for row in table_data]) / total_samples
        weighted_f1 = np.sum([float(row[3]) * row[4] for row in table_data]) / total_samples
        
        # Add summary rows
        table_data.append([])
        table_data.append(["Macro avg", f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}", total_samples])
        table_data.append(["Weighted avg", f"{weighted_precision:.4f}", f"{weighted_recall:.4f}", f"{weighted_f1:.4f}", total_samples])
        table_data.append([])
        table_data.append(["Overall Accuracy", "", "", f"{overall_accuracy:.4f}", total_samples])
        
        headers = ["Label", "Precision", "Recall", "F1 Score", "Support"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate hierarchical insect classifier')
    parser.add_argument('--validation_dir', type=str, required=True,
                       help='Path to validation directory with subdirectories for each species')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to hierarchical model weights')
    parser.add_argument('--species', type=str, nargs='+', required=True,
                       help='List of species names (must match training order)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size for validation (should match training, default: 640)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101'],
                       help='ResNet backbone to use (default: resnet50)')
    
    args = parser.parse_args()
    
    validate(
        species_list=args.species,
        validation_dir=args.validation_dir,
        hierarchical_weights=args.weights,
        img_size=args.img_size,
        batch_size=args.batch_size,
        backbone=args.backbone
    )

