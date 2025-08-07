import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import requests
import time
import logging
from tqdm import tqdm  
import sys

def train(batch_size=4, epochs=30, patience=3, img_size=640, data_dir='input', output_dir='./output', species_list=None, num_workers=4):
    """
    Main function to run the entire training pipeline.
    Sets up datasets, model, training process and handles errors.
    
    Args:
        batch_size (int): Number of samples per batch. Default: 4
        epochs (int): Maximum number of training epochs. Default: 30
        patience (int): Early stopping patience (epochs without improvement). Default: 3
        img_size (int): Target image size for training. Default: 640
        data_dir (str): Directory containing train/valid subdirectories. Default: 'input'
        output_dir (str): Directory to save trained model and logs. Default: './output'
        species_list (list): List of species names for training. Required.
        num_workers (int): Number of DataLoader worker processes. 
                          Set to 0 to disable multiprocessing (most stable). Default: 4
    """
    global logger, device

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Hyperparameters - Batch size: {batch_size}, Epochs: {epochs}, Patience: {patience}, Image size: {img_size}, Data directory: {data_dir}, Output directory: {output_dir}, Num workers: {num_workers}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    np.random.seed(42)
    
    learning_rate = 1.0e-4
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    
    os.makedirs(output_dir, exist_ok=True)
    
    missing_species = []
    for species in species_list:
        species_dir = os.path.join(train_dir, species)
        if not os.path.isdir(species_dir):
            missing_species.append(species)
    
    if missing_species:
        raise ValueError(f"The following species directories were not found: {missing_species}")
    
    logger.info(f"Using {len(species_list)} species in the specified order")
    
    taxonomy = get_taxonomy(species_list)
    
    level_to_idx, parent_child_relationship = create_mappings(taxonomy, species_list)
    
    num_classes_per_level = [len(taxonomy[level]) if isinstance(taxonomy[level], list) 
                            else len(taxonomy[level].keys()) for level in sorted(taxonomy.keys())]
    
    train_dataset = InsectDataset(
        root_dir=train_dir,
        transform=get_transforms(is_training=True, img_size=img_size),
        taxonomy=taxonomy,
        level_to_idx=level_to_idx
    )
    
    val_dataset = InsectDataset(
        root_dir=val_dir,
        transform=get_transforms(is_training=False, img_size=img_size),
        taxonomy=taxonomy,
        level_to_idx=level_to_idx
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    try:
        logger.info("Initializing model...")
        model = HierarchicalInsectClassifier(
            num_classes_per_level=num_classes_per_level,
            level_to_idx=level_to_idx,
            parent_child_relationship=parent_child_relationship
        )
        logger.info(f"Model structure initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        logger.info("Setting up loss function and optimizer...")
        criterion = HierarchicalLoss(
            alpha=0.5,
            level_to_idx=level_to_idx,
            parent_child_relationship=parent_child_relationship
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        logger.info("Testing model with a dummy input...")
        model.to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        with torch.no_grad():
            try:
                test_outputs = model(dummy_input)
                logger.info(f"Forward pass test successful, output shapes: {[out.shape for out in test_outputs]}")
            except Exception as e:
                logger.error(f"Forward pass test failed: {str(e)}")
                raise
        
        logger.info("Starting training process...")
        best_model_path = os.path.join(output_dir, 'best_multitask.pt')
        
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            level_to_idx=level_to_idx,
            parent_child_relationship=parent_child_relationship,
            taxonomy=taxonomy,
            species_list=species_list,
            num_epochs=epochs,
            patience=patience,
            best_model_path=best_model_path
        )
        
        logger.info("Model saved successfully with taxonomy information!")
        print("Model saved successfully with taxonomy information!")
        
        return trained_model, taxonomy, level_to_idx, parent_child_relationship
    
    except Exception as e:
        logger.error(f"Critical error during model setup or training: {str(e)}")
        logger.exception("Stack trace:")
        raise

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
    return taxonomy

def get_species_from_directory(train_dir):
    """
    Extracts a list of species names from subdirectories in the training directory.
    Returns a sorted list of species names found.
    """
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    
    species_list = []
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path):
            species_list.append(item)
    
    species_list.sort()
    
    if not species_list:
        raise ValueError(f"No species subdirectories found in {train_dir}")
    
    logger.info(f"Found {len(species_list)} species in {train_dir}")
    return species_list

def create_mappings(taxonomy, species_list=None):
    """
    Creates mapping dictionaries from taxonomy data.
    Returns level-to-index mapping and parent-child relationships between taxonomic levels.
    """
    level_to_idx = {}
    parent_child_relationship = {}

    for level, labels in taxonomy.items():
        if isinstance(labels, list):
            # Level 1: Family (already sorted)
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels)}
        else:  # dict for levels 2 and 3
            if level == 3 and species_list is not None:
                # For species, the order is determined by species_list
                level_to_idx[level] = {label: idx for idx, label in enumerate(species_list)}
            else:
                # For genus (and as a fallback for species), sort alphabetically
                sorted_keys = sorted(labels.keys())
                level_to_idx[level] = {label: idx for idx, label in enumerate(sorted_keys)}
            
            for child, parent in labels.items():
                if (level, parent) not in parent_child_relationship:
                    parent_child_relationship[(level, parent)] = []
                parent_child_relationship[(level, parent)].append(child)
    
    return level_to_idx, parent_child_relationship

class InsectDataset(Dataset):
    """
    PyTorch dataset for loading and processing insect images.
    Organizes data according to taxonomic hierarchy and validates images.
    """
    def __init__(self, root_dir, transform=None, taxonomy=None, level_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.taxonomy = taxonomy
        self.level_to_idx = level_to_idx
        self.samples = []
        
        species_to_genus = {species: genus for species, genus in taxonomy[3].items()}
        genus_to_family = {genus: family for genus, family in taxonomy[2].items()}
        
        for species_name in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_name)
            if os.path.isdir(species_path):
                if species_name in species_to_genus:
                    genus_name = species_to_genus[species_name]
                    family_name = genus_to_family[genus_name]
                    
                    for img_file in os.listdir(species_path):
                        if img_file.endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(species_path, img_file)
                            # Validate the image can be opened
                            try:
                                with Image.open(img_path) as img:
                                    img.convert('RGB')
                                # Only add valid images to samples
                                self.samples.append({
                                    'image_path': img_path,
                                    'labels': [family_name, genus_name, species_name]
                                })

                            except Exception as e:
                                logger.warning(f"Skipping invalid image: {img_path} - Error: {str(e)}")
                else:
                    logger.warning(f"Warning: Species '{species_name}' not found in taxonomy. Skipping.")
        
        # Log statistics about valid/invalid images
        logger.info(f"Dataset loaded with {len(self.samples)} valid images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_indices = [self.level_to_idx[level+1][label] for level, label in enumerate(sample['labels'])]
        
        return image, torch.tensor(label_indices)

class HierarchicalInsectClassifier(nn.Module):
    """
    Deep learning model for hierarchical insect classification.
    Uses ResNet50 backbone with multiple classification branches for different taxonomic levels.
    Includes anomaly detection capabilities.
    """
    def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None):
        super(HierarchicalInsectClassifier, self).__init__()
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
        
    def forward(self, x):
        R0 = self.backbone(x)
        
        outputs = []
        for branch in self.branches:
            outputs.append(branch(R0))
            
        return outputs
    
    def predict_with_hierarchy(self, x):
        outputs = self.forward(x)
        predictions = []
        confidences = []
        is_unsure = []
        
        level1_output = outputs[0]
        level1_probs = torch.softmax(level1_output, dim=1)
        level1_pred = torch.argmax(level1_output, dim=1)
        level1_conf = torch.gather(level1_probs, 1, level1_pred.unsqueeze(1)).squeeze(1)
        
        start_idx = 0
        level1_unsure = self.detect_anomalies(level1_output, level1_pred, start_idx)
        
        predictions.append(level1_pred)
        confidences.append(level1_conf)
        is_unsure.append(level1_unsure)
        
        # Check if taxonomy mappings are available
        if self.level_to_idx is None or self.parent_child_relationship is None:
            # Return basic predictions if taxonomy isn't available
            for level in range(1, self.num_levels):
                level_output = outputs[level]
                level_probs = torch.softmax(level_output, dim=1)
                level_pred = torch.argmax(level_output, dim=1)
                level_conf = torch.gather(level_probs, 1, level_pred.unsqueeze(1)).squeeze(1)
                start_idx += outputs[level-1].shape[1]
                level_unsure = self.detect_anomalies(level_output, level_pred, start_idx)
                
                predictions.append(level_pred)
                confidences.append(level_conf)
                is_unsure.append(level_unsure)
                
            return predictions, confidences, is_unsure
        
        # If taxonomy is available, use hierarchical constraints
        for level in range(1, self.num_levels):
            level_output = outputs[level]
            level_probs = torch.softmax(level_output, dim=1)
            level_pred = torch.argmax(level_output, dim=1)
            level_conf = torch.gather(level_probs, 1, level_pred.unsqueeze(1)).squeeze(1)
            
            start_idx += outputs[level-1].shape[1]
            level_unsure = self.detect_anomalies(level_output, level_pred, start_idx)
            
            level_unsure_hierarchy = torch.zeros_like(level_pred, dtype=torch.bool)
            for i in range(level_pred.shape[0]):
                prev_level_pred_idx = predictions[level-1][i].item()
                curr_level_pred_idx = level_pred[i].item()
                
                prev_level_label = list(self.level_to_idx[level])[prev_level_pred_idx]
                curr_level_label = list(self.level_to_idx[level+1])[curr_level_pred_idx]
                
                if (level+1, prev_level_label) in self.parent_child_relationship:
                    valid_children = self.parent_child_relationship[(level+1, prev_level_label)]
                    if curr_level_label not in valid_children:
                        level_unsure_hierarchy[i] = True
                else:
                    level_unsure_hierarchy[i] = True
            
            level_unsure = torch.logical_or(level_unsure, level_unsure_hierarchy)
            
            predictions.append(level_pred)
            confidences.append(level_conf)
            is_unsure.append(level_unsure)
            
        return predictions, confidences, is_unsure
    
    def detect_anomalies(self, outputs, predictions, start_idx):
        unsure = torch.zeros_like(predictions, dtype=torch.bool)
        
        if self.training:
            for i in range(outputs.shape[0]):
                pred_class = predictions[i].item()
                class_idx = start_idx + pred_class
                self.output_history[class_idx].append(outputs[i, pred_class].item())
        else:
            for i in range(outputs.shape[0]):
                pred_class = predictions[i].item()
                class_idx = start_idx + pred_class
                
                if len(self.output_history[class_idx]) > 0:
                    mean = np.mean(self.output_history[class_idx])
                    std = np.std(self.output_history[class_idx])
                    threshold = mean - 2 * std
                    
                    if outputs[i, pred_class].item() < threshold:
                        unsure[i] = True
        
        return unsure
    
    def update_anomaly_stats(self):
        for class_idx, outputs in self.output_history.items():
            if len(outputs) > 0:
                self.class_means[class_idx] = torch.tensor(np.mean(outputs))
                self.class_stds[class_idx] = torch.tensor(np.std(outputs))

class HierarchicalLoss(nn.Module):
    """
    Custom loss function for hierarchical classification.
    Combines cross-entropy loss with dependency loss to enforce taxonomic constraints.
    """
    def __init__(self, alpha=0.5, level_to_idx=None, parent_child_relationship=None):
        super(HierarchicalLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.level_to_idx = level_to_idx
        self.parent_child_relationship = parent_child_relationship
        
    def forward(self, outputs, targets, predictions):
        ce_losses = []
        for level, output in enumerate(outputs):
            ce_losses.append(self.ce_loss(output, targets[:, level]))
        
        total_ce_loss = sum(ce_losses)
        
        dependency_losses = []
        
        # Skip dependency loss calculation if taxonomy isn't available
        if self.level_to_idx is None or self.parent_child_relationship is None:
            return total_ce_loss, total_ce_loss, torch.zeros(1, device=outputs[0].device)
        
        for level in range(1, len(outputs)):
            dependency_loss = torch.zeros(1, device=outputs[0].device)
            for i in range(targets.shape[0]):
                prev_level_pred_idx = predictions[level-1][i].item()
                curr_level_pred_idx = predictions[level][i].item()
                
                prev_level_label = list(self.level_to_idx[level])[prev_level_pred_idx]
                curr_level_label = list(self.level_to_idx[level+1])[curr_level_pred_idx]
                
                is_valid = False
                if (level+1, prev_level_label) in self.parent_child_relationship:
                    valid_children = self.parent_child_relationship[(level+1, prev_level_label)]
                    if curr_level_label in valid_children:
                        is_valid = True
                
                D_l = 0 if is_valid else 1
                dependency_loss += torch.exp(torch.tensor(D_l, device=outputs[0].device)) - 1
            
            dependency_loss /= targets.shape[0]
            dependency_losses.append(dependency_loss)
        
        total_dependency_loss = sum(dependency_losses) if dependency_losses else torch.zeros(1, device=outputs[0].device)
        
        total_loss = self.alpha * total_ce_loss + (1 - self.alpha) * total_dependency_loss
        
        return total_loss, total_ce_loss, total_dependency_loss

def get_transforms(is_training=True, img_size=640):
    """
    Creates image transformation pipelines.
    Returns different transformations for training and validation data.
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Fixed size for all validation images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_model(model, train_loader, val_loader, criterion, optimizer, level_to_idx, parent_child_relationship, taxonomy, species_list, num_epochs=10, patience=5, best_model_path='best_multitask.pt'):
    """
    Trains the hierarchical classifier model.
    Implements early stopping, validation, and model checkpointing.
    """
    logger.info("Starting training")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dep_loss = 0.0
        correct_predictions = [0] * model.num_levels
        total_predictions = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                predictions = []
                for output in outputs:
                    pred = torch.argmax(output, dim=1)
                    predictions.append(pred)
                
                loss, ce_loss, dep_loss = criterion(outputs, labels, predictions)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_ce_loss += ce_loss.item()
                running_dep_loss += dep_loss.item() if dep_loss.numel() > 0 else 0
                
                for level in range(model.num_levels):
                    correct_predictions[level] += (predictions[level] == labels[:, level]).sum().item()
                total_predictions += labels.size(0)
                
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue  # Skip this batch and continue with the next one
        
        epoch_loss = running_loss / len(train_loader)
        epoch_ce_loss = running_ce_loss / len(train_loader)
        epoch_dep_loss = running_dep_loss / len(train_loader)
        epoch_accuracies = [correct / total_predictions for correct in correct_predictions]
        
        model.update_anomaly_stats()
        
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = [0] * model.num_levels
        val_total_predictions = 0
        val_unsure_count = [0] * model.num_levels
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                try:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    predictions, confidences, is_unsure = model.predict_with_hierarchy(images)
                    outputs = model(images)
                    
                    loss, _, _ = criterion(outputs, labels, predictions)
                    val_running_loss += loss.item()
                    
                    for level in range(model.num_levels):
                        correct_mask = (predictions[level] == labels[:, level]) & ~is_unsure[level]
                        val_correct_predictions[level] += correct_mask.sum().item()
                        val_unsure_count[level] += is_unsure[level].sum().item()
                    val_total_predictions += labels.size(0)
                    
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracies = [correct / val_total_predictions for correct in val_correct_predictions]
        val_unsure_rates = [unsure / val_total_predictions for unsure in val_unsure_count]
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} (CE: {epoch_ce_loss:.4f}, Dep: {epoch_dep_loss:.4f})")
        print(f"Valid Loss: {val_epoch_loss:.4f}")
        
        for level in range(model.num_levels):
            print(f"Level {level+1} - Train Acc: {epoch_accuracies[level]:.4f}, "
                  f"Valid Acc: {val_epoch_accuracies[level]:.4f}, "
                  f"Unsure: {val_unsure_rates[level]:.4f}")
        print('-' * 60)
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'taxonomy': taxonomy,
                'level_to_idx': level_to_idx,
                'parent_child_relationship': parent_child_relationship,
                'species_list': species_list
            }, best_model_path)
            logger.info(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs. Best val loss: {best_val_loss:.4f}")
            
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training completed successfully")
    return model

if __name__ == '__main__':
    species_list = [
        "Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
        "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
        "Eristalis tenax", "unknown"
    ]
    train(species_list=species_list, epochs=2)

