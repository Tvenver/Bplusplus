# pip install ultralytics torchvision pillow numpy scikit-learn tabulate tqdm
#python3 train-resnet.py --data_dir '' --output_dir '' --arch resnet50 --img_size 956 --num_epochs 50 --batch_size 4

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
import logging
from torchvision.models import ResNet152_Weights, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from datetime import datetime

def train_resnet(species_list, model_type='resnet152', batch_size=4, num_epochs=50, patience=5, output_dir=None, data_dir=None, img_size=1024):
    """Main entry point for training the model."""
    # Setup output directory
    output_dir = setup_directories(output_dir)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    
    # Get transforms directly using the img_size parameter
    train_transforms, val_transforms = get_transforms(img_size)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Hyperparameters - Batch size: {batch_size}, Epochs: {num_epochs}, Patience: {patience}, Data directory: {data_dir}, Output directory: {output_dir}")
    
    # Use InsectDataset instead of OrderedImageFolder
    train_dataset = InsectDataset(train_dir, species_list=species_list, transform=train_transforms)
    val_dataset = InsectDataset(val_dir, species_list=species_list, transform=val_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = train_dataset.classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Initialize the model based on the model_type parameter
    logger.info(f"Initializing {model_type} model...")
    if model_type == 'resnet152':
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)
    logger.info(f"Model structure initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Print class distributions as in hierarchical-train.py
    print("\nTraining set class distribution:")
    train_counts = Counter(train_dataset.targets)
    for idx, count in train_counts.items():
        print(f"  {class_names[idx]}: {count}")
        
    print("\nValidation set class distribution:")
    val_counts = Counter(val_dataset.targets)
    for idx, count in val_counts.items():
        print(f"  {class_names[idx]}: {count}")
    
    logger.info("Starting training process...")
    model = train_model(model, criterion, optimizer, scheduler, num_epochs, device, train_loader, val_loader, 
                       dataset_sizes, class_names, output_dir, patience=patience)
    
    # Save the best model
    model_filename = f'best_{model_type}.pt'
    if output_dir:
        model_path = os.path.join(output_dir, model_filename)
    else:
        model_path = model_filename
        
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved best model to {model_path}")
    print(f"Model saved successfully!")
    
    return model

class InsectDataset(Dataset):
    """Dataset for loading and processing insect images with validation."""
    def __init__(self, root_dir, species_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = species_list
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(species_list)}
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loading dataset from {root_dir} with {len(species_list)} species")
        
        # Validate all images similar to SafeImageFolder
        invalid_count = 0
        for species_name in species_list:
            species_path = os.path.join(root_dir, species_name)
            if os.path.isdir(species_path):
                for img_file in os.listdir(species_path):
                    if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(species_path, img_file)
                        try:
                            # Validate the image can be opened
                            with Image.open(img_path) as img:
                                img.convert('RGB')
                            # Only add valid images to samples
                            class_idx = self.class_to_idx[species_name]
                            self.samples.append((img_path, class_idx))
                            self.targets.append(class_idx)
                        except Exception as e:
                            invalid_count += 1
                            logger.warning(f"Skipping invalid image {img_path}: {str(e)}")
            else:
                logger.warning(f"Species directory not found: {species_path}")
                
        logger.info(f"Dataset loaded with {len(self.samples)} valid images ({invalid_count} invalid images skipped)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get an image and its target in a simple approach."""
        img_path, target = self.samples[idx]
        # Simple direct loading
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

def setup_directories(output_dir):
    """Create output directory if needed."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_transforms(img_size):
    """Return training and validation transforms based on the desired image size."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def print_class_assignments(dataset, class_names, max_images_per_class=10):
    """
    Print information about which images belong to each class.
    
    Args:
        dataset: The dataset to analyze
        class_names: List of class names
        max_images_per_class: Maximum number of images to print per class
    """
    print("\n==== Class Assignment Information ====")
    
    # Count images per class
    class_counts = {}
    for _, target in dataset.samples:
        class_name = class_names[target]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    # Print summary of class distribution
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Print sample file paths for each class
    print("\nSample images for each class:")
    for class_name in class_names:
        print(f"\nClass: {class_name}")
        
        # Collect paths for this class
        paths = []
        for path, target in dataset.samples:
            if class_names[target] == class_name:
                paths.append(path)
                
                # Limit the number of paths to print
                if len(paths) >= max_images_per_class:
                    break
        
        # Print paths
        for i, path in enumerate(paths, 1):
            print(f"  {i}. {path}")
        
        # Check if any images were found
        if not paths:
            print("  No images found for this class!")

def train_model(model, criterion, optimizer, scheduler, num_epochs, device, train_loader, val_loader, dataset_sizes, class_names, output_dir, patience=10):
    """Train the model and return the best model based on validation accuracy."""
    since = time.time()  # Record start time
    best_model_wts = copy.deepcopy(model.state_dict())  # Store initial model state
    best_acc = 0.0  # Initialize best accuracy
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    # Early stopping variables
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        # Display current epoch
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                loader = train_loader
            else:
                model.eval()  # Set model to evaluation mode
                loader = val_loader
                
            running_loss = 0.0  # Initialize loss accumulator
            running_corrects = 0  # Initialize correct predictions accumulator
            
            # Create progress bar with appropriate description
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [{phase.capitalize()}]")
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)  # Move inputs to device
                labels = labels.to(device)  # Move labels to device
                
                optimizer.zero_grad()  # Reset gradients
                
                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients only in train phase
                    with torch.cuda.amp.autocast():  # Enable mixed precision
                        outputs = model(inputs)  # Forward pass
                        _, preds = torch.max(outputs, 1)  # Get predictions
                        loss = criterion(outputs, labels)  # Compute loss
                        
                    if phase == 'train':
                        scaler.scale(loss).backward()  # Backpropagation with scaling
                        scaler.step(optimizer)  # Update parameters
                        scaler.update()  # Update scaler for mixed precision
                        
                running_loss += loss.item() * inputs.size(0)  # Accumulate loss
                running_corrects += torch.sum(preds == labels.data)  # Accumulate correct predictions
                
                # Update progress bar with current loss
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
            if phase == 'train':
                scheduler.step()  # Update learning rate scheduler
                
            # Calculate epoch metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Print epoch results
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            
            # Save best model if validation accuracy improved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0  # Reset counter since we improved
                print(f"Saved best model with validation accuracy: {best_acc:.4f}")
            elif phase == 'val':
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs. Best accuracy: {best_acc:.4f}")
                
        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement")
            early_stop = True
            break
            
        torch.cuda.empty_cache()  # Clear GPU cache after each epoch
        
    if early_stop:
        print(f"Training stopped early due to no improvement for {patience} epochs")
    
    time_elapsed = time.time() - since  # Calculate elapsed time
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Accuracy: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)  # Load best model weights
    return model

if __name__ == "__main__":
    DEFAULT_SPECIES_LIST = [
        "Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
        "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
        "Eristalis tenax"
    ]
    
    # Call main with default parameters
    train_resnet(
        species_list=DEFAULT_SPECIES_LIST,
        model_type='resnet50',
        batch_size=4,
        num_epochs=2,
        patience=5,
        output_dir='./output',
        data_dir='/mnt/nvme0n1p1/datasets/insect/bjerge-train2',
        img_size=256
    )

