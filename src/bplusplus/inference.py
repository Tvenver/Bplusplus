import cv2
import time
import os
import sys
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from .tracker import InsectTracker
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet50
import requests
import logging
from collections import defaultdict
import uuid

# Add this check for backwards compatibility
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        'torch.LongTensor',
        'torch.cuda.LongTensor',
        'torch.FloatStorage',
        'torch.FloatStorage',
        'torch.cuda.FloatStorage',
    ])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    
    print(f"\n{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)
    
    for species_name in species_list_for_gbif:
        url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
        try:
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') in ['ACCEPTED', 'SYNONYM']:
                family = data.get('family')
                genus = data.get('genus')
                
                if family and genus:
                    print(f"{species_name:<30} {family:<20} {genus:<20} OK")
                    
                    species_to_genus[species_name] = genus
                    genus_to_family[genus] = family
                    
                    if family not in taxonomy[1]:
                        taxonomy[1].append(family)
                    
                    taxonomy[2][genus] = family
                    taxonomy[3][species_name] = genus
                else:
                    print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                    logger.error(f"Species '{species_name}' found but missing family/genus data")
            else:
                print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                logger.error(f"Species '{species_name}' not found in GBIF")
                
        except Exception as e:
            print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
            logger.error(f"Error retrieving data for '{species_name}': {str(e)}")

    if has_unknown:
        unknown_family = "Unknown"
        unknown_genus = "Unknown"
        unknown_species = "unknown"
        
        if unknown_family not in taxonomy[1]:
            taxonomy[1].append(unknown_family)
        
        taxonomy[2][unknown_genus] = unknown_family
        taxonomy[3][unknown_species] = unknown_genus
        species_to_genus[unknown_species] = unknown_genus
        genus_to_family[unknown_genus] = unknown_family
        
        print(f"{unknown_species:<30} {unknown_family:<20} {unknown_genus:<20} {'OK'}")
    
    taxonomy[1] = sorted(list(set(taxonomy[1])))
    print("-" * 80)
    
    # Print indices
    for level, name, items in [(1, "Family", taxonomy[1]), (2, "Genus", taxonomy[2].keys()), (3, "Species", species_list)]:
        print(f"\n{name} indices:")
        for i, item in enumerate(items):
            print(f"  {i}: {item}")
    
    logger.info(f"Taxonomy built: {len(taxonomy[1])} families, {len(taxonomy[2])} genera, {len(taxonomy[3])} species")
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
        else:  # Dictionary for levels 2 and 3
            if level == 3 and species_list is not None:
                # For species, the order is determined by species_list
                sorted_keys = species_list
            else:
                # For genus, sort alphabetically
                sorted_keys = sorted(labels.keys())
            
            level_to_idx[level] = {label: idx for idx, label in enumerate(sorted_keys)}
            idx_to_level[level] = {idx: label for idx, label in enumerate(sorted_keys)}
    
    return level_to_idx, idx_to_level

# ============================================================================
# MODEL CLASSES
# ============================================================================

class HierarchicalInsectClassifier(nn.Module):
    def __init__(self, num_classes_per_level):
        """
        Args:
            num_classes_per_level (list): Number of classes for each taxonomic level [family, genus, species]
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
        
    def forward(self, x):
        R0 = self.backbone(x)
        
        outputs = []
        for branch in self.branches:
            outputs.append(branch(R0))
            
        return outputs

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class FrameVisualizer:
    """Modern, slick visualization system for insect detection and tracking"""
    
    # Modern color palette - vibrant but professional
    COLORS = [
        (68, 189, 50),    # Vibrant Green
        (255, 59, 48),    # Red
        (0, 122, 255),    # Blue  
        (255, 149, 0),    # Orange
        (175, 82, 222),   # Purple
        (255, 204, 0),    # Yellow
        (50, 173, 230),   # Light Blue
        (255, 45, 85),    # Pink
        (48, 209, 88),    # Light Green
        (90, 200, 250),   # Sky Blue
        (255, 159, 10),   # Amber
        (191, 90, 242),   # Lavender
    ]
    
    @staticmethod
    def get_track_color(track_id):
        """Get a consistent, vibrant color for a track ID"""
        if track_id is None:
            return (68, 189, 50)  # Default green for untracked
        
        # Generate consistent index from track_id
        try:
            track_uuid = uuid.UUID(track_id)
        except (ValueError, TypeError):
            track_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(track_id))
            
        color_index = track_uuid.int % len(FrameVisualizer.COLORS)
        return FrameVisualizer.COLORS[color_index]
    
    @staticmethod
    def draw_rounded_rectangle(frame, pt1, pt2, color, thickness, radius=8):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw main rectangle
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, thickness)
    
    @staticmethod
    def draw_gradient_background(frame, x1, y1, x2, y2, color, alpha=0.85):
        """Draw a modern gradient background with rounded corners"""
        overlay = frame.copy()
        
        # Create gradient effect
        height = y2 - y1
        for i in range(height):
            intensity = 1.0 - (i / height) * 0.3  # Gradient from top to bottom
            gradient_color = tuple(int(c * intensity) for c in color)
            cv2.rectangle(overlay, (x1, y1 + i), (x2, y1 + i + 1), gradient_color, -1)
        
        # Blend with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    @staticmethod
    def draw_detection_on_frame(frame, x1, y1, x2, y2, track_id, detection_data):
        """Draw modern, sleek detection visualization"""
        
        # Get colors
        primary_color = FrameVisualizer.get_track_color(track_id)
        
        # Simple, clean bounding box
        box_thickness = 2
        
        # Draw single clean rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), primary_color, box_thickness)
        
        # Prepare label content without emojis
        if track_id is not None:
            track_short = str(track_id)[:8]
            track_display = f"ID: {track_short}"
        else:
            track_display = "NEW"
        
        # Classification results without icons
        classification_lines = []
        
        for level, key in [("family", "family_confidence"), 
                          ("genus", "genus_confidence"), 
                          ("species", "species_confidence")]:
            if detection_data.get(level):
                conf = detection_data.get(key, 0)
                name = detection_data[level]
                # Truncate long names
                if len(name) > 18:
                    name = name[:15] + "..."
                level_short = level[0].upper()  # F, G, S
                classification_lines.append(f"{level_short}: {name}")
                classification_lines.append(f"   {conf:.1%}")
        
        if not classification_lines and track_id is None:
            return
        
        # Calculate label box dimensions with smaller, lighter font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        padding = 8
        line_spacing = 6
        
        # Calculate text dimensions
        all_lines = [track_display] + classification_lines
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in all_lines]
        max_w = max(size[0] for size in text_sizes) if text_sizes else 100
        text_h = text_sizes[0][1] if text_sizes else 20
        
        total_h = len(all_lines) * (text_h + line_spacing) + padding * 2
        label_w = max_w + padding * 2
        
        # Position label box (above bbox, or below if no space)
        label_x1 = max(0, int(x1))
        label_y1 = max(0, int(y1) - total_h - 5)
        if label_y1 < 0:
            label_y1 = int(y2) + 5
        
        label_x2 = min(frame.shape[1], label_x1 + label_w)
        label_y2 = min(frame.shape[0], label_y1 + total_h)
        
        # Draw modern gradient background with rounded corners
        FrameVisualizer.draw_gradient_background(frame, label_x1, label_y1, label_x2, label_y2, 
                                               (20, 20, 20), alpha=0.88)
        
        # Add subtle border
        FrameVisualizer.draw_rounded_rectangle(frame, 
                                             (label_x1, label_y1), 
                                             (label_x2, label_y2), 
                                             primary_color, 1, radius=6)
        
        # Draw text with modern styling
        current_y = label_y1 + padding + text_h
        
        for i, line in enumerate(all_lines):
            if i == 0:  # Track ID line - use primary color
                text_color = primary_color
                line_thickness = 1
            elif "%" in line:  # Confidence lines - use lighter color
                text_color = (160, 160, 160)
                line_thickness = 1
            else:  # Classification name lines - use white
                text_color = (255, 255, 255)
                line_thickness = 1
            
            # Add subtle text shadow for readability
            cv2.putText(frame, line, (label_x1 + padding + 1, current_y + 1), 
                       font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Main text
            cv2.putText(frame, line, (label_x1 + padding, current_y), 
                       font, font_scale, text_color, line_thickness, cv2.LINE_AA)
            
            current_y += text_h + line_spacing

# ============================================================================
# MAIN PROCESSING CLASS
# ============================================================================

class VideoInferenceProcessor:
    def __init__(self, species_list, yolo_model_path, hierarchical_model_path, 
                 confidence_threshold=0.35, device_id="video_processor"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.species_list = species_list
        self.confidence_threshold = confidence_threshold
        self.device_id = device_id
        
        print(f"Using device: {self.device}")
        
        # Build taxonomy from species list
        self.taxonomy, self.species_to_genus, self.genus_to_family = get_taxonomy(species_list)
        self.level_to_idx, self.idx_to_level = create_mappings(self.taxonomy, species_list)
        self.family_list = sorted(self.taxonomy[1])
        self.genus_list = sorted(list(self.taxonomy[2].keys()))
        
        # Load models
        print(f"Loading YOLO model from {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        print(f"Loading hierarchical model from {hierarchical_model_path}")
        checkpoint = torch.load(hierarchical_model_path, map_location='cpu')
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        num_classes_per_level = [len(self.family_list), len(self.genus_list), len(self.species_list)]
        print(f"Model architecture: {num_classes_per_level} classes per level")
        
        self.classification_model = HierarchicalInsectClassifier(num_classes_per_level)
        self.classification_model.load_state_dict(state_dict, strict=False)
        self.classification_model.to(self.device)
        self.classification_model.eval()
        
        # Classification preprocessing
        self.classification_transform = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.all_detections = []
        self.frame_count = 0
        print("Models loaded successfully!")
    
    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def convert_bbox_to_normalized(self, x, y, x2, y2, width, height):
        x_center = (x + x2) / 2.0 / width
        y_center = (y + y2) / 2.0 / height
        norm_width = (x2 - x) / width
        norm_height = (y2 - y) / height
        return [x_center, y_center, norm_width, norm_height]
    
    def store_detection(self, detection_data, timestamp, frame_time_seconds, track_id, bbox):
        """Store detection for final aggregation"""
        payload = {
            "timestamp": timestamp,
            "frame_time_seconds": frame_time_seconds,
            "track_id": track_id,
            "bbox": bbox,
            **detection_data
        }
        
        self.all_detections.append(payload)
        
        # Print frame-by-frame prediction (only for processed frames)
        track_display = str(track_id)[:8] if track_id else "NEW"
        species = payload.get('species', 'Unknown')
        species_conf = payload.get('species_confidence', 0)
        print(f"Processed {frame_time_seconds:6.2f}s | Track {track_display} | {species} ({species_conf:.1%})")
    
    # ------------------------------------------------------------------------
    # DETECTION AND CLASSIFICATION METHODS
    # ------------------------------------------------------------------------
    
    def process_classification_results(self, classification_outputs):
        """Process raw classification outputs to get predictions and probabilities"""
        family_output = classification_outputs[0].cpu().numpy().flatten()
        genus_output = classification_outputs[1].cpu().numpy().flatten()  
        species_output = classification_outputs[2].cpu().numpy().flatten()
        
        # Apply softmax to get probabilities
        family_probs = torch.softmax(classification_outputs[0], dim=1).cpu().numpy().flatten()
        genus_probs = torch.softmax(classification_outputs[1], dim=1).cpu().numpy().flatten()
        species_probs = torch.softmax(classification_outputs[2], dim=1).cpu().numpy().flatten()
        
        # Get top predictions
        family_idx = np.argmax(family_probs)
        genus_idx = np.argmax(genus_probs)
        species_idx = np.argmax(species_probs)
        
        # Get names
        family_name = self.family_list[family_idx] if family_idx < len(self.family_list) else f"Family_{family_idx}"
        genus_name = self.genus_list[genus_idx] if genus_idx < len(self.genus_list) else f"Genus_{genus_idx}"
        species_name = self.species_list[species_idx] if species_idx < len(self.species_list) else f"Species_{species_idx}"
        
        detection_data = {
            "family": family_name,
            "genus": genus_name, 
            "species": species_name,
            "family_confidence": float(family_probs[family_idx]),
            "genus_confidence": float(genus_probs[genus_idx]),
            "species_confidence": float(species_probs[species_idx]),
            "family_probs": family_probs.tolist(),
            "genus_probs": genus_probs.tolist(),
            "species_probs": species_probs.tolist()
        }
        
        return detection_data
    
    def _extract_yolo_detections(self, frame):
        """Extract and validate YOLO detections from frame"""
        with torch.no_grad():
            results = self.yolo_model(frame, conf=self.confidence_threshold, iou=0.5, verbose=False)
        
        detections = results[0].boxes
        valid_detections = []
        valid_detection_data = []
        
        if detections is not None and len(detections) > 0:
            height, width = frame.shape[:2]
            
            for box in detections:
                xyxy = box.xyxy.cpu().numpy().flatten()
                confidence = box.conf.cpu().numpy().item()
                
                if confidence < self.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = xyxy[:4]
                
                # Clamp coordinates
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Store detection for tracking (x1, y1, x2, y2 format)
                valid_detections.append([x1, y1, x2, y2])
                valid_detection_data.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })
        
        return valid_detections, valid_detection_data
    
    def _classify_detection(self, frame, x1, y1, x2, y2):
        """Perform hierarchical classification on a detection crop"""
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        input_tensor = self.classification_transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            classification_outputs = self.classification_model(input_tensor)
        
        return self.process_classification_results(classification_outputs)
    
    def _process_single_detection(self, frame, det_data, track_id, frame_time_seconds):
        """Process a single detection: classify, store, and visualize"""
        x1, y1, x2, y2 = det_data['x1'], det_data['y1'], det_data['x2'], det_data['y2']
        height, width = frame.shape[:2]
        
        # Perform classification
        detection_data = self._classify_detection(frame, x1, y1, x2, y2)
        
        # Store detection for aggregation
        bbox = self.convert_bbox_to_normalized(x1, y1, x2, y2, width, height)
        timestamp = datetime.now().isoformat()
        self.store_detection(detection_data, timestamp, frame_time_seconds, track_id, bbox)
        
        # Visualization
        FrameVisualizer.draw_detection_on_frame(frame, x1, y1, x2, y2, track_id, detection_data)

    def process_frame(self, frame, frame_time_seconds, tracker, global_frame_count):
        """Process a single frame from the video."""
        self.frame_count += 1
        
        # Extract YOLO detections
        valid_detections, valid_detection_data = self._extract_yolo_detections(frame)
        
        # Update tracker with detections
        track_ids = tracker.update(valid_detections, global_frame_count)
        
        # Process each detection with its track ID
        for i, det_data in enumerate(valid_detection_data):
            track_id = track_ids[i] if i < len(track_ids) else None
            self._process_single_detection(frame, det_data, track_id, frame_time_seconds)
        
        return frame
    
    # ------------------------------------------------------------------------
    # AGGREGATION AND RESULTS METHODS
    # ------------------------------------------------------------------------

    def hierarchical_aggregation(self):
        """
        Perform hierarchical aggregation of results per track ID.
        
        CONFIDENCE CALCULATION EXPLANATION:
        1. For each track, average the softmax probabilities across all detections
        2. Use hierarchical selection: 
           - Find best family (highest averaged family probability)
           - Find best genus within that family (highest averaged genus probability among genera in that family)
           - Find best species within that genus (highest averaged species probability among species in that genus)
        3. Final confidence = averaged probability of the selected class at each taxonomic level
        """
        print("\n=== Performing Hierarchical Aggregation ===")
        
        # Group detections by track ID
        track_detections = defaultdict(list)
        for detection in self.all_detections:
            if detection['track_id'] is not None:
                track_detections[detection['track_id']].append(detection)
        
        aggregated_results = []
        
        for track_id, detections in track_detections.items():
            print(f"\nProcessing Track ID: {track_id} ({len(detections)} detections)")
            
            # Aggregate probabilities across all detections for this track
            prob_sums = [
                np.zeros(len(self.family_list)),
                np.zeros(len(self.genus_list)), 
                np.zeros(len(self.species_list))
            ]
            
            for detection in detections:
                prob_sums[0] += np.array(detection['family_probs'])
                prob_sums[1] += np.array(detection['genus_probs'])
                prob_sums[2] += np.array(detection['species_probs'])
            
            # Average the probabilities
            prob_avgs = [prob_sum / len(detections) for prob_sum in prob_sums]
            
            # Hierarchical selection: Start with family
            best_family_idx = np.argmax(prob_avgs[0])
            best_family = self.family_list[best_family_idx]
            best_family_prob = prob_avgs[0][best_family_idx]
            print(f"  Best family: {best_family} (prob: {best_family_prob:.3f})")
            
            # Find genera belonging to this family
            family_genera_indices = [i for i, genus in enumerate(self.genus_list) 
                                   if genus in self.genus_to_family and self.genus_to_family[genus] == best_family]
            
            if family_genera_indices:
                family_genus_probs = prob_avgs[1][family_genera_indices]
                best_genus_idx = family_genera_indices[np.argmax(family_genus_probs)]
            else:
                best_genus_idx = np.argmax(prob_avgs[1])
            
            best_genus = self.genus_list[best_genus_idx]
            best_genus_prob = prob_avgs[1][best_genus_idx]
            print(f"  Best genus: {best_genus} (prob: {best_genus_prob:.3f})")
            
            # Find species belonging to this genus
            genus_species_indices = [i for i, species in enumerate(self.species_list) 
                                   if species in self.species_to_genus and self.species_to_genus[species] == best_genus]
            
            if genus_species_indices:
                genus_species_probs = prob_avgs[2][genus_species_indices]
                best_species_idx = genus_species_indices[np.argmax(genus_species_probs)]
            else:
                best_species_idx = np.argmax(prob_avgs[2])
            
            best_species = self.species_list[best_species_idx]
            best_species_prob = prob_avgs[2][best_species_idx]
            print(f"  Best species: {best_species} (prob: {best_species_prob:.3f})")
            
            # Calculate track statistics
            frame_times = [d['frame_time_seconds'] for d in detections]
            first_frame, last_frame = min(frame_times), max(frame_times)
            
            aggregated_results.append({
                'track_id': track_id,
                'num_detections': len(detections),
                'first_frame_time': first_frame,
                'last_frame_time': last_frame,
                'duration': last_frame - first_frame,
                'final_family': best_family,
                'final_genus': best_genus,
                'final_species': best_species,
                'family_confidence': best_family_prob,
                'genus_confidence': best_genus_prob,
                'species_confidence': best_species_prob
            })
        
        return aggregated_results
    
    def print_simplified_summary(self, aggregated_results):
        """Print a simplified, readable summary of tracking results"""
        print("\n" + "="*60)
        print("🐛 INSECT TRACKING SUMMARY")
        print("="*60)
        
        if not aggregated_results:
            print("No insects were tracked in this video.")
            return
        
        # Sort by number of detections (most active insects first)
        sorted_results = sorted(aggregated_results, key=lambda x: x['num_detections'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            duration = result['duration']
            detection_count = result['num_detections']
            
            print(f"\n🐞 Insect {i}:")
            print(f"   Detections: {detection_count}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Family: {result['final_family']}")
            print(f"   Genus: {result['final_genus']}")
            print(f"   Species: {result['final_species']}")
            print(f"   Confidence: {result['species_confidence']:.1%}")
        
        print(f"\n📈 Total: {len(aggregated_results)} unique insects tracked")
        print("="*60)
    
    def save_results_table(self, aggregated_results, output_path):
        """Save aggregated results to CSV"""
        df = pd.DataFrame(aggregated_results).sort_values('track_id')
        csv_path = str(output_path).replace('.mp4', '_results.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Results saved to: {csv_path}")
        
        # Print simplified summary
        self.print_simplified_summary(aggregated_results)
        
        return df

# ============================================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================================

def process_video(video_path, processor, output_video_path=None, show_video=False, tracker_max_frames=30, fps=None):
    """Process an MP4 video file frame by frame.
    
    Args:
        fps (float, optional): FPS for processing and output. If provided, frames will be skipped 
                              to match this rate and output video will use this FPS. If None, all frames are processed.
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / input_fps if input_fps > 0 else 0
    
    print(f"Processing video: {video_path}")
    print(f"Properties: {total_frames} frames, {input_fps:.2f} FPS, {duration:.2f}s duration")
    
    # Initialize tracker and output writer
    tracker = InsectTracker(height, width, max_frames=tracker_max_frames, debug=False)
    out = None
    
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        write_fps = fps if fps is not None else input_fps
        out = cv2.VideoWriter(output_video_path, fourcc, write_fps, (width, height))
        print(f"Output video: {output_video_path} at {write_fps:.2f} FPS")
    
    frame_number = 0
    processed_frame_count = 0
    start_time = time.time()
    
    # Calculate frame skip interval if fps is specified
    frame_skip_interval = None
    if fps is not None and fps > 0 and input_fps > 0:
        frame_skip_interval = max(1, int(input_fps / fps))
        print(f"Processing every {frame_skip_interval} frame(s) to achieve ~{fps:.2f} FPS")
        print(f"Tracker will only receive the processed frames, not all {total_frames} frames")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time_seconds = frame_number / input_fps if input_fps > 0 else 0
            
            # Check if we should process this frame
            should_process = True
            if frame_skip_interval is not None:
                should_process = (frame_number % frame_skip_interval == 0)
            
            if should_process:
                # Process the frame
                processed_frame = processor.process_frame(frame, frame_time_seconds, tracker, frame_number)
                processed_frame_count += 1
                last_processed_frame = processed_frame.copy()  # Keep for frame duplication
                
                # Show video if requested
                if show_video:
                    cv2.imshow('Video Inference', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User requested quit")
                        break
                
                # If we're skipping frames, skip ahead to save time
                if frame_skip_interval is not None and frame_skip_interval > 1:
                    # Skip the next (frame_skip_interval - 1) frames
                    for _ in range(frame_skip_interval - 1):
                        ret, _ = cap.read()
                        if not ret:
                            break
                        frame_number += 1
            else:
                # Use the last processed frame to maintain video length
                if 'last_processed_frame' in locals():
                    processed_frame = last_processed_frame.copy()
                else:
                    processed_frame = frame  # Fallback for first frames
            
            # Write to output video if requested (always write to maintain duration)
            if out:
                out.write(processed_frame)
            
            frame_number += 1
            
            # Progress update based on processed frames only
            if should_process and processed_frame_count % 25 == 0:
                if frame_skip_interval is not None:
                    estimated_total_processed = total_frames // frame_skip_interval
                    progress = (processed_frame_count / estimated_total_processed) * 100 if estimated_total_processed > 0 else 0
                    print(f"Processed: {processed_frame_count}/{estimated_total_processed} frames ({progress:.1f}%)")
                else:
                    progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processed: {processed_frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    finally:
        cap.release()
        if out:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
    
    processing_time = time.time() - start_time
    print(f"\nProcessing complete! {processed_frame_count}/{frame_number} frames processed in {processing_time:.2f}s")
    if processed_frame_count > 0:
        print(f"Processing speed: {processed_frame_count/processing_time:.2f} FPS, Detections: {len(processor.all_detections)}")
    else:
        print(f"No frames processed, Detections: {len(processor.all_detections)}")
    
    # Perform hierarchical aggregation and save results
    aggregated_results = processor.hierarchical_aggregation()
    if output_video_path:
        processor.save_results_table(aggregated_results, output_video_path)
    
    return aggregated_results

# ============================================================================
# MAIN ENTRY POINT FUNCTIONS
# ============================================================================

def inference(species_list, yolo_model_path, hierarchical_model_path, confidence_threshold, 
                  video_path, output_path, tracker_max_frames, fps=None):
    """
    Run inference on a single video file.
    
    Args:
        species_list (list): List of species names for classification
        yolo_model_path (str): Path to YOLO model weights
        hierarchical_model_path (str): Path to hierarchical classification model weights
        confidence_threshold (float): Confidence threshold for detections
        video_path (str): Path to input video file
        output_path (str): Path for output video file (including filename)
        tracker_max_frames (int): Maximum frames for tracker context
        fps (float, optional): Processing and output FPS. If provided, frames will be skipped 
                              to match this rate and output video will use this FPS. If None, all frames are processed.
    
    Returns:
        dict: Summary of processing results
    """
    # Check if input video exists
    if not os.path.exists(video_path):
        error_msg = f"Video file not found: {video_path}"
        print(f"Error: {error_msg}")
        return {"error": error_msg}

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Processing single video: {video_path}")

    # Create processor instance
    print("Initializing models...")
    processor = VideoInferenceProcessor(
        species_list=species_list,
        yolo_model_path=yolo_model_path,
        hierarchical_model_path=hierarchical_model_path,
        confidence_threshold=confidence_threshold
    )
    
    # Track processing results
    processing_results = {
        "video_file": os.path.basename(video_path),
        "output_path": output_path,
        "success": False,
        "detections": 0,
        "tracks": 0,
        "error": None
    }
    
    print(f"\n{'='*20}\nProcessing: {video_path}\n{'='*20}")

    try:
        # Process the video
        aggregated_results = process_video(
            video_path=video_path,
            processor=processor,
            output_video_path=output_path,
            show_video=False,
            tracker_max_frames=tracker_max_frames,
            fps=fps
        )
        
        processing_results.update({
            "success": True,
            "detections": len(processor.all_detections),
            "tracks": len(aggregated_results)
        })
        
        print(f"Finished processing. Output saved to {output_path}")
        
    except Exception as e:
        error_msg = f"Failed to process {os.path.basename(video_path)}: {str(e)}"
        print(f"Error: {error_msg}")
        processing_results["error"] = error_msg
    
    if processing_results["success"]:
        print(f"\nProcessing complete!")
        print(f"Total detections: {processing_results['detections']}")
        print(f"Total tracks: {processing_results['tracks']}")
    
    return processing_results


def main():
    """Example usage for processing a single video."""
    # Define your species list (replace with your actual species)
    species_list = [
        "Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
        "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
        "Eristalis tenax", "unknown"
    ]
    
    # Paths (replace with your actual paths)
    video_path = "input_videos/sample_video.mp4"
    output_path = "output_videos/sample_video_predictions.mp4"
    yolo_model_path = "weights/yolo_model.pt"
    hierarchical_model_path = "weights/hierarchical_model.pth"
    
    # Run inference
    results = inference(
        species_list=species_list,
        yolo_model_path=yolo_model_path,
        hierarchical_model_path=hierarchical_model_path,
        confidence_threshold=0.35,
        video_path=video_path,
        output_path=output_path,
        tracker_max_frames=60,
        fps=None  # Set to e.g. 5.0 to process only 5 frames per second
    )
    
    return results


if __name__ == "__main__":
    main() 