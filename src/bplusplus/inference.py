"""
Video inference module for insect detection and classification.

Processes video files through a multi-phase pipeline:
    1. Detection & Tracking: Motion-based detection with Hungarian tracking
    2. Topology Analysis: Path analysis to confirm insect-like movement
    3. Classification: Hierarchical classification of confirmed tracks
    4. Video Rendering: Annotated output videos (optional)

Usage:
    from bplusplus import inference
    result = inference(model_path, video_path, output_dir)
    
    # Or via CLI:
    python -m bplusplus.inference --video input.mp4 --model model.pt \\
        --output-dir results/
    
    # Optionally override species list from checkpoint:
    python -m bplusplus.inference --video input.mp4 --model model.pt \\
        --output-dir results/ --species "Apis mellifera" "Bombus terrestris"
"""

import cv2
import time
import os
import yaml
import json
import numpy as np
import pandas as pd
import argparse
import logging
import uuid
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests

from bugspot import (
    InsectTracker,
    DEFAULT_DETECTION_CONFIG,
    ScaledDetector,
    get_default_config,
    resolve_detection_params,
    calculate_revisit_ratio,
    calculate_progression_ratio,
    calculate_directional_variance,
    check_track_consistency,
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Torch serialization compatibility
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        'torch.LongTensor',
        'torch.cuda.LongTensor',
        'torch.FloatStorage',
        'torch.cuda.FloatStorage',
    ])


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Classification:
    """Hierarchical classification result."""
    family: str
    genus: str
    species: str
    family_confidence: float
    genus_confidence: float
    species_confidence: float
    family_probs: List[float] = field(default_factory=list)
    genus_probs: List[float] = field(default_factory=list)
    species_probs: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'family': self.family,
            'genus': self.genus,
            'species': self.species,
            'family_confidence': self.family_confidence,
            'genus_confidence': self.genus_confidence,
            'species_confidence': self.species_confidence,
            'family_probs': self.family_probs,
            'genus_probs': self.genus_probs,
            'species_probs': self.species_probs,
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load detection configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file (.yaml, .yml, or .json)
        
    Returns:
        dict: Configuration parameters merged with defaults
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif ext == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
    
    # Merge with defaults
    params = get_default_config()
    for key, value in config.items():
        if key in params:
            params[key] = value
        else:
            logger.warning(f"Unknown config parameter ignored: {key}")
    
    return params


# =============================================================================
# TRACK METRICS
# =============================================================================

def compute_full_track_metrics(path: List[Tuple[float, float]], params: Dict) -> Dict:
    """
    Compute all topology metrics for a track path, even if it is too short
    to be considered for confirmation.

    Unlike ``analyze_path_topology``, this always returns the raw measured
    values (using NaN only when a metric is genuinely undefined) alongside
    per-criterion pass flags and the config thresholds used. This allows
    unconfirmed tracks to appear in the tracks CSV with diagnostic info
    explaining *why* they failed.

    Args:
        path: List of (x, y) centroid positions for the track.
        params: RESOLVED detection config parameters (see
            ``bugspot.resolve_detection_params``). Pixel-scale thresholds
            like ``min_displacement`` and ``revisit_radius`` must be in
            absolute pixel units; the matching ``{key}_frac`` values from
            the original fraction-based config are also recorded in the
            output for traceability.

    Returns:
        dict: Raw metrics, per-criterion pass flags, and both the resolved
        pixel thresholds and their originating fraction-of-image values.
    """
    min_path_points = params.get("min_path_points", 10)
    min_displacement = params.get("min_displacement", 50)
    max_revisit_ratio = params.get("max_revisit_ratio", 0.30)
    min_progression_ratio = params.get("min_progression_ratio", 0.70)
    max_directional_variance = params.get("max_directional_variance", 0.90)
    revisit_radius = params.get("revisit_radius", 50)

    # Original fraction values from the config (may be None if params was
    # never resolved, e.g. when called directly with absolute pixel values).
    min_displacement_frac = params.get("min_displacement_frac")
    revisit_radius_frac = params.get("revisit_radius_frac")

    num_path_points = len(path)
    path_points_pass = num_path_points >= min_path_points

    if num_path_points >= 2:
        path_arr = np.array(path)
        net_displacement = float(np.linalg.norm(path_arr[-1] - path_arr[0]))
        revisit_ratio = float(calculate_revisit_ratio(path_arr, revisit_radius))
        progression_ratio = float(calculate_progression_ratio(path_arr))
        directional_variance = float(calculate_directional_variance(path_arr))

        step_dists = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
        total_path_length = float(step_dists.sum())
    else:
        net_displacement = float("nan")
        revisit_ratio = float("nan")
        progression_ratio = float("nan")
        directional_variance = float("nan")
        total_path_length = float("nan")

    displacement_pass = (
        net_displacement >= min_displacement if not np.isnan(net_displacement) else False
    )
    revisit_pass = (
        revisit_ratio <= max_revisit_ratio if not np.isnan(revisit_ratio) else False
    )
    progression_pass = (
        progression_ratio >= min_progression_ratio if not np.isnan(progression_ratio) else False
    )
    variance_pass = (
        directional_variance <= max_directional_variance
        if not np.isnan(directional_variance) else False
    )

    passes_topology = (
        path_points_pass
        and displacement_pass
        and revisit_pass
        and progression_pass
        and variance_pass
    )

    return {
        # Raw measurements (all in pixels for length/displacement)
        "num_path_points": num_path_points,
        "total_path_length": total_path_length,
        "net_displacement": net_displacement,
        "revisit_ratio": revisit_ratio,
        "progression_ratio": progression_ratio,
        "directional_variance": directional_variance,
        # Per-criterion pass flags
        "path_points_pass": path_points_pass,
        "displacement_pass": displacement_pass,
        "revisit_pass": revisit_pass,
        "progression_pass": progression_pass,
        "directional_variance_pass": variance_pass,
        "passes_topology": passes_topology,
        # Resolved pixel thresholds that were actually compared against
        "min_path_points": min_path_points,
        "min_displacement": min_displacement,
        "max_revisit_ratio": max_revisit_ratio,
        "min_progression_ratio": min_progression_ratio,
        "max_directional_variance": max_directional_variance,
        "revisit_radius": revisit_radius,
        # Original fraction-of-image-width values from the config file
        "min_displacement_frac": min_displacement_frac,
        "revisit_radius_frac": revisit_radius_frac,
    }


# =============================================================================
# TAXONOMY UTILITIES
# =============================================================================

def get_taxonomy(species_list: List[str]) -> Tuple[Dict, Dict[str, str], Dict[str, str]]:
    """
    Retrieve taxonomic information from GBIF API.
    
    Args:
        species_list: List of species names
        
    Returns:
        tuple: (taxonomy_dict, species_to_genus, genus_to_family)
    """
    taxonomy = {1: [], 2: {}, 3: {}}
    species_to_genus = {}
    genus_to_family = {}
    
    species_for_gbif = [s for s in species_list if s.lower() != 'unknown']
    has_unknown = len(species_for_gbif) != len(species_list)
    
    logger.info(f"Building taxonomy from GBIF for {len(species_for_gbif)} species")
    print(f"\n{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)
    
    for species_name in species_for_gbif:
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
                    logger.error(f"Species '{species_name}' missing family/genus")
            else:
                print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                logger.error(f"Species '{species_name}' not found in GBIF")
        except Exception as e:
            print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
            logger.error(f"Error for '{species_name}': {e}")

    if has_unknown:
        if "Unknown" not in taxonomy[1]:
            taxonomy[1].append("Unknown")
        taxonomy[2]["Unknown"] = "Unknown"
        taxonomy[3]["unknown"] = "Unknown"
        species_to_genus["unknown"] = "Unknown"
        genus_to_family["Unknown"] = "Unknown"
        print(f"{'unknown':<30} {'Unknown':<20} {'Unknown':<20} OK")
    
    taxonomy[1] = sorted(set(taxonomy[1]))
    print("-" * 80)
    
    for level, name, items in [(1, "Family", taxonomy[1]), 
                                (2, "Genus", taxonomy[2].keys()), 
                                (3, "Species", species_list)]:
        print(f"\n{name} indices:")
        for i, item in enumerate(items):
            print(f"  {i}: {item}")
    
    logger.info(f"Taxonomy: {len(taxonomy[1])} families, {len(taxonomy[2])} genera, {len(taxonomy[3])} species")
    return taxonomy, species_to_genus, genus_to_family


def create_mappings(taxonomy: Dict, species_list: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
    """Create index mappings from taxonomy."""
    level_to_idx = {}
    idx_to_level = {}

    for level, labels in taxonomy.items():
        if isinstance(labels, list):
            level_to_idx[level] = {label: idx for idx, label in enumerate(labels)}
            idx_to_level[level] = {idx: label for idx, label in enumerate(labels)}
        else:
            sorted_keys = species_list if level == 3 and species_list else sorted(labels.keys())
            level_to_idx[level] = {label: idx for idx, label in enumerate(sorted_keys)}
            idx_to_level[level] = {idx: label for idx, label in enumerate(sorted_keys)}
    
    return level_to_idx, idx_to_level


# =============================================================================
# CLASSIFICATION MODEL
# =============================================================================

class HierarchicalInsectClassifier(nn.Module):
    """
    Hierarchical classifier with ResNet backbone and multi-branch heads.
    
    Outputs predictions for Family, Genus, and Species levels.
    """
    
    def __init__(self, num_classes_per_level: List[int], backbone: str = "resnet50"):
        super().__init__()
        self.backbone = self._build_backbone(backbone)
        self.backbone_name = backbone
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            ) for num_classes in num_classes_per_level
        ])
        self.num_levels = len(num_classes_per_level)
    
    @staticmethod
    def _build_backbone(backbone: str) -> nn.Module:
        """Build ResNet backbone by name."""
        name = backbone.lower()
        if name == "resnet18":
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if name == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if name == "resnet101":
            return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from 'resnet18', 'resnet50', 'resnet101'.")
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        return [branch(features) for branch in self.branches]


# =============================================================================
# VISUALIZATION
# =============================================================================

class FrameVisualizer:
    """Visualization utilities for detection overlay."""
    
    COLORS = [
        (68, 189, 50), (255, 59, 48), (0, 122, 255), (255, 149, 0),
        (175, 82, 222), (255, 204, 0), (50, 173, 230), (255, 45, 85),
        (48, 209, 88), (90, 200, 250), (255, 159, 10), (191, 90, 242),
    ]
    
    @staticmethod
    def get_track_color(track_id: Optional[str]) -> Tuple[int, int, int]:
        if track_id is None:
            return (68, 189, 50)
        try:
            track_uuid = uuid.UUID(track_id)
        except (ValueError, TypeError):
            track_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(track_id))
        return FrameVisualizer.COLORS[track_uuid.int % len(FrameVisualizer.COLORS)]
    
    @staticmethod
    def draw_path(frame: np.ndarray, path: List[Tuple[float, float]], track_id: str) -> None:
        """Draw track path on frame."""
        if len(path) < 2:
            return
        color = FrameVisualizer.get_track_color(track_id)
        path_points = np.array(path, dtype=np.int32)
        cv2.polylines(frame, [path_points], False, color, 2)
        cx, cy = path[-1]
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
    
    @staticmethod
    def draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       track_id: Optional[str], detection_data: Dict) -> None:
        """Draw bounding box and classification label on frame."""
        color = FrameVisualizer.get_track_color(track_id)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        track_display = f"ID: {str(track_id)[:8]}" if track_id else "NEW"
        lines = [track_display]
        
        for level, conf_key in [("family", "family_confidence"), 
                                ("genus", "genus_confidence"), 
                                ("species", "species_confidence")]:
            if detection_data.get(level):
                name = detection_data[level]
                conf = detection_data.get(conf_key, 0)
                name = name[:15] + "..." if len(name) > 18 else name
                lines.append(f"{level[0].upper()}: {name}")
                lines.append(f"   {conf:.1%}")
        
        if not lines[1:] and track_id is None:
            return
        
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        padding, spacing = 8, 6
        text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        max_w = max(s[0] for s in text_sizes)
        text_h = text_sizes[0][1]
        
        total_h = len(lines) * (text_h + spacing) + padding * 2
        label_x1 = max(0, int(x1))
        label_y1 = max(0, int(y1) - total_h - 5)
        if label_y1 < 0:
            label_y1 = int(y2) + 5
        label_x2 = min(frame.shape[1], label_x1 + max_w + padding * 2)
        label_y2 = min(frame.shape[0], label_y1 + total_h)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, 1)
        
        y = label_y1 + padding + text_h
        for i, line in enumerate(lines):
            text_color = color if i == 0 else ((160, 160, 160) if "%" in line else (255, 255, 255))
            cv2.putText(frame, line, (label_x1 + padding, y), font, scale, text_color, thickness, cv2.LINE_AA)
            y += text_h + spacing


# =============================================================================
# VIDEO PROCESSOR
# =============================================================================

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoInferenceProcessor:
    """
    Processes video frames for insect detection and classification.
    
    Pipeline:
        1. Motion detection using GMM background subtraction
        2. Hungarian algorithm tracking
        3. Path topology analysis for confirmation
        4. Hierarchical classification of confirmed tracks
    """
    
    def __init__(
        self,
        params: Dict,
        hierarchical_model_path: Optional[str] = None,
        species_list: Optional[List[str]] = None,
        backbone: str = "resnet50",
        img_size: int = 60,
        classify: bool = True,
    ):
        """
        Initialize the processor.
        
        Args:
            params: Detection parameters dict
            hierarchical_model_path: Path to trained model weights (required if classify=True)
            species_list: Optional list of species names (if None, loaded from checkpoint)
            backbone: ResNet backbone ('resnet18', 'resnet50', 'resnet101')
            img_size: Image size for classification (should match training)
            classify: If True, load model and classify confirmed tracks. If False, detection only.
        """
        self.img_size = img_size
        # Raw (fraction-based) config as provided by the caller. Values
        # like min_area/min_displacement are fractions of image dimensions;
        # they get resolved to absolute pixels in ``setup`` once the video
        # frame size is known.
        self._raw_params = dict(params)
        # Resolved (pixel-based) config — populated by ``setup``. Falls
        # back to the raw dict until then for consumers that inspect it.
        self.params: Dict = dict(params)
        self.classify = classify
        self._is_setup = False

        # Detector — created lazily in ``setup``. ScaledDetector (bugspot) owns
        # the detection-resolution policy: downscaling, param scaling, and
        # mapping bounding boxes back to native pixels.
        self._scaled: Optional[ScaledDetector] = None

        # Track state
        self.all_detections: List[Dict] = []
        self.track_paths: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.track_areas: Dict[str, List[float]] = defaultdict(list)
        
        # Classification model (only if classify=True)
        if classify:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            if hierarchical_model_path is None:
                raise ValueError("hierarchical_model_path is required when classify=True")
            
            print(f"Loading hierarchical model from {hierarchical_model_path}")
            checkpoint = torch.load(hierarchical_model_path, map_location='cpu')
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            # Load species list from checkpoint if not provided
            if species_list is None:
                if 'species_list' in checkpoint:
                    species_list = checkpoint['species_list']
                    print(f"Loaded species list from checkpoint: {len(species_list)} species")
                else:
                    raise ValueError("species_list not found in checkpoint and not provided as argument")
            
            self.species_list = species_list
        
            # Build taxonomy
            self.taxonomy, self.species_to_genus, self.genus_to_family = get_taxonomy(species_list)
            self.level_to_idx, self.idx_to_level = create_mappings(self.taxonomy, species_list)
            self.family_list = sorted(self.taxonomy[1])
            self.genus_list = sorted(self.taxonomy[2].keys())
            
            model_backbone = checkpoint.get("backbone", backbone)
            if model_backbone != backbone:
                print(f"Note: Using backbone '{model_backbone}' from checkpoint (overrides '{backbone}')")
            
            num_classes = [len(self.family_list), len(self.genus_list), len(self.species_list)]
            print(f"Model architecture: {num_classes} classes per level, backbone: {model_backbone}")
            
            self.model = HierarchicalInsectClassifier(num_classes, backbone=model_backbone)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            print("Detection-only mode (no classification)")
            self.species_list = []
            self.family_list = []
            self.genus_list = []
            self.taxonomy = {}
            self.species_to_genus = {}
            self.genus_to_family = {}
        
        print("Processor initialized successfully!")
    
    def setup(self, image_width: int, image_height: int) -> None:
        """
        Resolve fraction-based config to absolute pixels and build the
        detector. Must be called before ``process_frame`` (done automatically
        by ``process_video`` once the video is opened).

        If ``detection_resolution`` is set in the config (a ``(width, height)``
        pixel pair), the detector runs on frames resized to that resolution for
        speed; bounding boxes are scaled back to native resolution so tracking,
        crops, and composites stay full-res. The whole detection-resolution
        policy (downscaling, param scaling, bbox mapping) lives in bugspot's
        ``ScaledDetector`` so it stays identical across all consumers.
        ``self.params`` always holds the NATIVE-resolution resolved config used
        by the tracker, topology, and consistency checks.
        """
        self.params = resolve_detection_params(
            self._raw_params, image_width, image_height
        )
        self._scaled = ScaledDetector(self._raw_params, image_width, image_height)
        self._is_setup = True

    def _classify(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Classification:
        """Classify a detection crop."""
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
        
        probs = [torch.softmax(o, dim=1).cpu().numpy().flatten() for o in outputs]
        idxs = [np.argmax(p) for p in probs]
        
        return Classification(
            family=self.family_list[idxs[0]] if idxs[0] < len(self.family_list) else f"Family_{idxs[0]}",
            genus=self.genus_list[idxs[1]] if idxs[1] < len(self.genus_list) else f"Genus_{idxs[1]}",
            species=self.species_list[idxs[2]] if idxs[2] < len(self.species_list) else f"Species_{idxs[2]}",
            family_confidence=float(probs[0][idxs[0]]),
            genus_confidence=float(probs[1][idxs[1]]),
            species_confidence=float(probs[2][idxs[2]]),
            family_probs=probs[0].tolist(),
            genus_probs=probs[1].tolist(),
            species_probs=probs[2].tolist(),
        )
    
    def process_frame(self, frame: np.ndarray, frame_time: float,
                      tracker: InsectTracker, frame_number: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect and track only (no classification).
        Classification happens later for confirmed tracks only.
        
        Args:
            frame: BGR image frame
            frame_time: Time in seconds
            tracker: InsectTracker instance
            frame_number: Frame index
            
        Returns:
            tuple: (foreground_mask, list of detections with track_ids)
        """
        if not self._is_setup:
            height, width = frame.shape[:2]
            self.setup(width, height)

        height, width = frame.shape[:2]

        # ScaledDetector returns bboxes already mapped to native pixels
        # (and transparently downscales the frame for detection if configured).
        bboxes, fg_mask = self._scaled.detect(frame, frame_number)

        # Track
        track_ids = tracker.update(bboxes, frame_number)

        frame_detections = []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            track_id = track_ids[i] if i < len(track_ids) else None
            
            # Per-frame consistency check: ignore this frame if it
            # jumps too far or changes size too drastically.
            # The track keeps its last good position as reference.
            if track_id:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                if self.track_paths[track_id]:
                    prev_pos = self.track_paths[track_id][-1]
                    prev_area = self.track_areas[track_id][-1] if self.track_areas[track_id] else area
                    
                    if not check_track_consistency(
                        prev_pos, (cx, cy), prev_area, area, 
                        self.params["max_frame_jump"],
                        self.params.get("max_area_change_ratio", 3.0)
                    ):
                        continue
                
                self.track_paths[track_id].append((cx, cy))
                self.track_areas[track_id].append(area)
            
            detection_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_number": frame_number,
                "frame_time_seconds": frame_time,
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [
                    (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                    (x2 - x1) / width, (y2 - y1) / height
                ],
            }
            self.all_detections.append(detection_data)
            frame_detections.append(detection_data)
            
            track_display = str(track_id)[:8] if track_id else "NEW"
            print(f"Frame {frame_time:6.2f}s | Track {track_display} | Detected")
        
        return fg_mask, frame_detections
    
    def classify_confirmed_tracks(self, video_path: str, confirmed_track_ids: Set[str],
                                  crops_dir: Optional[str] = None) -> Dict[str, List[Classification]]:
        """
        Classify only the confirmed tracks by re-reading relevant frames.
        
        Args:
            video_path: Path to original video
            confirmed_track_ids: Set of track IDs that passed topology analysis
            crops_dir: Optional directory to save cropped frames
            
        Returns:
            dict: track_id -> list of classifications
        """
        if not confirmed_track_ids:
            print("No confirmed tracks to classify.")
            return {}
        
        print(f"\nClassifying {len(confirmed_track_ids)} confirmed tracks...")
        
        if crops_dir:
            os.makedirs(crops_dir, exist_ok=True)
            for track_id in confirmed_track_ids:
                track_dir = os.path.join(crops_dir, str(track_id)[:8])
                os.makedirs(track_dir, exist_ok=True)
            print(f"  Saving crops to: {crops_dir}")
        
        frames_to_classify = defaultdict(list)
        for det in self.all_detections:
            if det['track_id'] in confirmed_track_ids:
                frames_to_classify[det['frame_number']].append(det)
        
        if not frames_to_classify:
            return {}
        
        cap = cv2.VideoCapture(video_path)
        track_classifications: Dict[str, List[Classification]] = defaultdict(list)
        
        frame_numbers = sorted(frames_to_classify.keys())
        current_frame = 0
        classified_count = 0
        
        for target_frame in frame_numbers:
            while current_frame < target_frame:
                cap.read()
                current_frame += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            
            for det in frames_to_classify[target_frame]:
                x1, y1, x2, y2 = det['bbox']
                classification = self._classify(frame, x1, y1, x2, y2)
                
                det.update(classification.to_dict())
                track_classifications[det['track_id']].append(classification)
                classified_count += 1
                
                if crops_dir:
                    track_id = det['track_id']
                    track_dir = os.path.join(crops_dir, str(track_id)[:8])
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        crop_path = os.path.join(track_dir, f"frame_{target_frame:06d}.jpg")
                        cv2.imwrite(crop_path, crop)
                
                if classified_count % 20 == 0:
                    print(f"  Classified {classified_count} detections...", end='\r')
        
        cap.release()
        print(f"\n✓ Classified {classified_count} detections from {len(confirmed_track_ids)} tracks")
        if crops_dir:
            print(f"✓ Saved {classified_count} crops to {crops_dir}")
        
        return dict(track_classifications)
    
    def save_crops(self, video_path: str, confirmed_track_ids: Set[str],
                   crops_dir: str) -> int:
        """
        Save cropped detection frames for confirmed tracks (no classification needed).
        
        Args:
            video_path: Path to original video
            confirmed_track_ids: Set of track IDs that passed topology analysis
            crops_dir: Directory to save cropped frames
            
        Returns:
            int: Number of crops saved
        """
        if not confirmed_track_ids:
            print("No confirmed tracks for crop saving.")
            return 0
        
        os.makedirs(crops_dir, exist_ok=True)
        for track_id in confirmed_track_ids:
            track_dir = os.path.join(crops_dir, str(track_id)[:8])
            os.makedirs(track_dir, exist_ok=True)
        
        frames_to_crop = defaultdict(list)
        for det in self.all_detections:
            if det['track_id'] in confirmed_track_ids:
                frames_to_crop[det['frame_number']].append(det)
        
        if not frames_to_crop:
            return 0
        
        cap = cv2.VideoCapture(video_path)
        frame_numbers = sorted(frames_to_crop.keys())
        current_frame = 0
        saved_count = 0
        
        for target_frame in frame_numbers:
            while current_frame < target_frame:
                cap.read()
                current_frame += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            
            for det in frames_to_crop[target_frame]:
                x1, y1, x2, y2 = det['bbox']
                track_id = det['track_id']
                track_dir = os.path.join(crops_dir, str(track_id)[:8])
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    crop_path = os.path.join(track_dir, f"frame_{target_frame:06d}.jpg")
                    cv2.imwrite(crop_path, crop)
                    saved_count += 1
                
                if saved_count % 20 == 0:
                    print(f"  Saved {saved_count} crops...", end='\r')
        
        cap.release()
        print(f"\n✓ Saved {saved_count} crops to {crops_dir}")
        return saved_count
    
    def analyze_tracks(self) -> Tuple[Set[str], Dict]:
        """
        Analyze all tracks (confirmed and unconfirmed) to determine which
        pass topology, collecting full diagnostic info for each.

        Returns:
            tuple: (confirmed_track_ids set, all_track_info dict)
        """
        print("\n" + "="*60)
        print("TRACK TOPOLOGY ANALYSIS")
        print("="*60)

        track_detections = defaultdict(list)
        for det in self.all_detections:
            if det['track_id']:
                track_detections[det['track_id']].append(det)

        confirmed_track_ids: Set[str] = set()
        all_track_info: Dict = {}

        for track_id, detections in track_detections.items():
            path = self.track_paths.get(track_id, [])
            areas = self.track_areas.get(track_id, [])
            metrics = compute_full_track_metrics(path, self.params)

            frame_numbers = [d['frame_number'] for d in detections]
            frame_times = [d['frame_time_seconds'] for d in detections]

            if areas:
                mean_area = float(np.mean(areas))
                min_area = float(np.min(areas))
                max_area = float(np.max(areas))
            else:
                mean_area = min_area = max_area = float("nan")

            track_info = {
                'track_id': track_id,
                'status': 'confirmed' if metrics['passes_topology'] else 'unconfirmed',
                'num_detections': len(detections),
                'first_frame': min(frame_numbers),
                'last_frame': max(frame_numbers),
                'first_frame_time': min(frame_times),
                'last_frame_time': max(frame_times),
                'duration': max(frame_times) - min(frame_times),
                'mean_area': mean_area,
                'min_area_observed': min_area,
                'max_area_observed': max_area,
                **metrics,
            }
            all_track_info[track_id] = track_info

            status = "✓ CONFIRMED" if metrics['passes_topology'] else "? unconfirmed"
            print(f"Track {str(track_id)[:8]}: {len(detections)} detections, "
                  f"{track_info['duration']:.1f}s - {status}")

            if metrics['passes_topology']:
                confirmed_track_ids.add(track_id)

        print(f"\n✓ {len(confirmed_track_ids)} confirmed / {len(track_detections)} total tracks")
        return confirmed_track_ids, all_track_info
    
    def detection_only_results(self, confirmed_track_ids: Set[str]) -> List[Dict]:
        """
        Generate results for detection-only mode (no classification).
        All classification fields are set to NaN.
        
        Args:
            confirmed_track_ids: Set of confirmed track IDs
            
        Returns:
            list: Results with NaN classification fields
        """
        print("\n" + "="*60)
        print("DETECTION-ONLY RESULTS (No Classification)")
        print("="*60)
        
        track_detections = defaultdict(list)
        for det in self.all_detections:
            if det['track_id'] in confirmed_track_ids:
                track_detections[det['track_id']].append(det)
        
        results = []
        for track_id, detections in track_detections.items():
            path = self.track_paths.get(track_id, [])
            metrics = compute_full_track_metrics(path, self.params)

            frame_times = [d['frame_time_seconds'] for d in detections]

            result = {
                'track_id': track_id,
                'num_detections': len(detections),
                'first_frame_time': min(frame_times),
                'last_frame_time': max(frame_times),
                'duration': max(frame_times) - min(frame_times),
                'final_family': float('nan'),
                'final_genus': float('nan'),
                'final_species': float('nan'),
                'family_confidence': float('nan'),
                'genus_confidence': float('nan'),
                'species_confidence': float('nan'),
                **metrics,
            }
            results.append(result)

            print(f"Track {str(track_id)[:8]}: {len(detections)} detections, "
                  f"{result['duration']:.1f}s")

        return results
    
    def hierarchical_aggregation(self, confirmed_track_ids: Set[str]) -> List[Dict]:
        """
        Aggregate predictions for confirmed tracks using hierarchical selection.
        Must be called AFTER classify_confirmed_tracks().
        
        Args:
            confirmed_track_ids: Set of confirmed track IDs
            
        Returns:
            list: Aggregated results for confirmed tracks only
        """
        print("\n" + "="*60)
        print("HIERARCHICAL AGGREGATION (Confirmed Tracks)")
        print("="*60)
        
        track_detections = defaultdict(list)
        for det in self.all_detections:
            if det['track_id'] in confirmed_track_ids:
                track_detections[det['track_id']].append(det)
        
        results = []
        for track_id, detections in track_detections.items():
            if 'family_probs' not in detections[0]:
                print(f"Warning: Track {str(track_id)[:8]} has no classifications, skipping")
                continue
            
            print(f"\nTrack {str(track_id)[:8]}: {len(detections)} classified detections")

            path = self.track_paths.get(track_id, [])
            metrics = compute_full_track_metrics(path, self.params)
            
            # Average probabilities
            prob_avgs = [
                np.mean([d['family_probs'] for d in detections], axis=0),
                np.mean([d['genus_probs'] for d in detections], axis=0),
                np.mean([d['species_probs'] for d in detections], axis=0),
            ]
            
            # Hierarchical selection
            best_family_idx = np.argmax(prob_avgs[0])
            best_family = self.family_list[best_family_idx]
            
            family_genera = [i for i, g in enumerate(self.genus_list) 
                           if self.genus_to_family.get(g) == best_family]
            if family_genera:
                best_genus_idx = family_genera[np.argmax(prob_avgs[1][family_genera])]
            else:
                best_genus_idx = np.argmax(prob_avgs[1])
            best_genus = self.genus_list[best_genus_idx]
            
            genus_species = [i for i, s in enumerate(self.species_list)
                           if self.species_to_genus.get(s) == best_genus]
            if genus_species:
                best_species_idx = genus_species[np.argmax(prob_avgs[2][genus_species])]
            else:
                best_species_idx = np.argmax(prob_avgs[2])
            best_species = self.species_list[best_species_idx]
            
            frame_times = [d['frame_time_seconds'] for d in detections]
            
            result = {
                'track_id': track_id,
                'num_detections': len(detections),
                'first_frame_time': min(frame_times),
                'last_frame_time': max(frame_times),
                'duration': max(frame_times) - min(frame_times),
                'final_family': best_family,
                'final_genus': best_genus,
                'final_species': best_species,
                'family_confidence': float(prob_avgs[0][best_family_idx]),
                'genus_confidence': float(prob_avgs[1][best_genus_idx]),
                'species_confidence': float(prob_avgs[2][best_species_idx]),
                **metrics,
            }
            results.append(result)
            
            print(f"  → {best_family} / {best_genus} / {best_species} "
                  f"({result['species_confidence']:.1%})")
        
        return results
    
    def _build_tracks_rows(
        self,
        all_track_info: Dict,
        results: List[Dict],
    ) -> List[Dict]:
        """
        Build per-track rows for the tracks CSV, merging classification
        results (for confirmed tracks) into the full track info.

        Columns are ordered for readability: identification, detection
        stats, topology calculations, per-criterion pass flags, config
        thresholds, and classification output.
        """
        classification_by_id: Dict[str, Dict] = {}
        for r in results:
            tid = r.get('track_id')
            if tid is not None:
                classification_by_id[tid] = {
                    'final_family': r.get('final_family'),
                    'final_genus': r.get('final_genus'),
                    'final_species': r.get('final_species'),
                    'family_confidence': r.get('family_confidence'),
                    'genus_confidence': r.get('genus_confidence'),
                    'species_confidence': r.get('species_confidence'),
                }

        column_order = [
            # Identification
            'track_id', 'status', 'passes_topology',
            # Detection stats
            'num_detections', 'num_path_points',
            'first_frame', 'last_frame',
            'first_frame_time', 'last_frame_time', 'duration',
            'mean_area', 'min_area_observed', 'max_area_observed',
            # Topology calculations (raw values — lengths in pixels)
            'net_displacement', 'total_path_length',
            'revisit_ratio', 'progression_ratio', 'directional_variance',
            # Per-criterion pass flags (which config check passed/failed)
            'path_points_pass', 'displacement_pass', 'revisit_pass',
            'progression_pass', 'directional_variance_pass',
            # Resolved pixel thresholds actually used for each check
            'min_path_points', 'min_displacement',
            'max_revisit_ratio', 'min_progression_ratio',
            'max_directional_variance', 'revisit_radius',
            # Originating fraction-of-image values from the config file
            'min_displacement_frac', 'revisit_radius_frac',
            # Classification (only set for confirmed+classified tracks)
            'final_family', 'final_genus', 'final_species',
            'family_confidence', 'genus_confidence', 'species_confidence',
        ]

        rows = []
        for track_id, info in all_track_info.items():
            row = {k: info.get(k) for k in column_order if k in info}
            cls = classification_by_id.get(track_id, {})
            for k in ('final_family', 'final_genus', 'final_species',
                      'family_confidence', 'genus_confidence', 'species_confidence'):
                row[k] = cls.get(k, float('nan'))
            rows.append({k: row.get(k) for k in column_order})
        return rows

    def save_results(
        self,
        results: List[Dict],
        output_paths: Dict,
        all_track_info: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Save results to CSV and print summary.

        Args:
            results: Aggregated results list (confirmed tracks only)
            output_paths: Dict with output file paths
            all_track_info: Dict of {track_id: info} for ALL tracks (confirmed
                and unconfirmed). When provided, and ``tracks_csv`` is in
                ``output_paths``, a tracks.csv is written containing every
                track with its full topology calculations and pass/fail flags
                per config criterion.

        Returns:
            pd.DataFrame: Results dataframe (confirmed tracks)
        """
        total_tracks = len(self.track_paths)
        num_confirmed = len(results)
        num_unconfirmed = total_tracks - num_confirmed

        if results:
            df = pd.DataFrame(results).sort_values('num_detections', ascending=False)
            df.to_csv(output_paths["results_csv"], index=False)
            print(f"\n📊 Confirmed results saved: {output_paths['results_csv']} ({num_confirmed} tracks)")
        else:
            df = pd.DataFrame(columns=[
                'track_id', 'num_detections', 'first_frame_time', 'last_frame_time',
                'duration', 'final_family', 'final_genus', 'final_species',
                'family_confidence', 'genus_confidence', 'species_confidence',
                'passes_topology', 'net_displacement', 'revisit_ratio',
                'progression_ratio', 'directional_variance',
                'min_displacement_frac', 'revisit_radius_frac',
            ])
            df.to_csv(output_paths["results_csv"], index=False)
            print(f"\n📊 Results file created (empty): {output_paths['results_csv']}")

        det_df = pd.DataFrame(self.all_detections)
        det_df.to_csv(output_paths["detections_csv"], index=False)
        print(f"📋 Frame-by-frame detections saved: {output_paths['detections_csv']}")

        if all_track_info is not None and "tracks_csv" in output_paths:
            tracks_rows = self._build_tracks_rows(all_track_info, results)
            tracks_df = pd.DataFrame(tracks_rows)
            if not tracks_df.empty:
                if 'status' in tracks_df.columns and 'num_detections' in tracks_df.columns:
                    tracks_df = tracks_df.sort_values(
                        by=['status', 'num_detections'],
                        ascending=[True, False],
                    )
            tracks_df.to_csv(output_paths["tracks_csv"], index=False)
            print(f"🧵 All tracks (confirmed + unconfirmed) saved: {output_paths['tracks_csv']} "
                  f"({len(tracks_rows)} tracks)")
        
        # Summary
        print("\n" + "="*60)
        print("🐛 FINAL SUMMARY")
        print("="*60)
        
        if results:
            print(f"\n✓ CONFIRMED INSECTS ({num_confirmed}):")
            for r in results:
                print(f"  • {r['final_species']} - {r['num_detections']} detections, "
                      f"{r['duration']:.1f}s, {r['species_confidence']:.1%}")
        
        if num_unconfirmed > 0:
            print(f"\n? Unconfirmed tracks: {num_unconfirmed} (failed topology analysis)")
        
        print(f"\n📈 Total: {total_tracks} tracks ({num_confirmed} confirmed, {num_unconfirmed} unconfirmed)")
        
        if not results:
            print("\n" + "!"*60)
            print("⚠️  WARNING: NO CONFIRMED INSECT TRACKS DETECTED!")
            print("!"*60)
            print("Possible reasons:")
            print("  • No insects present in the video")
            print("  • Detection parameters too strict (try lowering min_area)")
            print("  • Tracking parameters too strict (try increasing max_lost_frames)")
            print("  • Path topology too strict (try lowering min_displacement)")
            print("  • Video quality/resolution issues")
            if num_unconfirmed > 0:
                print(f"\nNote: {num_unconfirmed} tracks were detected but failed topology check.")
            print("!"*60)
        
        print("="*60)
        return df


# =============================================================================
# VIDEO PROCESSING PIPELINE
# =============================================================================

def process_video(video_path: str, processor: VideoInferenceProcessor,
                  output_paths: Dict, show_video: bool = False,
                  fps: Optional[float] = None, crops_dir: Optional[str] = None) -> List[Dict]:
    """
    Process video file with efficient classification (confirmed tracks only).
    
    Pipeline:
        1. Detection & Tracking: Process all frames, detect motion, build tracks
        2. Topology Analysis: Determine which tracks are confirmed insects
        3. Classification: Classify ONLY confirmed tracks (saves compute)
        4. Render Videos: Debug (all detections) + Annotated (confirmed with classifications)
    
    Args:
        video_path: Input video path
        processor: VideoInferenceProcessor instance
        output_paths: Dict with output file paths
        show_video: Display video while processing
        fps: Target FPS (skip frames if lower than input)
        crops_dir: Optional directory to save cropped frames
        
    Returns:
        list: Aggregated results
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open: {video_path}")
    
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {video_path}")
    print(f"Properties: {total_frames} frames, {input_fps:.1f} FPS, {total_frames/input_fps:.1f}s")

    # Resolve fraction-based config to absolute pixels now that we know
    # the frame size. After this, processor.params holds pixel values
    # plus `{key}_frac` companions for record/diagnostics.
    processor.setup(width, height)

    # Setup tracker (using resolved pixel values)
    max_lost_frames = processor.params.get("max_lost_frames", 45)

    tracker = InsectTracker(
        image_height=height,
        image_width=width,
        max_lost_frames=max_lost_frames,
        max_frame_jump=processor.params.get("max_frame_jump"),
        w_dist=processor.params.get("tracker_w_dist", 0.6),
        w_area=processor.params.get("tracker_w_area", 0.4),
        cost_threshold=processor.params.get("tracker_cost_threshold", 0.3),
    )
    
    # Frame skip
    skip_interval = max(1, int(input_fps / fps)) if fps and fps > 0 else 1
    if skip_interval > 1:
        print(f"Processing every {skip_interval} frame(s)")
    
    # =========================================================================
    # PHASE 1: Detection & Tracking
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: DETECTION & TRACKING")
    print("="*60)
    
    frame_num = 0
    processed = 0
    start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_num / input_fps if input_fps > 0 else 0
        
        if frame_num % skip_interval == 0:
            processor.process_frame(frame, frame_time, tracker, frame_num)
            processed += 1
            
            if processed % 50 == 0:
                print(f"  Progress: {processed} frames, {len(processor.all_detections)} detections", end='\r')
        
        frame_num += 1
    
    cap.release()
    elapsed = time.time() - start
    print(f"\n✓ Phase 1 complete: {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} FPS)")
    print(f"  Total detections: {len(processor.all_detections)}")
    print(f"  Unique tracks: {len(processor.track_paths)}")
    
    # =========================================================================
    # PHASE 2: Topology Analysis
    # =========================================================================
    confirmed_track_ids, all_track_info = processor.analyze_tracks()
    
    # =========================================================================
    # PHASE 3: Classification (or detection-only)
    # =========================================================================
    if processor.classify:
        print("\n" + "="*60)
        print("PHASE 3: CLASSIFICATION (Confirmed Tracks Only)")
        print("="*60)
        
        if confirmed_track_ids:
            processor.classify_confirmed_tracks(video_path, confirmed_track_ids, crops_dir=crops_dir)
            results = processor.hierarchical_aggregation(confirmed_track_ids)
        else:
            results = []
    else:
        if confirmed_track_ids:
            results = processor.detection_only_results(confirmed_track_ids)
            if crops_dir:
                processor.save_crops(video_path, confirmed_track_ids, crops_dir)
        else:
            results = []
    
    # =========================================================================
    # PHASE 4: Render Videos & Track Composites
    # =========================================================================
    has_video = "annotated_video" in output_paths or "debug_video" in output_paths
    has_composites = "track_composites_dir" in output_paths
    
    if has_video or has_composites:
        print("\n" + "="*60)
        print("PHASE 4: RENDERING OUTPUT")
        print("="*60)
        
        if "debug_video" in output_paths:
            print(f"\nRendering debug video (all detections)...")
            _render_debug_video(
                video_path, output_paths["debug_video"],
                processor, confirmed_track_ids, all_track_info, input_fps
            )
        
        if "annotated_video" in output_paths:
            print(f"\nRendering annotated video ({len(confirmed_track_ids)} confirmed tracks)...")
            _render_annotated_video(
                video_path, output_paths["annotated_video"],
                processor, confirmed_track_ids, input_fps
            )
        
        if has_composites:
            print(f"\nRendering track composite images...")
            _render_track_composites(
                video_path, output_paths["track_composites_dir"],
                processor, confirmed_track_ids
            )
    else:
        print("\n(Video rendering skipped)")
    
    processor.save_results(results, output_paths, all_track_info=all_track_info)
    return results


# =============================================================================
# VIDEO RENDERING
# =============================================================================

def _render_debug_video(video_path: str, output_path: str,
                        processor: VideoInferenceProcessor, confirmed_track_ids: Set[str],
                        all_track_info: Dict, fps: float) -> None:
    """Render debug video showing all detections with confirmed/unconfirmed status."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=processor.params.get("gmm_history", 500),
        varThreshold=processor.params.get("gmm_var_threshold", 16),
        detectShadows=False
    )
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))
    
    frame_detections = defaultdict(list)
    confirmed_track_points = defaultdict(list)
    for det in processor.all_detections:
        frame_detections[det['frame_number']].append(det)
        if det['track_id'] in confirmed_track_ids:
            bbox = det['bbox']
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            confirmed_track_points[det['track_id']].append((det['frame_number'], cx, cy))
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = back_sub.apply(frame)
        fg_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        for track_id, points in confirmed_track_points.items():
            path_to_draw = [(cx, cy) for fn, cx, cy in points if fn <= frame_num]
            if len(path_to_draw) > 1:
                FrameVisualizer.draw_path(frame, path_to_draw, track_id)
        
        for det in frame_detections[frame_num]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            track_id = det['track_id']
            
            if track_id in confirmed_track_ids:
                color = (0, 255, 0)
                label = f"{str(track_id)[:6]} ✓"
                if det.get('species'):
                    label += f" {det['species'][:12]}"
            else:
                color = (0, 255, 255)
                label = f"{str(track_id)[:6] if track_id else 'NEW'}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.rectangle(fg_display, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(frame, f"Frame {frame_num} | Detections (Green=Confirmed, Yellow=Tracking)", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(fg_display, "GMM Motion Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        combined = np.hstack((frame, fg_display))
        out.write(combined)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Debug: {frame_num} frames", end='\r')
    
    cap.release()
    out.release()
    print(f"\n✓ Debug video saved: {output_path}")


def _render_annotated_video(video_path: str, output_path: str,
                            processor: VideoInferenceProcessor,
                            confirmed_track_ids: Set[str], fps: float) -> None:
    """Render annotated video showing only confirmed tracks with classifications."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    if not confirmed_track_ids:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "No confirmed insect tracks", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)
            frame_num += 1
        cap.release()
        out.release()
        print(f"✓ Annotated video saved (no confirmed tracks): {output_path}")
        return
    
    frame_detections = defaultdict(list)
    for det in processor.all_detections:
        if det['track_id'] in confirmed_track_ids:
            frame_detections[det['frame_number']].append(det)
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        for track_id in confirmed_track_ids:
            path_to_draw = []
            for det in processor.all_detections:
                if det['track_id'] == track_id and det['frame_number'] <= frame_num:
                    bbox = det['bbox']
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    path_to_draw.append((cx, cy))
            
            if len(path_to_draw) > 1:
                FrameVisualizer.draw_path(frame, path_to_draw, track_id)
        
        for det in frame_detections[frame_num]:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            
            classification = {
                'family': det.get('family', ''),
                'genus': det.get('genus', ''),
                'species': det.get('species', ''),
                'family_confidence': det.get('family_confidence', 0),
                'genus_confidence': det.get('genus_confidence', 0),
                'species_confidence': det.get('species_confidence', 0),
            }
            FrameVisualizer.draw_detection(frame, x1, y1, x2, y2, track_id, classification)
        
        cv2.putText(frame, f"Confirmed Insects ({len(confirmed_track_ids)} tracks)", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(frame)
        frame_num += 1
        
        if frame_num % 100 == 0:
            print(f"  Annotated: {frame_num} frames", end='\r')
    
    cap.release()
    out.release()
    print(f"\n✓ Annotated video saved: {output_path}")


# =============================================================================
# TRACK COMPOSITE IMAGES (delegates to bugspot)
# =============================================================================

def _render_track_composites(video_path: str, output_dir: str,
                             processor: VideoInferenceProcessor,
                             confirmed_track_ids: Set[str]) -> None:
    """Render composite images using bugspot pipeline."""
    from bugspot.pipeline import DetectionPipeline
    
    if not confirmed_track_ids:
        print("No confirmed tracks for composite rendering.")
        return
    
    # Create a temporary pipeline just for rendering composites
    # (reuse processor's detection data and track paths)
    tmp = DetectionPipeline.__new__(DetectionPipeline)
    tmp.all_detections = processor.all_detections
    tmp.track_paths = processor.track_paths
    tmp.config = processor.params
    
    composites = tmp._render_composites(video_path, confirmed_track_ids, save_dir=output_dir)
    print(f"✓ Saved {len(composites)} track composite images to: {output_dir}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def inference(
    video_path: str,
    output_dir: str,
    hierarchical_model_path: Optional[str] = None,
    species_list: Optional[List[str]] = None,
    fps: Optional[float] = None,
    config: Optional[Dict] = None,
    backbone: str = "resnet50",
    crops: bool = False,
    save_video: bool = True,
    img_size: int = 60,
    classify: bool = True,
    track_composites: bool = False,
) -> Dict:
    """
    Run inference on a video file.
    
    Args:
        video_path: Input video path
        output_dir: Output directory for all generated files
        hierarchical_model_path: Path to trained model weights (required if classify=True)
        species_list: Optional list of species names (if None, loaded from checkpoint)
        fps: Target processing FPS (None = use input FPS)
        config: Detection config - can be:
            - None: use defaults
            - str: path to YAML/JSON config file
            - dict: config parameters directly
        backbone: ResNet backbone ('resnet18', 'resnet50', 'resnet101')
        crops: If True, save cropped frames for each classified track
        save_video: If True, save annotated and debug videos. Defaults to True.
        img_size: Image size for classification (should match training). Default: 60.
        classify: If True, classify confirmed tracks. If False, detection only (NaN for classification).
        track_composites: If True, save composite images showing each track's movement over time.
    
    Returns:
        dict: Processing results with output file paths
        
    Generated files in output_dir:
        - {video_name}_annotated.mp4: Video with detection boxes and paths (if save_video=True)
        - {video_name}_debug.mp4: Side-by-side with GMM motion mask (if save_video=True)
        - {video_name}_results.csv: Aggregated track results (confirmed tracks only)
        - {video_name}_detections.csv: Frame-by-frame detections
        - {video_name}_tracks.csv: Every track (confirmed + unconfirmed) with
            full topology calculations, per-criterion pass flags, and the
            config thresholds used for each check. Useful for diagnosing
            why tracks were rejected.
        - {video_name}_crops/ (if crops=True): Directory with cropped frames per track
        - {video_name}_composites/ (if track_composites=True): Composite images per track
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return {"error": f"Video not found: {video_path}", "success": False}
    
    if classify and hierarchical_model_path is None:
        return {"error": "hierarchical_model_path is required when classify=True", "success": False}
    
    # Build parameters from config
    if config is None:
        params = get_default_config()
    elif isinstance(config, str):
        params = load_config(config)
    elif isinstance(config, dict):
        params = get_default_config()
        for key, value in config.items():
            if key in params:
                params[key] = value
            else:
                logger.warning(f"Unknown config parameter: {key}")
    else:
        raise ValueError("config must be None, a file path (str), or a dict")
    
    # Setup output paths
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_paths = {
        "results_csv": os.path.join(output_dir, f"{video_name}_results.csv"),
        "detections_csv": os.path.join(output_dir, f"{video_name}_detections.csv"),
        "tracks_csv": os.path.join(output_dir, f"{video_name}_tracks.csv"),
    }
    
    if save_video:
        output_paths["annotated_video"] = os.path.join(output_dir, f"{video_name}_annotated.mp4")
        output_paths["debug_video"] = os.path.join(output_dir, f"{video_name}_debug.mp4")
    
    crops_dir = os.path.join(output_dir, f"{video_name}_crops") if crops else None
    if crops_dir:
        output_paths["crops_dir"] = crops_dir
    
    if track_composites:
        output_paths["track_composites_dir"] = os.path.join(output_dir, f"{video_name}_composites")
    
    print("\n" + "="*60)
    print("BPLUSPLUS INFERENCE")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Mode: {'Detection + Classification' if classify else 'Detection only'}")
    if classify:
        print(f"Model: {hierarchical_model_path}")
    print(f"Output directory: {output_dir}")
    print("\nOutput files:")
    for name, path in output_paths.items():
        print(f"  {name}: {os.path.basename(path)}")
    print("\nDetection Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    processor = VideoInferenceProcessor(
        params=params,
        hierarchical_model_path=hierarchical_model_path,
        species_list=species_list,
        backbone=backbone,
        img_size=img_size,
        classify=classify,
    )
    
    try:
        results = process_video(
            video_path=video_path,
            processor=processor,
            output_paths=output_paths,
            fps=fps,
            crops_dir=crops_dir
        )
        
        return {
            "video_file": os.path.basename(video_path),
            "output_dir": output_dir,
            "output_files": output_paths,
            "success": True,
            "detections": len(processor.all_detections),
            "tracks": len(results),
            "confirmed_tracks": len([r for r in results if r.get('passes_topology', False)]),
        }
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": str(e), "success": False}


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command line interface for inference."""
    parser = argparse.ArgumentParser(
        description='Bplusplus Video Inference - Detect and classify insects in videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (species list loaded from checkpoint)
  python -m bplusplus.inference --video input.mp4 --model model.pt \\
      --output-dir results/
  
  # Override species list from checkpoint
  python -m bplusplus.inference --video input.mp4 --model model.pt \\
      --output-dir results/ --species "Apis mellifera" "Bombus terrestris"
  
  # With config file
  python -m bplusplus.inference --video input.mp4 --model model.pt \\
      --output-dir results/ --config detection_config.yaml

Output files generated in output directory:
  - {video_name}_annotated.mp4: Video with detection boxes and paths
  - {video_name}_debug.mp4: Side-by-side view with GMM motion mask  
  - {video_name}_results.csv: Aggregated track results (confirmed tracks only)
  - {video_name}_detections.csv: Frame-by-frame detections
  - {video_name}_tracks.csv: All tracks (confirmed + unconfirmed) with
      full topology calculations and per-criterion pass flags
  - {video_name}_crops/ (with --crops): Cropped frames for each track
        """
    )
    
    # Required arguments
    parser.add_argument('--video', '-v', required=True, help='Input video path')
    parser.add_argument('--model', '-m', help='Path to hierarchical model weights (required unless --no-classify)')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for all generated files')
    parser.add_argument('--species', '-s', nargs='+', help='List of species names (optional, loaded from checkpoint if not provided)')
    
    # Config
    parser.add_argument('--config', '-c', help='Path to config file (YAML or JSON)')
    
    # Processing
    parser.add_argument('--fps', type=float, help='Target processing FPS')
    parser.add_argument('--show', action='store_true', help='Display video while processing')
    parser.add_argument('--backbone', '-b', default='resnet50',
                       choices=['resnet18', 'resnet50', 'resnet101'],
                       help='ResNet backbone (default: resnet50)')
    parser.add_argument('--crops', action='store_true',
                       help='Save cropped frames for each classified track')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip saving annotated and debug videos')
    parser.add_argument('--img-size', type=int, default=60,
                       help='Image size for classification (default: 60)')
    parser.add_argument('--no-classify', action='store_true',
                       help='Detection only - skip classification (NaN for species)')
    parser.add_argument('--track-composites', action='store_true',
                       help='Save composite images showing each track over time')
    parser.add_argument('--detection-resolution', type=int, nargs=2, metavar=('W', 'H'),
                       help='Run detection at this WxH resolution for speed; '
                            'bounding boxes are scaled back to native res so '
                            'crops/composites stay full resolution')
    
    # Detection parameters
    defaults = DEFAULT_DETECTION_CONFIG
    
    cohesive = parser.add_argument_group('Cohesiveness parameters')
    cohesive.add_argument('--min-blob-ratio', type=float, 
                         help=f'Min largest blob ratio (default: {defaults["min_largest_blob_ratio"]})')
    cohesive.add_argument('--max-num-blobs', type=int,
                         help=f'Max number of blobs (default: {defaults["max_num_blobs"]})')
    
    shape = parser.add_argument_group('Shape parameters')
    shape.add_argument('--min-area', type=float,
                      help=f'Min contour area as fraction of image area (default: {defaults["min_area"]})')
    shape.add_argument('--max-area', type=float,
                      help=f'Max contour area as fraction of image area (default: {defaults["max_area"]})')
    shape.add_argument('--min-density', type=float,
                      help=f'Min density (default: {defaults["min_density"]})')
    shape.add_argument('--min-solidity', type=float,
                      help=f'Min solidity (default: {defaults["min_solidity"]})')

    tracking = parser.add_argument_group('Tracking parameters')
    tracking.add_argument('--min-displacement', type=float,
                         help=f'Min NET displacement as fraction of image width (default: {defaults["min_displacement"]})')
    tracking.add_argument('--min-path-points', type=int,
                         help=f'Min path points (default: {defaults["min_path_points"]})')
    tracking.add_argument('--max-frame-jump', type=float,
                         help=f'Max per-frame jump as fraction of image width (default: {defaults["max_frame_jump"]})')
    tracking.add_argument('--max-lost-frames', type=int,
                         help=f'Frames before lost track deleted (default: {defaults["max_lost_frames"]})')
    
    topology = parser.add_argument_group('Path topology parameters')
    topology.add_argument('--max-revisit-ratio', type=float,
                         help=f'Max revisit ratio (default: {defaults["max_revisit_ratio"]})')
    topology.add_argument('--min-progression-ratio', type=float,
                         help=f'Min progression ratio (default: {defaults["min_progression_ratio"]})')
    topology.add_argument('--max-directional-variance', type=float,
                         help=f'Max directional variance (default: {defaults["max_directional_variance"]})')
    
    args = parser.parse_args()
    
    # Build config
    if args.config:
        config = args.config
    else:
        cli_overrides = {
            "min_largest_blob_ratio": args.min_blob_ratio,
            "max_num_blobs": args.max_num_blobs,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "min_density": args.min_density,
            "min_solidity": args.min_solidity,
            "min_displacement": args.min_displacement,
            "min_path_points": args.min_path_points,
            "max_frame_jump": args.max_frame_jump,
            "max_lost_frames": args.max_lost_frames,
            "max_revisit_ratio": args.max_revisit_ratio,
            "min_progression_ratio": args.min_progression_ratio,
            "max_directional_variance": args.max_directional_variance,
            "detection_resolution": args.detection_resolution,
        }
        config = {k: v for k, v in cli_overrides.items() if v is not None} or None
    
    classify = not args.no_classify
    
    result = inference(
        video_path=args.video,
        output_dir=args.output_dir,
        hierarchical_model_path=args.model if classify else None,
        species_list=args.species,
        fps=args.fps,
        config=config,
        backbone=args.backbone,
        crops=args.crops,
        save_video=not args.no_video,
        img_size=args.img_size,
        classify=classify,
        track_composites=args.track_composites,
    )
    
    if result.get("success"):
        print(f"\n✓ Inference complete!")
        print(f"  Output directory: {result['output_dir']}")
        print(f"  Detections: {result['detections']}")
        print(f"  Tracks: {result['tracks']} ({result['confirmed_tracks']} confirmed)")
    else:
        print(f"\n✗ Inference failed: {result.get('error')}")


if __name__ == "__main__":
    main()
