"""
Insect Detection Backend Module
===============================

This module provides motion-based insect detection utilities used by the inference pipeline.
It is NOT meant to be run directly - use inference.py instead.

Exports:
    - DEFAULT_DETECTION_CONFIG: Default parameters for detection
    - build_detection_params(): Build detection params dict
    - extract_motion_detections(): Extract detections from a frame
    - Path topology functions for track analysis
"""

import cv2
import numpy as np
from collections import defaultdict

# Support both standalone and package import
try:
    from .tracker import InsectTracker
except ImportError:
    from tracker import InsectTracker


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_DETECTION_CONFIG = {
    # Cohesiveness parameters
    "min_largest_blob_ratio": 0.80,  # Min ratio of largest blob to total motion
    "max_num_blobs": 5,              # Max blobs in detection region
    
    # Shape parameters
    "min_area": 200,                 # Min contour area (px²)
    "max_area": 40000,               # Max contour area (px²)
    "min_density": 3.0,              # Min area/perimeter ratio
    "min_solidity": 0.55,            # Min convex hull fill ratio
    
    # Tracking parameters
    "min_displacement": 50,          # Min NET displacement for confirmation (px)
    "min_path_points": 10,           # Min path points for analysis
    "max_frame_jump": 100,           # Max pixels between frames
    "lost_track_seconds": 1.5,       # Track memory duration (seconds)
    
    # Path topology parameters
    "max_revisit_ratio": 0.30,       # Max revisit ratio (explores new areas)
    "min_progression_ratio": 0.70,   # Min progression ratio (moves forward)
    "max_directional_variance": 0.90, # Max directional variance (consistent heading)
}


def get_default_config():
    """Return a copy of the default detection configuration."""
    return DEFAULT_DETECTION_CONFIG.copy()


def build_detection_params(**kwargs):
    """
    Build detection parameters dict from defaults + overrides.
    
    Args:
        **kwargs: Parameter overrides (any key from DEFAULT_DETECTION_CONFIG)
        
    Returns:
        dict: Complete detection parameters
    """
    params = get_default_config()
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            raise ValueError(f"Unknown detection parameter: {key}")
    return params


# ============================================================================
# COHESIVENESS ANALYSIS
# ============================================================================

def is_cohesive_blob(fg_mask_region, bbox_area, min_largest_blob_ratio=0.80, max_num_blobs=10):
    """
    Check if motion in a region is cohesive (insect) vs scattered (plant).
    
    Args:
        fg_mask_region: Foreground mask of the detection region
        bbox_area: Area of bounding box
        min_largest_blob_ratio: Min ratio of largest blob to total motion
        max_num_blobs: Max number of blobs allowed
        
    Returns:
        tuple: (is_cohesive, metrics_dict or None)
    """
    motion_pixels = np.count_nonzero(fg_mask_region)
    
    if motion_pixels == 0:
        return False, None
    
    motion_ratio = motion_pixels / bbox_area
    if motion_ratio < 0.15:
        return False, None
    
    contours, _ = cv2.findContours(fg_mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False, None
    
    num_blobs = len(contours)
    if num_blobs > max_num_blobs:
        return False, None
    
    largest_blob = max(contours, key=cv2.contourArea)
    largest_blob_area = cv2.contourArea(largest_blob)
    largest_blob_ratio = largest_blob_area / motion_pixels
    
    if largest_blob_ratio < min_largest_blob_ratio:
        return False, None
    
    return True, {
        'motion_ratio': motion_ratio,
        'num_blobs': num_blobs,
        'largest_blob_ratio': largest_blob_ratio
    }


def passes_shape_filters(contour, min_area=200, max_area=40000, min_density=3.0, min_solidity=0.55):
    """
    Check if contour passes size and shape filters.
    
    Args:
        contour: OpenCV contour
        min_area: Minimum area in pixels²
        max_area: Maximum area in pixels²
        min_density: Minimum area/perimeter ratio
        min_solidity: Minimum convex hull fill ratio
        
    Returns:
        bool: True if contour passes all filters
    """
    area = cv2.contourArea(contour)
    
    if area < min_area or area > max_area:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    density = area / perimeter
    if density < min_density:
        return False
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    if solidity < min_solidity:
        return False
    
    return True


# ============================================================================
# MOTION DETECTION
# ============================================================================

def extract_motion_detections(frame, back_sub, morph_kernel, params):
    """
    Extract motion-based detections from a frame.
    
    Args:
        frame: BGR image frame
        back_sub: cv2.BackgroundSubtractor instance
        morph_kernel: Morphological kernel for noise removal
        params: Detection parameters dict
        
    Returns:
        tuple: (detections_list, foreground_mask)
            - detections_list: List of [x1, y1, x2, y2] bounding boxes
            - foreground_mask: Binary foreground mask
    """
    fg_mask = back_sub.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_detections = []
    height, width = frame.shape[:2]

    for contour in contours:
        if not passes_shape_filters(
            contour,
            params["min_area"],
            params["max_area"],
            params["min_density"],
            params["min_solidity"],
        ):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        region = fg_mask[y:y + h, x:x + w]
        is_cohesive, _ = is_cohesive_blob(
            region,
            w * h,
            params["min_largest_blob_ratio"],
            params["max_num_blobs"],
        )
        if not is_cohesive:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        valid_detections.append([x1, y1, x2, y2])

    return valid_detections, fg_mask


# ============================================================================
# PATH TOPOLOGY ANALYSIS
# ============================================================================

def calculate_revisit_ratio(path, revisit_radius=50):
    """
    Calculate how much the path revisits previous locations.
    Low ratio = exploring new areas (insect), High ratio = oscillating (plant).
    """
    revisit_count = 0
    for i in range(len(path)):
        for j in range(i):
            if np.linalg.norm(path[i] - path[j]) < revisit_radius:
                revisit_count += 1
    
    max_revisits = len(path) * (len(path) - 1) / 2
    return revisit_count / (max_revisits + 1e-6)


def calculate_progression_ratio(path):
    """
    Calculate if path progressively explores outward.
    High ratio = linear progression (insect), Low = backtracking (plant).
    """
    if len(path) < 2:
        return 0
    
    start_point = path[0]
    end_point = path[-1]
    net_displacement = np.linalg.norm(end_point - start_point)
    
    progressive_distances = [np.linalg.norm(p - start_point) for p in path]
    max_progressive = max(progressive_distances)
    
    return net_displacement / (max_progressive + 1e-6)


def calculate_directional_variance(path):
    """
    Calculate variance in movement direction.
    Low variance = consistent direction (insect), High = random (plant).
    """
    if len(path) < 2:
        return 1.0
    
    directions = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if dx != 0 or dy != 0:
            angle = np.arctan2(dy, dx)
            directions.append(angle)
    
    if not directions:
        return 1.0
    
    mean_sin = np.mean(np.sin(directions))
    mean_cos = np.mean(np.cos(directions))
    return 1 - np.sqrt(mean_sin**2 + mean_cos**2)


def analyze_path_topology(path, params):
    """
    Analyze path to determine if it exhibits insect-like movement.
    
    Args:
        path: List of (x, y) positions
        params: Detection parameters dict
        
    Returns:
        tuple: (passes_criteria, metrics_dict)
    """
    if len(path) < 3:
        return False, {}
    
    path = np.array(path)
    
    net_displacement = np.linalg.norm(path[-1] - path[0])
    revisit_ratio = calculate_revisit_ratio(path)
    progression_ratio = calculate_progression_ratio(path)
    directional_variance = calculate_directional_variance(path)
    
    metrics = {
        'net_displacement': net_displacement,
        'revisit_ratio': revisit_ratio,
        'progression_ratio': progression_ratio,
        'directional_variance': directional_variance
    }
    
    passes = (
        net_displacement >= params["min_displacement"] and
        revisit_ratio <= params["max_revisit_ratio"] and
        progression_ratio >= params["min_progression_ratio"] and
        directional_variance <= params["max_directional_variance"]
    )
    
    return passes, metrics


# ============================================================================
# TRACK MANAGEMENT
# ============================================================================

def check_track_consistency(prev_pos, curr_pos, prev_area, curr_area, max_frame_jump):
    """
    Check if track update is consistent (not a bad match).
    
    Args:
        prev_pos: Previous (x, y) position
        curr_pos: Current (x, y) position
        prev_area: Previous bounding box area
        curr_area: Current bounding box area
        max_frame_jump: Maximum allowed position jump
        
    Returns:
        bool: True if consistent, False if likely bad match
    """
    # Position jump check
    frame_jump = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
    if frame_jump > max_frame_jump:
        return False
    
    # Size change check (3x threshold)
    area_ratio = max(curr_area, prev_area) / (min(curr_area, prev_area) + 1e-6)
    if area_ratio > 3.0:
        return False
    
    return True


def create_tracker(height, width, params):
    """
    Create an InsectTracker with parameters from config.
    
    Args:
        height: Frame height
        width: Frame width
        params: Detection parameters dict
    
    Returns:
        InsectTracker: Configured tracker instance
    """
    # Note: lost_track_seconds needs FPS to convert to frames
    # This is handled by the caller who knows the FPS
    return InsectTracker(
        image_height=height,
        image_width=width,
        max_frames=30,  # Will be overridden by caller with FPS info
        track_memory_frames=30,
        w_dist=0.6,
        w_area=0.4,
        cost_threshold=0.3,
        debug=False
    )
