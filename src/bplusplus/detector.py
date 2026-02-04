"""
Motion-based detection module.

Provides motion detection using GMM background subtraction,
with shape and cohesiveness filters to identify insects.

Used by inference.py - not meant to be run directly.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: float
    frame_number: int


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_DETECTION_CONFIG = {
    # GMM Background Subtractor parameters
    "gmm_history": 500,
    "gmm_var_threshold": 16,
    
    # Morphological filtering
    "morph_kernel_size": 3,
    
    # Cohesiveness parameters
    "min_largest_blob_ratio": 0.80,
    "max_num_blobs": 5,
    "min_motion_ratio": 0.15,
    
    # Shape parameters
    "min_area": 200,
    "max_area": 40000,
    "min_density": 3.0,
    "min_solidity": 0.55,
    
    # Tracking parameters
    "min_displacement": 50,
    "min_path_points": 10,
    "max_frame_jump": 100,
    "max_lost_frames": 45,
    "max_area_change_ratio": 3.0,
    
    # Tracker matching parameters
    "tracker_w_dist": 0.6,
    "tracker_w_area": 0.4,
    "tracker_cost_threshold": 0.3,
    
    # Path topology parameters
    "max_revisit_ratio": 0.30,
    "min_progression_ratio": 0.70,
    "max_directional_variance": 0.90,
    "revisit_radius": 50,
}


def get_default_config() -> Dict:
    """Return a copy of the default detection configuration."""
    return DEFAULT_DETECTION_CONFIG.copy()


def build_detection_params(**kwargs) -> Dict:
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


# =============================================================================
# MOTION DETECTOR CLASS
# =============================================================================

class MotionDetector:
    """
    Motion-based detector using GMM background subtraction.
    
    Detects moving objects and filters by shape/cohesiveness
    to identify likely insects vs plants/noise.
    """
    
    def __init__(self, params: Dict):
        """
        Initialize the motion detector.
        
        Args:
            params: Detection parameters dict
        """
        self.params = params
        
        # GMM background subtractor
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=params.get("gmm_history", 500),
            varThreshold=params.get("gmm_var_threshold", 16),
            detectShadows=False
        )
        
        # Morphological kernel for noise removal
        kernel_size = params.get("morph_kernel_size", 3)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    def detect(self, frame: np.ndarray, frame_number: int = 0) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect insects in a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_number: Current frame index
            
        Returns:
            Tuple of (list of Detection objects, foreground mask)
        """
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = frame.shape[:2]
        detections = []
        
        for contour in contours:
            # Shape filtering
            if not passes_shape_filters(
                contour,
                self.params["min_area"],
                self.params["max_area"],
                self.params["min_density"],
                self.params["min_solidity"]
            ):
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Cohesiveness check
            region = fg_mask[y:y+h, x:x+w]
            is_cohesive, _ = is_cohesive_blob(
                region, w * h,
                self.params["min_largest_blob_ratio"],
                self.params["max_num_blobs"],
                self.params.get("min_motion_ratio", 0.15)
            )
            
            if not is_cohesive:
                continue
            
            # Clamp to frame bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                area=cv2.contourArea(contour),
                frame_number=frame_number
            ))
        
        return detections, fg_mask
    
    def reset(self) -> None:
        """Reset the background model."""
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=self.params.get("gmm_history", 500),
            varThreshold=self.params.get("gmm_var_threshold", 16),
            detectShadows=False
        )


# =============================================================================
# SHAPE AND COHESIVENESS FILTERS
# =============================================================================

def is_cohesive_blob(fg_mask_region: np.ndarray, bbox_area: int,
                     min_largest_blob_ratio: float = 0.80,
                     max_num_blobs: int = 5,
                     min_motion_ratio: float = 0.15) -> Tuple[bool, Optional[Dict]]:
    """
    Check if motion in a region is cohesive (insect) vs scattered (plant).
    
    Args:
        fg_mask_region: Foreground mask of the detection region
        bbox_area: Area of bounding box
        min_largest_blob_ratio: Min ratio of largest blob to total motion
        max_num_blobs: Max number of blobs allowed
        min_motion_ratio: Min ratio of motion pixels to bbox area
        
    Returns:
        tuple: (is_cohesive, metrics_dict or None)
    """
    motion_pixels = np.count_nonzero(fg_mask_region)
    
    if motion_pixels == 0:
        return False, None
    
    motion_ratio = motion_pixels / bbox_area
    if motion_ratio < min_motion_ratio:
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


def passes_shape_filters(contour, min_area: int = 200, max_area: int = 40000,
                         min_density: float = 3.0, min_solidity: float = 0.55) -> bool:
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


# =============================================================================
# STANDALONE DETECTION FUNCTION (for backwards compatibility)
# =============================================================================

def extract_motion_detections(frame: np.ndarray, back_sub, morph_kernel,
                              params: Dict) -> Tuple[List[List[int]], np.ndarray]:
    """
    Extract motion-based detections from a frame.
    
    This is a standalone function for backwards compatibility.
    For new code, use MotionDetector class instead.
    
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
            region, w * h,
            params["min_largest_blob_ratio"],
            params["max_num_blobs"],
            params.get("min_motion_ratio", 0.15),
        )
        if not is_cohesive:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        valid_detections.append([x1, y1, x2, y2])

    return valid_detections, fg_mask


# =============================================================================
# PATH TOPOLOGY ANALYSIS
# =============================================================================

def calculate_revisit_ratio(path: np.ndarray, revisit_radius: int = 50) -> float:
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


def calculate_progression_ratio(path: np.ndarray) -> float:
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


def calculate_directional_variance(path: np.ndarray) -> float:
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


def analyze_path_topology(path: List[Tuple[float, float]], params: Dict) -> Tuple[bool, Dict]:
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
    
    path_arr = np.array(path)
    
    net_displacement = float(np.linalg.norm(path_arr[-1] - path_arr[0]))
    revisit_ratio = calculate_revisit_ratio(path_arr, params.get("revisit_radius", 50))
    progression_ratio = calculate_progression_ratio(path_arr)
    directional_variance = calculate_directional_variance(path_arr)
    
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


# =============================================================================
# TRACK CONSISTENCY
# =============================================================================

def check_track_consistency(prev_pos: Tuple[float, float], curr_pos: Tuple[float, float],
                           prev_area: float, curr_area: float, max_frame_jump: int,
                           max_area_change_ratio: float = 3.0) -> bool:
    """
    Check if track update is consistent (not a bad match).
    
    Args:
        prev_pos: Previous (x, y) position
        curr_pos: Current (x, y) position
        prev_area: Previous bounding box area
        curr_area: Current bounding box area
        max_frame_jump: Maximum allowed position jump
        max_area_change_ratio: Maximum allowed area change ratio
        
    Returns:
        bool: True if consistent, False if likely bad match
    """
    # Position jump check
    frame_jump = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
    if frame_jump > max_frame_jump:
        return False
    
    # Size change check
    area_ratio = max(curr_area, prev_area) / (min(curr_area, prev_area) + 1e-6)
    if area_ratio > max_area_change_ratio:
        return False
    
    return True
