"""
Insect tracking module.

Uses Hungarian algorithm for optimal assignment of detections to tracks.
Handles track persistence across frames with lost track recovery.

Used by inference.py - not meant to be run directly.
"""

import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from scipy.optimize import linear_sum_assignment


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box with tracking metadata."""
    x: float
    y: float
    width: float
    height: float
    frame_id: int
    track_id: Optional[str] = None
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float,
                  frame_id: int, track_id: Optional[str] = None) -> "BoundingBox":
        """Create BoundingBox from x1,y1,x2,y2 coordinates."""
        return cls(x1, y1, x2 - x1, y2 - y1, frame_id, track_id)


@dataclass
class Track:
    """Track with history of detections."""
    track_id: str
    first_frame: int = 0
    last_frame: int = 0
    is_active: bool = True
    detections: List[Dict] = field(default_factory=list)


# =============================================================================
# TRACKER CLASS
# =============================================================================

class InsectTracker:
    """
    Insect tracker using Hungarian algorithm for optimal assignment.
    
    Features:
        - Lost track recovery within memory window
        - Weighted distance + area cost function
        - Configurable matching threshold
    """
    
    def __init__(
        self,
        image_height: int,
        image_width: int,
        max_frames: int = 30,
        w_dist: float = 0.7,
        w_area: float = 0.3,
        cost_threshold: float = 0.8,
        track_memory_frames: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Initialize the tracker.
        
        Args:
            image_height: Frame height for normalization
            image_width: Frame width for normalization
            max_frames: Maximum history frames to maintain
            w_dist: Weight for distance cost (0-1)
            w_area: Weight for area cost (0-1)
            cost_threshold: Maximum cost for valid match
            track_memory_frames: Frames to keep lost tracks (default: max_frames)
            debug: Enable debug logging
        """
        self.image_height = image_height
        self.image_width = image_width
        self.max_dist = np.sqrt(image_height**2 + image_width**2)
        self.max_frames = max_frames
        self.w_dist = w_dist
        self.w_area = w_area
        self.cost_threshold = cost_threshold
        self.debug = debug
        
        self.track_memory_frames = track_memory_frames if track_memory_frames is not None else max_frames
        
        if self.debug:
            print(f"Tracker: {image_width}x{image_height}, "
                  f"max_frames={max_frames}, threshold={cost_threshold}")
        
        # State
        self.tracking_history: deque = deque(maxlen=max_frames)
        self.current_tracks: List[BoundingBox] = []
        self.lost_tracks: Dict[str, Dict] = {}  # track_id -> {box, frames_lost}
    
    def update(self, new_detections: List, frame_id: int) -> List[Optional[str]]:
        """
        Update tracking with new detections.
        
        Args:
            new_detections: List of detection boxes (x1, y1, x2, y2)
            frame_id: Current frame number
            
        Returns:
            List of track IDs corresponding to each detection
        """
        # No detections: move all to lost
        if not new_detections:
            self._move_all_to_lost()
            self._age_lost_tracks()
            self.tracking_history.append([])
            return []
        
        # Convert to BoundingBox objects
        new_boxes = [
            BoundingBox.from_xyxy(*det[:4], frame_id)
            for det in new_detections
        ]
        
        # First frame: assign new IDs
        if not self.current_tracks and not self.lost_tracks:
            track_ids = self._assign_new_ids(new_boxes)
            self.current_tracks = new_boxes
            self.tracking_history.append(new_boxes)
            return track_ids
        
        # Combine current + lost for matching
        all_previous = self.current_tracks.copy()
        for track_id, info in self.lost_tracks.items():
            box = info['box']
            box.track_id = track_id
            all_previous.append(box)
        
        if not all_previous:
            track_ids = self._assign_new_ids(new_boxes)
            self.current_tracks = new_boxes
            self.tracking_history.append(new_boxes)
            return track_ids
        
        # Hungarian assignment
        cost_matrix, n_prev, n_curr = self._build_cost_matrix(all_previous, new_boxes)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        # Process assignments
        track_ids: List[Optional[str]] = [None] * len(new_boxes)
        assigned = set()
        recovered = set()
        
        for i, j in zip(row_idx, col_idx):
            if i < n_prev and j < n_curr:
                cost = cost_matrix[i, j]
                if cost < self.cost_threshold:
                    prev_id = all_previous[i].track_id
                    new_boxes[j].track_id = prev_id
                    track_ids[j] = prev_id
                    assigned.add(j)
                    
                    if prev_id in self.lost_tracks:
                        recovered.add(prev_id)
        
        # Remove recovered from lost
        for tid in recovered:
            del self.lost_tracks[tid]
        
        # Assign new IDs to unmatched detections
        for j in range(n_curr):
            if j not in assigned:
                new_id = self._generate_track_id()
                new_boxes[j].track_id = new_id
                track_ids[j] = new_id
        
        # Move unmatched current tracks to lost
        matched_ids = {track_ids[j] for j in assigned if track_ids[j]}
        for track in self.current_tracks:
            if track.track_id not in matched_ids and track.track_id not in recovered:
                if track.track_id not in self.lost_tracks:
                    self.lost_tracks[track.track_id] = {'box': track, 'frames_lost': 1}
        
        self._age_lost_tracks()
        self.current_tracks = new_boxes
        self.tracking_history.append(new_boxes)
        
        return track_ids
    
    def _build_cost_matrix(self, prev_boxes: List[BoundingBox],
                           curr_boxes: List[BoundingBox]) -> Tuple[np.ndarray, int, int]:
        """Build cost matrix for Hungarian algorithm."""
        n_prev, n_curr = len(prev_boxes), len(curr_boxes)
        n = max(n_prev, n_curr)
        
        cost_matrix = np.ones((n, n)) * 999.0
        
        for i in range(n_prev):
            for j in range(n_curr):
                cost_matrix[i, j] = self._calculate_cost(prev_boxes[i], curr_boxes[j])
        
        return cost_matrix, n_prev, n_curr
    
    def _calculate_cost(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate matching cost between two boxes.
        
        Cost = w_dist * normalized_distance + w_area * area_difference
        """
        cx1, cy1 = box1.center()
        cx2, cy2 = box2.center()
        
        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        norm_dist = dist / self.max_dist
        
        min_area = min(box1.area, box2.area)
        max_area = max(box1.area, box2.area)
        area_cost = min_area / max_area if max_area > 0 else 1.0
        
        return (norm_dist * self.w_dist) + ((1 - area_cost) * self.w_area)
    
    def _generate_track_id(self) -> str:
        """Generate unique track ID."""
        return str(uuid.uuid4())
    
    def _assign_new_ids(self, boxes: List[BoundingBox]) -> List[str]:
        """Assign new track IDs to all boxes."""
        track_ids = []
        for box in boxes:
            new_id = self._generate_track_id()
            box.track_id = new_id
            track_ids.append(new_id)
        return track_ids
    
    def _move_all_to_lost(self) -> None:
        """Move all current tracks to lost."""
        for track in self.current_tracks:
            if track.track_id not in self.lost_tracks:
                self.lost_tracks[track.track_id] = {'box': track, 'frames_lost': 1}
        self.current_tracks = []
    
    def _age_lost_tracks(self) -> None:
        """Age lost tracks and remove old ones."""
        to_remove = []
        for track_id, info in self.lost_tracks.items():
            info['frames_lost'] += 1
            if info['frames_lost'] > self.track_memory_frames:
                to_remove.append(track_id)
        
        for tid in to_remove:
            del self.lost_tracks[tid]
    
    def get_stats(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'active_tracks': len(self.current_tracks),
            'lost_tracks': len(self.lost_tracks),
            'active_track_ids': [t.track_id for t in self.current_tracks],
            'lost_track_ids': list(self.lost_tracks.keys()),
            'total_history_frames': len(self.tracking_history)
        }
