import numpy as np
import uuid
from scipy.optimize import linear_sum_assignment
from collections import deque

class BoundingBox:
    def __init__(self, x, y, width, height, frame_id, track_id=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = width * height
        self.frame_id = frame_id
        self.track_id = track_id
    
    def center(self):
        return (self.x + self.width/2, self.y + self.height/2)
    
    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2, frame_id, track_id=None):
        """Create BoundingBox from x1,y1,x2,y2 coordinates"""
        width = x2 - x1
        height = y2 - y1
        return cls(x1, y1, width, height, frame_id, track_id)

class InsectTracker:
    def __init__(self, image_height, image_width, max_frames=30, w_dist=0.7, w_area=0.3, cost_threshold=0.8, track_memory_frames=None, debug=False):
        self.image_height = image_height
        self.image_width = image_width
        self.max_dist = np.sqrt(image_height**2 + image_width**2)
        self.max_frames = max_frames
        self.w_dist = w_dist
        self.w_area = w_area
        self.cost_threshold = cost_threshold
        self.debug = debug
        
        # If track_memory_frames not specified, use max_frames (full history window)
        self.track_memory_frames = track_memory_frames if track_memory_frames is not None else max_frames
        if self.debug:
            print(f"DEBUG: Tracker initialized with max_frames={max_frames}, track_memory_frames={self.track_memory_frames}")
        
        self.tracking_history = deque(maxlen=max_frames)
        self.current_tracks = []
        self.lost_tracks = {}  # track_id -> {box: BoundingBox, frames_lost: int}
    
    def _generate_track_id(self):
        """Generate a unique UUID for a new track"""
        return str(uuid.uuid4())
    
    def calculate_cost(self, box1, box2):
        """Calculate cost between two bounding boxes as per equation (4)"""
        # Calculate center points
        cx1, cy1 = box1.center()
        cx2, cy2 = box2.center()
        
        # Euclidean distance (equation 1)
        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        # Normalized distance (equation 2 used for normalization)
        norm_dist = dist / self.max_dist
        
        # Area cost (equation 3)
        min_area = min(box1.area, box2.area)
        max_area = max(box1.area, box2.area)
        area_cost = min_area / max_area if max_area > 0 else 1.0
        
        # Final cost (equation 4)
        cost = (norm_dist * self.w_dist) + ((1 - area_cost) * self.w_area)
        
        return cost
    
    def build_cost_matrix(self, prev_boxes, curr_boxes):
        """Build cost matrix for Hungarian algorithm"""
        n_prev = len(prev_boxes)
        n_curr = len(curr_boxes)
        n = max(n_prev, n_curr)
        
        # Initialize cost matrix with high values
        cost_matrix = np.ones((n, n)) * 999.0
        
        # Fill in actual costs
        for i in range(n_prev):
            for j in range(n_curr):
                cost_matrix[i, j] = self.calculate_cost(prev_boxes[i], curr_boxes[j])
        
        return cost_matrix, n_prev, n_curr
    
    def update(self, new_detections, frame_id):
        """
        Update tracking with new detections from YOLO
        
        Args:
            new_detections: List of YOLO detection boxes (x1, y1, x2, y2 format)
            frame_id: Current frame number
            
        Returns:
            List of track IDs corresponding to each detection
        """
        # Handle empty detection list (no detections in this frame)
        if not new_detections:
            if self.debug:
                print(f"DEBUG: Frame {frame_id} has no detections")
            # Move all current tracks to lost tracks
            for track in self.current_tracks:
                if track.track_id not in self.lost_tracks:
                    self.lost_tracks[track.track_id] = {
                        'box': track,
                        'frames_lost': 1
                    }
                    if self.debug:
                        print(f"DEBUG: Moved track {track.track_id} to lost tracks")
                else:
                    self.lost_tracks[track.track_id]['frames_lost'] += 1
            
            # Age lost tracks and remove old ones
            self._age_lost_tracks()
            
            self.current_tracks = []
            self.tracking_history.append([])
            return []
        
        # Convert YOLO detections to BoundingBox objects
        new_boxes = []
        for i, detection in enumerate(new_detections):
            x1, y1, x2, y2 = detection[:4]
            bbox = BoundingBox.from_xyxy(x1, y1, x2, y2, frame_id)
            new_boxes.append(bbox)
        
        # If this is the first frame or no existing tracks, assign new track IDs to all boxes
        if not self.current_tracks and not self.lost_tracks:
            track_ids = []
            for box in new_boxes:
                box.track_id = self._generate_track_id()
                track_ids.append(box.track_id)
                if self.debug:
                    print(f"DEBUG: FIRST FRAME - Assigned track ID {box.track_id} to new detection")
            self.current_tracks = new_boxes
            self.tracking_history.append(new_boxes)
            return track_ids
        
        # Combine current tracks and lost tracks for matching
        all_previous_tracks = self.current_tracks.copy()
        lost_track_list = []
        
        for track_id, lost_info in self.lost_tracks.items():
            lost_track_list.append(lost_info['box'])
            lost_track_list[-1].track_id = track_id  # Ensure track_id is preserved
        
        all_previous_tracks.extend(lost_track_list)
        
        if not all_previous_tracks:
            # No previous tracks at all, assign new IDs
            track_ids = []
            for box in new_boxes:
                box.track_id = self._generate_track_id()
                track_ids.append(box.track_id)
                if self.debug:
                    print(f"DEBUG: No previous tracks - Assigned track ID {box.track_id} to new detection")
            self.current_tracks = new_boxes
            self.tracking_history.append(new_boxes)
            return track_ids
        
        # Build cost matrix including lost tracks
        cost_matrix, n_prev, n_curr = self.build_cost_matrix(all_previous_tracks, new_boxes)
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Assign track IDs based on the matching
        assigned_curr_indices = set()
        track_ids = [None] * len(new_boxes)
        recovered_tracks = set()  # Track IDs that were recovered from lost tracks
        
        if self.debug:
            print(f"DEBUG: Hungarian assignment - rows: {row_indices}, cols: {col_indices}")
            print(f"DEBUG: Cost threshold: {self.cost_threshold}")
            print(f"DEBUG: Current tracks: {len(self.current_tracks)}, Lost tracks: {len(self.lost_tracks)}")
        
        for i, j in zip(row_indices, col_indices):
            # Only consider valid assignments (not dummy rows/columns)
            if i < n_prev and j < n_curr:
                cost = cost_matrix[i, j]
                if self.debug:
                    print(f"DEBUG: Checking assignment {i}->{j}, cost: {cost:.3f}")
                # Check if cost is below threshold
                if cost < self.cost_threshold:
                    # Assign the track ID from previous box to current box
                    prev_track_id = all_previous_tracks[i].track_id
                    new_boxes[j].track_id = prev_track_id
                    track_ids[j] = prev_track_id
                    assigned_curr_indices.add(j)
                    
                    # Check if this was a lost track being recovered
                    if prev_track_id in self.lost_tracks:
                        recovered_tracks.add(prev_track_id)
                        if self.debug:
                            print(f"DEBUG: RECOVERED lost track ID {prev_track_id} for detection {j} (was lost for {self.lost_tracks[prev_track_id]['frames_lost']} frames)")
                    else:
                        if self.debug:
                            print(f"DEBUG: Continued track ID {prev_track_id} for detection {j}")
                else:
                    if self.debug:
                        print(f"DEBUG: Cost {cost:.3f} above threshold {self.cost_threshold}, not assigning")
        
        # Remove recovered tracks from lost tracks
        for track_id in recovered_tracks:
            del self.lost_tracks[track_id]
        
        # Assign new track IDs to unassigned current boxes (new insects)
        for j in range(n_curr):
            if j not in assigned_curr_indices:
                new_boxes[j].track_id = self._generate_track_id()
                track_ids[j] = new_boxes[j].track_id
                if self.debug:
                    print(f"DEBUG: Assigned NEW track ID {new_boxes[j].track_id} to detection {j}")
        
        # Move unmatched current tracks to lost tracks (tracks that disappeared this frame)
        matched_track_ids = {track_ids[j] for j in assigned_curr_indices if track_ids[j] is not None}
        for track in self.current_tracks:
            if track.track_id not in matched_track_ids and track.track_id not in recovered_tracks:
                if track.track_id not in self.lost_tracks:
                    self.lost_tracks[track.track_id] = {
                        'box': track,
                        'frames_lost': 1
                    }
                    if self.debug:
                        print(f"DEBUG: Track {track.track_id} disappeared, moved to lost tracks")
        
        # Age lost tracks and remove old ones
        self._age_lost_tracks()
        
        # Update current tracks
        self.current_tracks = new_boxes
        
        # Add to tracking history
        self.tracking_history.append(new_boxes)
        
        return track_ids 
    
    def _age_lost_tracks(self):
        """Age lost tracks and remove those that have been lost too long"""
        tracks_to_remove = []
        for track_id, lost_info in self.lost_tracks.items():
            lost_info['frames_lost'] += 1
            if lost_info['frames_lost'] > self.track_memory_frames:
                tracks_to_remove.append(track_id)
                if self.debug:
                    print(f"DEBUG: Permanently removing track {track_id} (lost for {lost_info['frames_lost']} frames)")
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id] 

    def get_tracking_stats(self):
        """Get current tracking statistics for debugging/monitoring"""
        return {
            'active_tracks': len(self.current_tracks),
            'lost_tracks': len(self.lost_tracks),
            'active_track_ids': [track.track_id for track in self.current_tracks],
            'lost_track_ids': list(self.lost_tracks.keys()),
            'total_history_frames': len(self.tracking_history)
        } 