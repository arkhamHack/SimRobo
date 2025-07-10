import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2

from ml.vjepa2 import VJEPA2Handler
from ml.trajectory_optimizer import TrajectoryOptimizer


class TrajectoryPredictor:
    """
    Predict object trajectories in video using V-JEPA 2 and CEM optimization
    """
    
    def __init__(
        self,
        vjepa_handler: VJEPA2Handler,
        planning_horizon: int = 10,
        confidence_threshold: float = 0.7,
        max_frames: int = 30
    ):
        """
        Initialize trajectory predictor
        
        Args:
            vjepa_handler: V-JEPA 2 model handler
            planning_horizon: Number of frames to predict ahead
            confidence_threshold: Minimum confidence for predictions
            max_frames: Maximum number of frames to process
        """
        self.vjepa_handler = vjepa_handler
        self.planning_horizon = planning_horizon
        self.confidence_threshold = confidence_threshold
        self.max_frames = max_frames
        
        # Initialize trajectory optimizer
        self.trajectory_optimizer = TrajectoryOptimizer(
            state_dim=4,  # x, y, width, height (bounding box)
            horizon=planning_horizon,
            population_size=64,
            elite_fraction=0.1,
            iterations=3
        )
        
        # Store current object information
        self.object_text_query = None
        self.object_embedding = None
        self.observed_trajectory = None
        
    def select_object_with_text(
        self, 
        video_frames: List[np.ndarray], 
        text_query: str
    ) -> Tuple[List[List[int]], str]:
        """
        Select object to track based on text description
        
        Args:
            video_frames: List of video frames as numpy arrays
            text_query: Text description of object to track
            
        Returns:
            Tuple of (observed trajectory, status message)
        """
        if not video_frames:
            return None, "No video frames provided"
            
        # Store text query
        self.object_text_query = text_query
        
        # Encode text query
        text_embedding = self.vjepa_handler.encode_text(text_query)
        self.object_embedding = text_embedding
        
        # Find object in first frame
        initial_frame = video_frames[0]
        bbox, confidence = self.vjepa_handler.find_object_in_frame(
            initial_frame, text_embedding, self.confidence_threshold
        )
        
        if confidence < self.confidence_threshold:
            return None, f"Object '{text_query}' not found with sufficient confidence"
            
        # Track object across frames to establish initial trajectory
        num_frames_to_track = min(len(video_frames), 10)  # Track in first 10 frames or all if fewer
        observed_trajectory = self.vjepa_handler.track_object_across_frames(
            video_frames[:num_frames_to_track], bbox
        )
        
        self.observed_trajectory = observed_trajectory
        return observed_trajectory, f"Object '{text_query}' tracked successfully"
        
    def _trajectory_to_states(self, trajectory: List[List[int]]) -> np.ndarray:
        """
        Convert trajectory of bounding boxes to state representation
        
        Args:
            trajectory: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Array of states [x, y, width, height]
        """
        states = []
        for bbox in trajectory:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            states.append([x1, y1, width, height])
        return np.array(states)
        
    def _states_to_trajectory(self, states: np.ndarray) -> List[List[int]]:
        """
        Convert states back to trajectory of bounding boxes
        
        Args:
            states: Array of states [x, y, width, height]
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        trajectory = []
        for state in states:
            x, y, width, height = map(int, state)
            trajectory.append([x, y, x + width, y + height])
        return trajectory
        
    def _get_state_bounds(self, frame_shape: Tuple[int, int], observed_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds for state prediction based on frame size and observed states
        
        Args:
            frame_shape: (height, width) of video frame
            observed_states: Array of observed states
            
        Returns:
            Tuple of (lower_bound, upper_bound) for states
        """
        height, width = frame_shape[:2]
        
        # Calculate mean and std of observed states
        mean_state = np.mean(observed_states, axis=0)
        std_state = np.std(observed_states, axis=0) + 1e-6  # Avoid zero std
        
        # Set bounds based on observed behavior with some margin
        # Position bounds (x, y)
        position_lower = np.array([0, 0])
        position_upper = np.array([width, height])
        
        # Size bounds (width, height) - allow reasonable variation around observed sizes
        size_mean = mean_state[2:4]
        size_std = std_state[2:4]
        size_lower = np.maximum(size_mean - 3 * size_std, 10)  # Minimum size 10 pixels
        size_upper = np.minimum(size_mean + 3 * size_std, np.array([width/2, height/2]))  # Max half of frame
        
        lower_bound = np.concatenate([position_lower, size_lower])
        upper_bound = np.concatenate([position_upper, size_upper])
        
        return lower_bound, upper_bound
        
    def predict_trajectory(
        self, 
        video_frames: List[np.ndarray], 
        text_query: str = None
    ) -> Dict[str, Any]:
        """
        Predict future trajectory of object described by text query
        
        Args:
            video_frames: List of video frames
            text_query: Text description of object to track (optional if already selected)
            
        Returns:
            Dictionary with prediction results including trajectories and visualizations
        """
        if not video_frames:
            return {"success": False, "message": "No video frames provided"}
            
        # If text query is provided or no object is selected yet, select object
        if text_query or self.object_embedding is None:
            observed_trajectory, status = self.select_object_with_text(
                video_frames, text_query or self.object_text_query
            )
            
            if observed_trajectory is None:
                return {"success": False, "message": status}
        else:
            observed_trajectory = self.observed_trajectory
            
        # Convert trajectory to state representation
        observed_states = self._trajectory_to_states(observed_trajectory)
        
        # Get state bounds based on frame size and observed states
        frame_shape = video_frames[0].shape
        state_bounds = self._get_state_bounds(frame_shape, observed_states)
        
        # Get object embedding from the last observed frame
        last_frame = video_frames[len(observed_trajectory) - 1]
        last_bbox = observed_trajectory[-1]
        
        # Extract the object from the last frame
        object_crop = self.vjepa_handler.extract_object_from_frame(last_frame, last_bbox)
        object_tensor = self.vjepa_handler.process_image(object_crop)
        current_embedding = self.vjepa_handler.encode_image(object_tensor)
        
        # Use CEM to predict future trajectory
        predicted_states = self.trajectory_optimizer.optimize(
            self.vjepa_handler,
            current_embedding,
            observed_states[-1],  # Start prediction from the last observed state
            state_bounds
        )
        
        # Convert predicted states back to bounding box trajectory
        predicted_trajectory = self._states_to_trajectory(predicted_states)
        
        # Create visualization frames
        visualization_frames = self._create_visualization(
            video_frames, 
            observed_trajectory,
            predicted_trajectory
        )
        
        return {
            "success": True,
            "observed_trajectory": observed_trajectory,
            "predicted_trajectory": predicted_trajectory,
            "visualization_frames": visualization_frames,  # These would be encoded to base64 in the server
            "object_query": text_query or self.object_text_query
        }
        
    def _create_visualization(
        self, 
        frames: List[np.ndarray],
        observed_trajectory: List[List[int]],
        predicted_trajectory: List[List[int]]
    ) -> List[np.ndarray]:
        """
        Create visualization of observed and predicted trajectories
        
        Args:
            frames: Original video frames
            observed_trajectory: List of observed bounding boxes
            predicted_trajectory: List of predicted bounding boxes
            
        Returns:
            List of visualization frames with trajectories drawn
        """
        visualization_frames = []
        
        # Color settings
        observed_color = (0, 255, 0)  # Green for observed
        predicted_color = (0, 0, 255)  # Red for predicted
        
        # Add observed trajectory to frames
        for i, bbox in enumerate(observed_trajectory):
            if i >= len(frames):
                break
                
            frame = frames[i].copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), observed_color, 2)
            cv2.putText(frame, "Observed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, observed_color, 2)
            visualization_frames.append(frame)
            
        # Add predicted trajectory to new frames (or repeat the last frame)
        last_frame = frames[min(len(observed_trajectory) - 1, len(frames) - 1)].copy()
        
        for bbox in predicted_trajectory:
            # Create a copy of the last observed frame for each prediction
            prediction_frame = last_frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(prediction_frame, (x1, y1), (x2, y2), predicted_color, 2)
            cv2.putText(prediction_frame, "Predicted", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, predicted_color, 2)
            visualization_frames.append(prediction_frame)
            
        return visualization_frames
