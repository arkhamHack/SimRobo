import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ml.vjepa2 import VJEPA2Handler

class TrajectoryOptimizer:
    """
    Cross-Entropy Method (CEM) optimizer for trajectory prediction
    Adapted from ActionOptimizer to predict object trajectories in video
    """
    
    def __init__(
        self,
        state_dim: int = 4,  # x, y, width, height (bounding box)
        horizon: int = 10,
        population_size: int = 64,
        elite_fraction: float = 0.1,
        iterations: int = 3,
        initial_std: float = 0.1
    ):
        """
        Initialize CEM optimizer for trajectory prediction
        
        Args:
            state_dim: Dimensionality of state space (default: 4 for bounding box)
            horizon: Planning horizon (number of steps to predict ahead)
            population_size: Size of candidate population
            elite_fraction: Fraction of population to select as elite
            iterations: Number of optimization iterations
            initial_std: Initial standard deviation for sampling
        """
        self.state_dim = state_dim
        self.horizon = horizon
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.iterations = iterations
        self.initial_std = initial_std
        
    def optimize(
        self,
        vjepa_handler: VJEPA2Handler,
        current_state_embedding: torch.Tensor,
        current_state: np.ndarray,
        state_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Optimize trajectory prediction using CEM
        
        Args:
            vjepa_handler: V-JEPA 2 handler
            current_state_embedding: Current state embedding from V-JEPA 2
            current_state: Current object state [x, y, width, height]
            state_bounds: Tuple of (lower_bound, upper_bound) for states
            
        Returns:
            Predicted future states as array of shape [horizon, state_dim]
        """
        lower_bound, upper_bound = state_bounds
        
        # Initialize mean and std for sampling
        mean = np.tile(current_state, (self.horizon, 1))  # Initialize with current state repeated
        std = np.ones_like(mean) * self.initial_std
        
        # Ensure standard deviation respects state bounds
        std = np.minimum(std, (upper_bound - lower_bound) / 4)
        
        device = current_state_embedding.device
        
        # Keep track of best trajectory
        best_trajectory = None
        best_score = float('inf')
        
        for iteration in range(self.iterations):
            # Sample population
            population = np.random.normal(
                loc=np.tile(mean, (self.population_size, 1, 1)), 
                scale=np.tile(std, (self.population_size, 1, 1))
            ).reshape(self.population_size, self.horizon, self.state_dim)
            
            # Clip states to bounds
            population = np.clip(population, lower_bound, upper_bound)
            
            # Evaluate population
            scores = np.zeros(self.population_size)
            
            for i in range(self.population_size):
                # Simulate trajectory and evaluate
                score = self._evaluate_trajectory(
                    population[i], 
                    current_state_embedding,
                    vjepa_handler
                )
                scores[i] = score
                
                # Track best trajectory
                if score < best_score:
                    best_score = score
                    best_trajectory = population[i].copy()
            
            # Select elite samples
            elite_indices = np.argsort(scores)[:self.elite_size]
            elite_samples = population[elite_indices]
            
            # Update distribution
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-6  # Avoid zero std
            
            # Reduce std over iterations for convergence
            std = std * np.sqrt((self.iterations - iteration) / self.iterations)
        
        # Return best trajectory found
        return best_trajectory
    
    def _evaluate_trajectory(
        self, 
        trajectory: np.ndarray, 
        current_embedding: torch.Tensor,
        vjepa_handler: VJEPA2Handler
    ) -> float:
        """
        Evaluate a candidate trajectory
        
        Args:
            trajectory: Sequence of states [horizon, state_dim]
            current_embedding: Current state embedding
            vjepa_handler: V-JEPA 2 handler
            
        Returns:
            Score (lower is better)
        """
        # This implementation evaluates trajectory based on:
        # 1. Smoothness (penalize jerky movements)
        # 2. Plausibility (using V-JEPA 2 to check if transitions are realistic)
        
        # Calculate smoothness penalty
        if trajectory.shape[0] > 1:
            # Calculate velocity (first derivative)
            velocities = trajectory[1:] - trajectory[:-1]
            # Calculate acceleration (second derivative)
            accelerations = velocities[1:] - velocities[:-1] if velocities.shape[0] > 1 else np.zeros_like(velocities)
            
            # Penalize large velocities and accelerations
            smoothness_penalty = (
                np.mean(np.square(velocities)) + 
                2 * np.mean(np.square(accelerations))
            )
        else:
            smoothness_penalty = 0.0
            
        # Calculate plausibility score using V-JEPA 2
        # This is a placeholder for using V-JEPA 2 to evaluate how realistic the trajectory is
        # In a real implementation, you would encode each state and check how well it follows
        # the expected visual dynamics according to the model
        plausibility_score = 0.0
        
        # Combined score (lower is better)
        return smoothness_penalty + plausibility_score
        
    def predict_next_state(
        self, 
        current_state: np.ndarray, 
        state_bounds: Tuple[np.ndarray, np.ndarray],
        vjepa_handler: VJEPA2Handler,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Predict the next state given the current state
        
        Args:
            current_state: Current bounding box [x, y, width, height]
            state_bounds: Tuple of (lower_bound, upper_bound) for states
            vjepa_handler: V-JEPA 2 handler
            frame: Current video frame
            
        Returns:
            Next predicted state [x, y, width, height]
        """
        # Extract object using current bounding box
        x1, y1 = int(current_state[0]), int(current_state[1])
        width, height = int(current_state[2]), int(current_state[3])
        x2, y2 = x1 + width, y1 + height
        
        # Get the cropped object
        object_crop = frame[y1:y2, x1:x2]
        if object_crop.size == 0:  # Check if crop is valid
            return current_state  # Return current state if crop is invalid
            
        # Process and encode the object
        object_tensor = vjepa_handler.process_image(object_crop)
        current_embedding = vjepa_handler.encode_image(object_tensor)
        
        # Use the optimizer to predict the next state
        predicted_trajectory = self.optimize(
            vjepa_handler,
            current_embedding,
            current_state,
            state_bounds
        )
        
        # Return just the next state (first state in the trajectory)
        return predicted_trajectory[0] if predicted_trajectory is not None else current_state
