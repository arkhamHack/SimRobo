import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
import os
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel


class VJEPA2Handler:
    """
    Handler for V-JEPA 2 model for robotic control
    Loads pretrained model and provides utility functions for encoding images
    and predicting future states based on actions
    """
    
    def __init__(
        self,
        model_name: str = "facebook/vjepa-2-ac",
        device: Optional[str] = None
    ):
        """
        Initialize V-JEPA 2 handler
        
        Args:
            model_name: Model name or path (default: "facebook/vjepa-2-ac")
            device: Device to run model on (default: GPU if available, else CPU)
        """
        if device is None:
            # Check for MPS (Apple Silicon) first, then CUDA, then fall back to CPU
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Loading V-JEPA 2 model from {model_name} on {self.device}...")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("V-JEPA 2 model loaded successfully")

    def process_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Process image for model input
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Processed image tensor ready for model input
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Apply preprocessing
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    def encode_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encode image to latent embedding using V-JEPA 2 encoder
        
        Args:
            image: RGB image as numpy array (H, W, 3) or preprocessed tensor
            
        Returns:
            Latent embedding tensor
        """
        # Process image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = self.process_image(image)
        
        # Ensure the image is on the correct device
        image = image.to(self.device)
        
        # Encode image
        with torch.no_grad():
            outputs = self.model.get_image_features(image)
            
        return outputs
    
    def predict_future_state(
        self, 
        current_state: torch.Tensor,
        action_encoding: torch.Tensor,
        steps_ahead: int = 1
    ) -> torch.Tensor:
        """
        Predict future state embedding based on current state and action
        
        Args:
            current_state: Current state embedding
            action_encoding: Action encoding
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Predicted future state embedding
        """
        with torch.no_grad():
            # For V-JEPA 2-AC, we can directly predict future embeddings
            # This is a simplified version - actual implementation may differ
            # based on the specific V-JEPA 2-AC model architecture
            future_state = self.model.predict(
                current_state, 
                action_encoding,
                prediction_steps=steps_ahead
            )
            
        return future_state
    
    def encode_action(self, action: np.ndarray) -> torch.Tensor:
        """
        Encode robot action to format expected by V-JEPA 2-AC
        
        Args:
            action: Robot action as numpy array [x, y, z, qx, qy, qz, qw, gripper]
            
        Returns:
            Action encoding tensor
        """
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Normalize action (application-specific, may need adjustment based on action space)
        # This is a placeholder - actual encoding depends on V-JEPA 2-AC action conditioning
        action_encoding = self.model.encode_action(action_tensor)
        
        return action_encoding
    
    def compute_embedding_distance(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute distance between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            L1 distance between embeddings
        """
        return torch.nn.functional.l1_loss(embedding1, embedding2).item()


class ActionOptimizer:
    """
    Cross-Entropy Method (CEM) optimizer for action sequences
    """
    
    def __init__(
        self,
        action_dim: int = 8,  # [x, y, z, qx, qy, qz, qw, gripper]
        horizon: int = 5,
        population_size: int = 64,
        elite_fraction: float = 0.1,
        iterations: int = 3,
        initial_std: float = 0.1
    ):
        """
        Initialize CEM optimizer
        
        Args:
            action_dim: Dimensionality of action space
            horizon: Planning horizon (number of steps to plan ahead)
            population_size: Size of candidate population
            elite_fraction: Fraction of population to select as elite
            iterations: Number of optimization iterations
            initial_std: Initial standard deviation for sampling
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_fraction))
        self.iterations = iterations
        self.initial_std = initial_std
        
    def optimize(
        self,
        vjepa_handler: VJEPA2Handler,
        current_state_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
        current_action: np.ndarray,
        action_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Optimize action sequence to reach goal state
        
        Args:
            vjepa_handler: V-JEPA 2 handler
            current_state_embedding: Current state embedding
            goal_embedding: Goal state embedding
            current_action: Current robot action [x, y, z, qx, qy, qz, qw, gripper]
            action_bounds: Tuple of (lower_bound, upper_bound) for actions
            
        Returns:
            Optimized action
        """
        lower_bound, upper_bound = action_bounds
        
        # Initialize mean and std for sampling
        mean = np.tile(current_action, (self.horizon, 1))
        std = np.ones_like(mean) * self.initial_std
        
        # Ensure standard deviation respects action bounds
        std = np.minimum(std, (upper_bound - lower_bound) / 4)
        
        device = current_state_embedding.device
        
        for iteration in range(self.iterations):
            # Sample population
            population = np.random.normal(
                loc=np.tile(mean, (self.population_size, 1, 1)), 
                scale=np.tile(std, (self.population_size, 1, 1))
            ).reshape(self.population_size, self.horizon, self.action_dim)
            
            # Clip actions to bounds
            population = np.clip(population, lower_bound, upper_bound)
            
            # Evaluate population
            distances = np.zeros(self.population_size)
            
            for i in range(self.population_size):
                # Initialize predicted state with current state
                predicted_state = current_state_embedding
                
                # Rollout actions and predict future states
                for t in range(self.horizon):
                    # Encode action
                    action = population[i, t]
                    action_encoding = vjepa_handler.encode_action(action)
                    
                    # Predict next state
                    predicted_state = vjepa_handler.predict_future_state(
                        predicted_state, 
                        action_encoding
                    )
                
                # Compute distance to goal
                distances[i] = vjepa_handler.compute_embedding_distance(
                    predicted_state, 
                    goal_embedding
                )
            
            # Select elite samples
            elite_indices = np.argsort(distances)[:self.elite_size]
            elite_samples = population[elite_indices]
            
            # Update distribution
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-6  # Avoid zero std
            
            # Reduce std over iterations for convergence
            std = std * np.sqrt((self.iterations - iteration) / self.iterations)
        
        # Return first action of best action sequence
        best_idx = np.argmin(distances)
        return population[best_idx, 0]
