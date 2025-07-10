import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
import os
import cv2
from huggingface_hub import hf_hub_download
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from torchvision.ops import box_iou
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VJEPA2Handler:
    """
    Handler for V-JEPA 2 model for visual trajectory prediction
    Loads pretrained model from HuggingFace and provides utility functions for
    encoding images and text, detecting objects, and tracking trajectories
    """
    
    def __init__(
        self,
        vision_model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        text_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",  # LLM for text encoding
        device: Optional[str] = None,
        max_frames: int = 16
    ):
        """
        Initialize V-JEPA 2 handler with LLM text encoder
        
        Args:
            vision_model_name: HuggingFace model name for V-JEPA2 (default: "facebook/vjepa2-vitl-fpc64-256")
            text_model_name: HuggingFace model name for text encoding (default: "mistralai/Mistral-7B-Instruct-v0.2")
            device: Device to run models on (default: GPU if available, else CPU)
            max_frames: Maximum number of frames to process at once (default: 16)
        """
        # Store parameters
        self.max_frames = max_frames
        
        # Set device
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading V-JEPA2 vision model: {vision_model_name}")
        start_time = time.time()
        
        try:
            self.vision_processor = AutoVideoProcessor.from_pretrained(vision_model_name)
            
            self.vision_model = AutoModel.from_pretrained(vision_model_name).to(self.device)
            
            self.vision_model.eval()
            
            logger.info(f"V-JEPA2 vision model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load V-JEPA2 model: {e}")
            raise
        
        logger.info(f"Loading text model: {text_model_name}")
        start_time = time.time()
        
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            
            if self.device.type == "cuda":
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    text_model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.text_model = AutoModelForCausalLM.from_pretrained(text_model_name).to(self.device)
            
            # Set to evaluation mode
            self.text_model.eval()
            
            self.vision_embedding_dim = self.vision_model.config.hidden_size
            self.text_embedding_dim = self.text_model.config.hidden_size
            
            logger.info(f"Text model loaded in {time.time() - start_time:.2f} seconds")
            logger.info(f"Vision embedding dimension: {self.vision_embedding_dim}")
            logger.info(f"Text embedding dimension: {self.text_embedding_dim}")
            
            self.text_projection = nn.Linear(
                self.text_embedding_dim, 
                self.vision_embedding_dim,
                bias=False
            ).to(self.device)
            
            # Initialize with identity-like mapping
            with torch.no_grad():
                if self.text_embedding_dim <= self.vision_embedding_dim:
                    eye = torch.eye(self.text_embedding_dim)
                    self.text_projection.weight[:self.text_embedding_dim, :] = eye
                else:
                    eye = torch.eye(self.text_embedding_dim)[:self.vision_embedding_dim, :]
                    self.text_projection.weight = nn.Parameter(eye)
                    
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            raise
        
        # Get the patch size for spatial operations
        try:
            self.patch_size = self.vision_model.config.patch_size
            logger.info(f"Patch size: {self.patch_size}")
        except AttributeError:
            logger.warning("Could not determine patch size, defaulting to 16")
            self.patch_size = 16
        
    def process_image(self, image: Union[np.ndarray, torch.Tensor, list, Image.Image]):
        """
        Process image or video frames for V-JEPA 2 model input using the HuggingFace processor
        
        Args:
            image: Single RGB image as numpy array (H, W, 3), PIL Image, torch.Tensor, or list of images
            
        Returns:
            Processed image inputs ready for the vision model
        """
        try:
            # Handle list of images (e.g., video frames)
            if isinstance(image, list):
                if len(image) > self.max_frames:
                    logger.warning(f"Number of frames ({len(image)}) exceeds max_frames ({self.max_frames}). Using first {self.max_frames} frames.")
                    image = image[:self.max_frames]
                
                # Convert all frames to PIL Images if they're numpy arrays
                if isinstance(image[0], np.ndarray):
                    images = [Image.fromarray(img.astype(np.uint8)) if img.dtype != np.uint8 else Image.fromarray(img) 
                              for img in image]
                else:
                    images = image  # Assume list of PIL Images
                    
                # Process the batch of images using the V-JEPA2 processor
                inputs = self.vision_processor(images=images, return_tensors="pt")
                return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Handle single image as numpy array
            elif isinstance(image, np.ndarray):
                # Convert to PIL image
                if image.ndim == 3 and image.shape[2] == 3:  # RGB image
                    image_pil = Image.fromarray(image.astype(np.uint8)) if image.dtype != np.uint8 else Image.fromarray(image)
                    inputs = self.vision_processor(images=image_pil, return_tensors="pt")
                    return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                else:
                    raise ValueError(f"Input numpy array must have shape [H, W, 3], got {image.shape}")
            
            # Handle PIL Image
            elif isinstance(image, Image.Image):
                inputs = self.vision_processor(images=image, return_tensors="pt")
                return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Handle torch Tensor (assume already processed or in the right format)
            elif isinstance(image, torch.Tensor):
                if image.ndim == 3 and (image.shape[0] == 3 or image.shape[2] == 3):  # [C, H, W] or [H, W, C]
                    # Convert to [C, H, W] if needed
                    if image.shape[2] == 3:
                        image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    
                    # Add batch dimension and process
                    image = image.unsqueeze(0)  # [1, C, H, W]
                    inputs = {"pixel_values": image.to(self.device)}
                    return inputs
                else:
                    raise ValueError(f"Input tensor must have unexpected shape: {image.shape}")
                
            # Handle dict format (for compatibility with old code)
            elif isinstance(image, dict) and "pixel_values" in image:
                return image  # Return the dict as-is for the model
                
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def encode_image(self, image: Union[np.ndarray, torch.Tensor, dict, Image.Image, list]) -> torch.Tensor:
        """
        Encode image or video frames to latent embeddings using V-JEPA2 model
        
        Args:
            image: RGB image as numpy array (H, W, 3), PIL Image, list of frames, torch.Tensor, or processor output dict
            
        Returns:
            Latent embedding tensor with shape [batch_size, sequence_length, embedding_dim]
        """
        try:
            # Handle dict format (from processor or previous processing)
            if isinstance(image, dict) and "pixel_values" in image:
                # Assume already preprocessed and ready for the model
                inputs = image
            else:
                # Process the image/video first
                inputs = self.process_image(image)
            
            # Run the vision model to get embeddings
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
                
                # Get the last hidden states which contain the embeddings
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = outputs.last_hidden_state
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    embeddings = outputs[0]  # Common pattern in transformer models
                else:
                    logger.warning("Unexpected output format from vision model, using raw output")
                    embeddings = outputs
                    
                logger.debug(f"Vision embeddings shape: {embeddings.shape}")
                return embeddings
                    
        except Exception as e:
            logger.error(f"Error in encoding image: {str(e)}")
            raise
            
    def predict_future_state(
        self, 
        current_state: torch.Tensor,
        action_encoding: torch.Tensor,
        state_tensor: Optional[torch.Tensor] = None,
        steps_ahead: int = 1
    ) -> torch.Tensor:
        """
        Predict future state embedding based on current state and action
        
        Args:
            current_state: Current state embedding from the encoder with shape [batch, tokens, dim]
            action_encoding: Action tensor with shape [batch, 7] representing [dx, dy, dz, dqw, dqx, dqy, dqz]
            state_tensor: Current state tensor with shape [batch, steps, 7] representing [x, y, z, qw, qx, qy, qz]
                         If None, will create a default state at origin with identity rotation
            steps_ahead: Number of steps to predict ahead (not used directly, but kept for API compatibility)
            
        Returns:
            Predicted future state embedding with same shape as current_state
        """
        try:
            # Ensure tensors are on the right device
            current_state = current_state.to(self.device)
            action_encoding = action_encoding.to(self.device)
            
            # Create default state tensor if not provided
            if state_tensor is None:
                # Default state at origin with identity quaternion rotation [x, y, z, qw, qx, qy, qz]
                batch_size = current_state.shape[0]
                state_tensor = torch.zeros(batch_size, 1, 7, device=self.device)
                state_tensor[:, :, 3] = 1.0  # qw=1 for identity quaternion
            else:
                state_tensor = state_tensor.to(self.device)
            
            # Calculate tokens per frame based on image size and patch size
            # This depends on image resolution / patch size^2 (e.g., 224×224 / 16^2 = 196)
            frame_size = 224  # Standard processed image size for most ViT models
            tokens_per_frame = (frame_size // self.patch_size) ** 2
            
            # Extract context embedding from current state (first frame tokens)
            z_context = current_state[:, :tokens_per_frame, :]
            
            # Use the predictor to get future state embedding
            with torch.no_grad():
                # Predictor expects (context_embedding, actions, states, extrinsics=None)
                future_embedding = self.predictor(z_context, action_encoding, state_tensor)
                
                # Return the full embedding matching the input shape
                # This assumes the predictor's output is matched to the input size or we need to pad
                if future_embedding.shape[1] != current_state.shape[1]:
                    # If the shapes don't match, pad with original content
                    # This is a simplification - in a real implementation you'd handle this properly
                    padded_future = current_state.clone()
                    padded_future[:, :future_embedding.shape[1], :] = future_embedding
                    return padded_future
                else:
                    return future_embedding
                
        except Exception as e:
            print(f"Error in predict_future_state: {e}")
            import traceback
            traceback.print_exc()
            # As fallback, just return the input state unmodified
            return current_state
    
    def encode_action(self, action: np.ndarray) -> torch.Tensor:
        """
        Encode robot action to format expected by V-JEPA 2-AC
        
        Args:
            action: Robot action as numpy array [x, y, z, qx, qy, qz, qw, gripper]
            
        Returns:
            Action encoding tensor formatted for the V-JEPA 2-AC predictor
        """
        try:
            # Convert action to tensor
            # Extract the position and orientation components
            # Assuming action format is [x, y, z, qx, qy, qz, qw, gripper]
            pos = action[0:3] 
            quat = action[3:7]
            
            # Reorder quaternion to match the model's expected format [qw, qx, qy, qz]
            # V-JEPA 2-AC expects [qw, qx, qy, qz] but our action has [qx, qy, qz, qw]
            quat_ordered = [quat[3], quat[0], quat[1], quat[2]]  # Reorder to [qw, qx, qy, qz]
            
            # Combine into the model's expected format [x, y, z, qw, qx, qy, qz]
            action_formatted = np.concatenate([pos, quat_ordered])
            
            # Convert to tensor and add batch dimension
            action_tensor = torch.tensor(action_formatted, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Format as expected by the predictor (may need adjustment based on specific model requirements)
            # The model might expect batched actions: [batch_size, 1, 7]  # 7D vector (position + quaternion)
            if action_tensor.ndim == 2:  # [batch, 7]
                action_tensor = action_tensor.unsqueeze(1)  # [batch, 1, 7]
            
            return action_tensor
            
        except Exception as e:
            print(f"Error encoding action: {e}")
            # Return a default action tensor in case of error
            # This should be handled better in production code
            return torch.zeros(1, 1, 7).to(self.device)
    
    def compute_embedding_distance(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ):
        """
        Compute distance between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            L1 distance between embeddings
        """
        # Ensure embeddings are on the same device
        if embedding1.device != embedding2.device:
            embedding2 = embedding2.to(embedding1.device)
            
        # Compute L1 distance over the embedding dimension
        distance = torch.abs(embedding1 - embedding2).mean().item()
        return distance
        
    def encode_text(self, text_query: str) -> torch.Tensor:
        """
        Encode text query using LLM (Mistral) and project to V-JEPA2 embedding space
        
        Args:
            text_query: Text description of object to track
            
        Returns:
            Text embedding compatible with visual embeddings
        """
        try:
            logger.debug(f"Encoding text: '{text_query}'")
            
            # Tokenize the text input
            tokens = self.text_tokenizer(text_query, return_tensors="pt").to(self.device)
            
            # Get text embeddings from LLM
            with torch.no_grad():
                # Forward pass through the text model
                outputs = self.text_model(**tokens, output_hidden_states=True)
                
                # Get the last hidden state (text embedding)
                if hasattr(outputs, "last_hidden_state"):
                    # Use the last token embedding (for instruction-tuned models)
                    text_embedding = outputs.last_hidden_state[:, -1:, :]  # Shape: [1, 1, text_dim]
                elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    # Use the last layer's representation
                    text_embedding = outputs.hidden_states[-1][:, -1:, :]  # Shape: [1, 1, text_dim]
                else:
                    raise ValueError("Could not extract embeddings from text model")
                
                logger.debug(f"Raw text embedding shape: {text_embedding.shape}")
                
                # Project text embedding to vision embedding space
                projected_embedding = self.text_projection(text_embedding)  # Shape: [1, 1, vision_dim]
                
                logger.debug(f"Projected text embedding shape: {projected_embedding.shape}")
                return projected_embedding
            
        except Exception as e:
            logger.error(f"Error in text encoding: {str(e)}")
            # Fallback to random embedding if something goes wrong
            logger.warning("Using fallback random text embedding")
            return torch.randn(1, 1, self.vision_embedding_dim, device=self.device)
    
    def compute_similarity_map(self, frame: np.ndarray, text_embedding: torch.Tensor) -> np.ndarray:
        """
        Compute similarity map between frame regions and text description
        
        Args:
            frame: Video frame as numpy array (H, W, 3)
            text_embedding: Text embedding of object description
            
        Returns:
            Similarity map as numpy array (H, W)
        """
        try:
            # Process frame
            frame_inputs = self.process_image(frame)
            
            # Calculate embedding for each patch of the frame
            with torch.no_grad():
                frame_embedding = self.encode_image(frame_inputs)  # [1, N, D]
                
                logger.debug(f"Frame embedding shape: {frame_embedding.shape}")
                logger.debug(f"Text embedding shape: {text_embedding.shape}")
                
                # Normalize embeddings for cosine similarity
                frame_emb_norm = frame_embedding / frame_embedding.norm(dim=-1, keepdim=True)
                text_emb_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                
                # Compute similarity between each patch and the text embedding
                # The text embedding might need to be expanded/broadcast to match frame tokens
                if text_emb_norm.shape[1] == 1:
                    # We have a single text token, so we can broadcast it against all frame patches
                    similarity = torch.matmul(frame_emb_norm, text_emb_norm.transpose(-2, -1))  # [1, N, 1]
                    similarity = similarity.squeeze(-1)  # [1, N]
                else:
                    # Multiple text tokens - take mean similarity across text tokens
                    similarity = torch.matmul(frame_emb_norm, text_emb_norm.transpose(-2, -1))  # [1, N, T]
                    similarity = similarity.mean(dim=-1)  # [1, N]
                
                # Remove batch dimension
                similarity = similarity.squeeze(0)  # [N]
                
                # Determine spatial dimensions for reshaping
                # Get input image dimensions
                if "pixel_values" in frame_inputs:
                    _, _, H_img, W_img = frame_inputs["pixel_values"].shape
                else:
                    # Default to standard input size if we can't determine
                    H_img, W_img = 224, 224
                
                # Calculate grid size based on patch size
                H = H_img // self.patch_size
                W = W_img // self.patch_size
                
                # Ensure we have the right number of patches
                expected_patches = H * W
                actual_patches = similarity.shape[0]
                
                if expected_patches != actual_patches:
                    logger.warning(f"Patch count mismatch: expected {expected_patches}, got {actual_patches}")
                    logger.warning(f"Using sqrt to estimate grid dimensions")
                    
                    # Try to find factors close to sqrt
                    grid_size = int(actual_patches ** 0.5)
                    if grid_size * grid_size == actual_patches:
                        # Perfect square
                        H = W = grid_size
                    else:
                        # Find factors
                        for h in range(int(actual_patches ** 0.5), 0, -1):
                            if actual_patches % h == 0:
                                H = h
                                W = actual_patches // h
                                break
                
                logger.debug(f"Reshaping similarity of shape {similarity.shape} to ({H}, {W})")
                
                # Reshape and convert to numpy
                try:
                    similarity_map = similarity.reshape(H, W).cpu().numpy()
                except Exception as reshape_error:
                    logger.error(f"Reshape error: {reshape_error}")
                    # Fallback to flattened map if reshape fails
                    return similarity.cpu().numpy()
                
                # Normalize to [0, 1] range
                similarity_map = (similarity_map - similarity_map.min()) / \
                                (similarity_map.max() - similarity_map.min() + 1e-8)
                
                return similarity_map
                
        except Exception as e:
            logger.error(f"Error computing similarity map: {e}")
            import traceback
            traceback.print_exc()
            # Return empty similarity map
            return np.zeros((24, 24))  # Default small grid size
    
    def find_object_in_frame(self, frame: np.ndarray, text_embedding: torch.Tensor, 
                             threshold: float = 0.7) -> Tuple[List[int], float]:
        """
        Locate object in frame based on text description
        
        Args:
            frame: Video frame as numpy array (H, W, 3)
            text_embedding: Text embedding of object description
            threshold: Similarity threshold
            
        Returns:
            bounding_box: [x1, y1, x2, y2] coordinates of object
            confidence: Similarity score
        """
        # Get similarity map
        similarity_map = self.compute_similarity_map(frame, text_embedding)
        
        # Resize similarity map to frame size
        similarity_map_resized = cv2.resize(similarity_map, 
                                          (frame.shape[1], frame.shape[0]), 
                                          interpolation=cv2.INTER_CUBIC)
        
        # Threshold map to create binary mask
        object_mask = (similarity_map_resized > threshold).astype(np.uint8) * 255
        
        # Find contours in the mask
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0, 0, 0, 0], 0.0
            
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Get confidence as maximum similarity in the region
        confidence = similarity_map_resized[y:y+h, x:x+w].max()
        
        return [x, y, x+w, y+h], float(confidence)
    
    def track_object_across_frames(self, frames: List[np.ndarray], initial_bbox: List[int]) -> List[List[int]]:
        """
        Track object across sequence of frames
        
        Args:
            frames: List of video frames
            initial_bbox: Initial bounding box [x1, y1, x2, y2]
            
        Returns:
            List of bounding boxes for each frame
        """
        if not frames:
            return []
            
        # Initialize with given bounding box
        bboxes = [initial_bbox]
        
        tracker = cv2.TrackerKCF_create() 
        tracker.init(frames[0], tuple(initial_bbox))  #
        for i in range(1, len(frames)):
            success, bbox = tracker.update(frames[i])
            
            if success:
                x, y, w, h = bbox
                bboxes.append([int(x), int(y), int(x+w), int(y+h)])
            else:
                bboxes.append(bboxes[-1])
                
        return bboxes
    
    def extract_object_from_frame(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract object from frame using bounding box
        
        Args:
            frame: Video frame as numpy array (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped object as numpy array
        """
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]


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
