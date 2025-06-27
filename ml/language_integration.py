import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .vjepa2 import VJEPA2Handler
import time
import json


class LanguageIntegrator:
    """
    Integration between V-JEPA 2 visual representations and language models
    Enables natural language task specification and visual question answering
    """
    
    def __init__(
        self, 
        vjepa_handler: VJEPA2Handler,
        language_model_name: str = "openai/gpt-4o-mini",
        device: Optional[str] = None
    ):
        """
        Initialize language integration module
        
        Args:
            vjepa_handler: Initialized V-JEPA 2 handler
            language_model_name: Name of language model to use
            device: Device to run language model on (default: same as V-JEPA 2)
        """
        self.vjepa_handler = vjepa_handler
        
        if device is None:
            self.device = vjepa_handler.device
        else:
            self.device = device
        
        print(f"Loading language model {language_model_name} on {self.device}...")
        
        # Load tokenizer and language model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            device_map=self.device,
            torch_dtype=torch.float16  # Use half precision for efficiency
        )
        
        # Create VQA pipeline
        self.vqa_pipeline = pipeline(
            "visual-question-answering",
            model=self.language_model,
            tokenizer=self.tokenizer,
            device=0 if self.device in ['cuda', 'mps'] else -1
        )
        
        # Template for task specification prompting
        self.task_prompt_template = """
        You are a robot control system that can perform physical tasks.
        
        Current scene description: {scene_description}
        
        Task: {task_description}
        
        Generate a concise action plan with specific steps:
        """
        
        print("Language integration module loaded successfully")
    
    def encode_scene_description(self, image: np.ndarray) -> str:
        """
        Generate a textual description of the scene from an image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            String description of the scene
        """
        # Encode image with V-JEPA 2
        embedding = self.vjepa_handler.encode_image(image)
        
        # Generate scene description using the language model
        scene_prompt = "Describe this scene in detail, focusing on physical objects and their arrangement:"
        
        inputs = self.tokenizer(scene_prompt, return_tensors="pt").to(self.device)
        image_features = embedding.unsqueeze(0).to(self.device)
        
        # Combine text and image features
        with torch.no_grad():
            outputs = self.language_model.generate(
                **inputs, 
                encoder_hidden_states=image_features,
                max_new_tokens=100,
                do_sample=False
            )
        
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return description
    
    def parse_task_to_actions(
        self, 
        image: np.ndarray, 
        task_description: str
    ) -> List[Dict[str, float]]:
        """
        Parse natural language task description into a sequence of robot actions
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            task_description: Natural language description of the task
            
        Returns:
            List of action dictionaries with robot control parameters
        """
        # Get scene description
        scene_description = self.encode_scene_description(image)
        
        # Format prompt
        task_prompt = self.task_prompt_template.format(
            scene_description=scene_description,
            task_description=task_description
        )
        
        # Generate action plan
        inputs = self.tokenizer(task_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.language_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
        
        action_plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse action plan to structured format
        # This would require additional parsing logic based on the language model's output format
        # Here's a simplified version that assumes a specific format
        try:
            # Extract action steps
            action_steps = []
            lines = action_plan.split("\n")
            for line in lines:
                if line.strip().startswith("Step"):
                    # Parse step into action parameters
                    # For example: "Step 1: Move to position x=0.3, y=0.2, z=0.5"
                    action_dict = {}
                    
                    # Extract position coordinates
                    if "x=" in line:
                        try:
                            x_val = float(line.split("x=")[1].split(",")[0])
                            action_dict["x"] = x_val
                        except:
                            pass
                            
                    if "y=" in line:
                        try:
                            y_val = float(line.split("y=")[1].split(",")[0])
                            action_dict["y"] = y_val
                        except:
                            pass
                            
                    if "z=" in line:
                        try:
                            z_val = float(line.split("z=")[1].split(",")[0])
                            action_dict["z"] = z_val
                        except:
                            pass
                    
                    # Add gripper action if mentioned
                    if "grip" in line.lower() or "grasp" in line.lower():
                        action_dict["gripper"] = 1.0
                    elif "release" in line.lower():
                        action_dict["gripper"] = 0.0
                    
                    if action_dict:  # Add to actions if any parameters were extracted
                        action_steps.append(action_dict)
            
            return action_steps
        except Exception as e:
            print(f"Error parsing action plan: {e}")
            return []
    
    def answer_visual_question(self, image: np.ndarray, question: str) -> str:
        """
        Answer a question about an image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            question: Question about the image
            
        Returns:
            Answer to the question
        """
        # Encode image with V-JEPA 2
        embedding = self.vjepa_handler.encode_image(image)
        
        # Use VQA pipeline to answer question
        result = self.vqa_pipeline(
            image=image,  # Pipeline internally handles conversion
            question=question,
            top_k=1
        )
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]["answer"]
        else:
            return "Unable to answer the question."
            
    def predict_interaction_outcome(
        self, 
        image: np.ndarray, 
        action_description: str
    ) -> str:
        """
        Predict the outcome of an interaction with an object in natural language
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            action_description: Description of the action to be performed
            
        Returns:
            Description of predicted outcome
        """
        # Encode image
        embedding = self.vjepa_handler.encode_image(image)
        
        # Create prompt
        prompt = f"""
        Scene: I have an image showing a physical environment.
        
        Action: {action_description}
        
        Predict what will happen when this action is performed on the objects in the scene.
        Focus on the physics of the interaction and be specific about object movements.
        Outcome:
        """
        
        # Generate prediction
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        image_features = embedding.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.language_model.generate(
                **inputs,
                encoder_hidden_states=image_features,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the outcome part
        if "Outcome:" in prediction:
            prediction = prediction.split("Outcome:")[1].strip()
            
        return prediction
