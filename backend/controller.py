import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Union

from backend.mujoco_environment import FrankaPandaEnv
from ml.vjepa2 import VJEPA2Handler, ActionOptimizer


class RobotController:
    """
    Controller for robotic arm using V-JEPA 2 model
    Handles planning and execution of actions to reach goal states
    """
    
    def __init__(
        self,
        env: FrankaPandaEnv,
        vjepa_handler: VJEPA2Handler,
        planning_horizon: int = 5,
        control_hz: int = 10,
        distance_threshold: float = 0.05,
        max_steps: int = 100
    ):
        """
        Initialize robot controller
        
        Args:
            env: Simulation environment
            vjepa_handler: V-JEPA 2 model handler
            planning_horizon: Planning horizon (number of steps to plan ahead)
            control_hz: Control frequency in Hz
            distance_threshold: Distance threshold for goal reaching
            max_steps: Maximum number of steps to take
        """
        self.env = env
        self.vjepa_handler = vjepa_handler
        self.planning_horizon = planning_horizon
        self.control_hz = control_hz
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        
        # Action optimizer
        self.action_optimizer = ActionOptimizer(
            action_dim=8,  # [x, y, z, qx, qy, qz, qw, gripper]
            horizon=planning_horizon,
            population_size=64,
            elite_fraction=0.1,
            iterations=3
        )
        
        # Define action bounds
        # Position bounds (x, y, z) in meters
        self.position_bounds = np.array([
            [0.2, -0.6, 0.05],  # Lower bounds
            [0.8, 0.6, 0.6]     # Upper bounds
        ])
        
        # Initialize goal
        self.goal_embedding = None
        self.goal_image = None
        
        # Control flow flags
        self.is_running = False
        
    def set_goal(self, goal_image: np.ndarray) -> None:
        """
        Set goal state from RGB image
        
        Args:
            goal_image: Goal RGB image as numpy array (H, W, 3)
        """
        self.goal_image = goal_image.copy()
        goal_image_tensor = self.vjepa_handler.process_image(goal_image)
        self.goal_embedding = self.vjepa_handler.encode_image(goal_image_tensor)
        print("Goal state set successfully")
        
    def get_current_action(self) -> np.ndarray:
        """
        Get current robot action
        
        Returns:
            Current action as [x, y, z, qx, qy, qz, qw, gripper]
        """
        # Get current end effector pose
        pos, orn = self.env.get_ee_pose()
        
        # Get current gripper state (approximation - would need proper gripper width sensing)
        gripper_state = 0.5  # Assume mid-position for now
        
        return np.concatenate([pos, orn, [gripper_state]])
    
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute action on robot
        
        Args:
            action: Action as [x, y, z, qx, qy, qz, qw, gripper]
            
        Returns:
            Success flag
        """
        # Extract position, orientation, and gripper command
        position = action[:3]
        orientation = action[3:7]
        gripper = action[7]
        
        # Set end effector pose
        success = self.env.set_ee_pose(position, orientation)
        
        # Set gripper state
        self.env.set_gripper(gripper)
        
        # Step simulation a few times to stabilize
        for _ in range(5):
            self.env.step_simulation()
        
        return success
    
    def plan_action(self) -> np.ndarray:
        """
        Plan next action to reach goal state
        
        Returns:
            Optimal next action as [x, y, z, qx, qy, qz, qw, gripper]
        """
        # Get current observation
        current_image = self.env.get_observation()
        current_image_tensor = self.vjepa_handler.process_image(current_image)
        current_embedding = self.vjepa_handler.encode_image(current_image_tensor)
        
        # Get current action
        current_action = self.get_current_action()
        
        # Define action bounds
        position_lower, position_upper = self.position_bounds
        
        # Full action bounds (position, orientation quaternion, gripper)
        lower_bound = np.concatenate([
            position_lower,
            [-1, -1, -1, -1],  # Quaternion bounds (will be normalized)
            [0]                # Gripper closed
        ])
        
        upper_bound = np.concatenate([
            position_upper,
            [1, 1, 1, 1],      # Quaternion bounds (will be normalized)
            [1]                # Gripper open
        ])
        
        # Optimize action sequence
        optimal_action = self.action_optimizer.optimize(
            self.vjepa_handler,
            current_embedding,
            self.goal_embedding,
            current_action,
            (lower_bound, upper_bound)
        )
        
        return optimal_action

    def normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit length
        
        Args:
            quat: Quaternion [x, y, z, w]
            
        Returns:
            Normalized quaternion
        """
        norm = np.linalg.norm(quat)
        if norm < 1e-10:
            # Default orientation if quaternion is degenerate
            return np.array([0, 0, 0, 1])
        return quat / norm
    
    def compute_goal_distance(self) -> float:
        """
        Compute distance to goal state
        
        Returns:
            Distance to goal in embedding space
        """
        current_image = self.env.get_observation()
        current_embedding = self.vjepa_handler.encode_image(current_image)
        
        distance = self.vjepa_handler.compute_embedding_distance(
            current_embedding, 
            self.goal_embedding
        )
        
        return distance
    
    def run_control_loop(self) -> Dict:
        """
        Run control loop to reach goal state
        
        Returns:
            Dictionary with control results
        """
        if self.goal_embedding is None:
            print("Error: Goal state not set")
            return {"success": False, "reason": "Goal state not set"}
        
        # Set control flag
        self.is_running = True
        
        print("Starting control loop to reach goal...")
        start_time = time.time()
        
        for step in range(self.max_steps):
            # Plan optimal action
            optimal_action = self.plan_action()
            
            # Normalize quaternion part
            optimal_action[3:7] = self.normalize_quaternion(optimal_action[3:7])
            
            # Execute action
            success = self.execute_action(optimal_action)
            
            if not success:
                print(f"Action execution failed at step {step}")
                continue
                
            # Check if goal reached
            distance = self.compute_goal_distance()
            print(f"Step {step}, Distance to goal: {distance:.4f}")
            
            if distance < self.distance_threshold:
                print(f"Goal reached at step {step} with distance {distance:.4f}")
                self.is_running = False
                return {
                    "success": True,
                    "steps": step + 1,
                    "final_distance": distance,
                    "time_taken": time.time() - start_time
                }
                
            # Control rate
            time.sleep(1.0 / self.control_hz)
        
        print(f"Failed to reach goal after {self.max_steps} steps")
        self.is_running = False
        return {
            "success": False,
            "reason": "Max steps reached",
            "final_distance": distance,
            "time_taken": time.time() - start_time
        }
