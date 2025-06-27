import os
import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
import time

class FrankaPandaEnv:
    """
    PyBullet simulation environment for Franka Panda robot
    """
    def __init__(self, gui=True, hz=20):
        """
        Initialize the simulation environment
        
        Args:
            gui: Whether to use GUI (True) or headless mode (False)
            hz: Control frequency in Hz
        """
        self.gui = gui
        self.dt = 1.0 / hz
        
        # Connect to physics server
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Robot parameters
        self.num_joints = p.getNumJoints(self.robot_id)
        self.ee_index = 7  # End effector link index
        
        # Initialize robot joints and end effector
        self._reset_robot()
        
        # Set up camera parameters
        self.camera_height = 256
        self.camera_width = 256
        self.camera_position = [1.0, 1.0, 1.0]
        self.camera_target = [0.0, 0.0, 0.3]
        self.camera_up_vector = [0, 0, 1]
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up_vector
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0
        )
        
        # Add additional objects
        self._setup_scene()
    
    def _reset_robot(self):
        """Reset robot to home position"""
        self.home_positions = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0]
        
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, self.home_positions[i])
            
        # Disable motor control for gripper
        p.setJointMotorControl2(self.robot_id, 9, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.robot_id, 10, p.VELOCITY_CONTROL, force=0)
    
    def _setup_scene(self):
        """Set up objects in the scene"""
        # Add a table
        table_pos = [0.5, 0, 0]
        self.table_id = p.loadURDF("table/table.urdf", table_pos, useFixedBase=True)
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
        
        # Add some manipulable objects
        self.objects = []
        
        # Add a red cube
        cube_pos = [0.5, 0, 0.65]
        cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_pos, cube_orientation)
        p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
        self.objects.append({"id": cube_id, "name": "red_cube"})
    
    def step_simulation(self):
        """Step the simulation forward"""
        p.stepSimulation()
        time.sleep(self.dt)
    
    def get_observation(self):
        """Get RGB observation from camera"""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # Drop alpha channel
        return rgb_array
    
    def get_ee_pose(self):
        """Get end effector pose"""
        state = p.getLinkState(self.robot_id, self.ee_index)
        pos = state[0]
        orn = state[1]
        return np.array(pos), np.array(orn)
    
    def set_ee_pose(self, target_pos, target_orn=None, max_iters=100):
        """
        Set end effector pose using inverse kinematics
        
        Args:
            target_pos: Target position [x, y, z]
            target_orn: Target orientation as quaternion [x, y, z, w], if None, current orientation is kept
            max_iters: Maximum iterations for IK
        
        Returns:
            success: Whether IK solution was found
        """
        if target_orn is None:
            _, target_orn = self.get_ee_pose()
        
        # Compute inverse kinematics
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos,
            target_orn,
            maxNumIterations=max_iters
        )
        
        # Set joint positions (first 7 joints, excluding gripper)
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )
        
        return True
    
    def set_gripper(self, width):
        """
        Set gripper width
        
        Args:
            width: Gripper width (0 = closed, 1 = open)
        """
        # Normalize width to joint range
        width = np.clip(width, 0, 1)
        target_pos = width * 0.04  # Max opening is 4cm
        
        # Control fingers to target position
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=target_pos)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=target_pos)
    
    def close(self):
        """Disconnect from physics server"""
        p.disconnect(self.physics_client)
