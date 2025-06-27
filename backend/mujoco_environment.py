import os
import numpy as np
from PIL import Image
import time
import mujoco
from mujoco import viewer
import io

class FrankaPandaEnv:
    """
    MuJoCo simulation environment for Franka Panda robot
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
        
        # Load the model from XML
        self.model_path = os.path.join(os.path.dirname(__file__), '../assets/franka_panda.xml')
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize the viewer if gui is enabled
        self.viewer = None
        if self.gui:
            self.viewer = viewer.launch_passive(self.model, self.data)
        
        # Set simulation timestep
        self.model.opt.timestep = self.dt
        
        # Robot parameters
        self._init_robot_params()
        
        # Initialize robot
        self._reset_robot()
        
        # Set up camera parameters
        self.camera_height = 256
        self.camera_width = 256
        
        # Set up scene
        self._setup_scene()
    
    def _init_robot_params(self):
        """Initialize robot parameters and find relevant joints and bodies"""
        # Find relevant joint and body IDs
        self.joint_names = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                           for name in ["joint1", "joint2", "joint3", "joint4", 
                                       "joint5", "joint6", "joint7", 
                                       "finger_joint1", "finger_joint2"]]
        
        # Find end effector body ID
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        
        # Store home position
        self.home_positions = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.04, 0.04]
        
        # Store joint ranges
        self.joint_ranges = []
        for i in range(len(self.joint_names)):
            jnt_id = self.joint_names[i]
            if jnt_id >= 0:  # Valid joint
                lower = self.model.jnt_range[jnt_id][0]
                upper = self.model.jnt_range[jnt_id][1]
                self.joint_ranges.append((lower, upper))
        
        # Identify objects in the scene
        self.objects = []
    
    def _reset_robot(self):
        """Reset robot to home position"""
        # Reset joint positions to home
        for i, jnt_id in enumerate(self.joint_names):
            if jnt_id >= 0:  # Valid joint
                self.data.qpos[jnt_id] = self.home_positions[i]
        
        # Reset velocities
        self.data.qvel[:] = 0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
    
    def _setup_scene(self):
        """Set up objects in the scene"""
        # The scene setup is primarily done in the XML file
        # But we can add additional objects or modify existing ones here
        pass
    
    def step_simulation(self):
        """Step the simulation forward"""
        mujoco.mj_step(self.model, self.data)
        
        if self.gui and self.viewer is not None:
            self.viewer.sync()
            time.sleep(self.dt)  # Sleep to maintain simulation rate
    
    def get_observation(self):
        """Get RGB observation from camera"""
        # Allocate arrays for the image
        rgb_arr = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        depth_arr = np.zeros((self.camera_height, self.camera_width), dtype=np.float32)
        
        # Render the camera view
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.model, self.data)
        viewport = mujoco.MjrRect(0, 0, self.camera_width, self.camera_height)
        
        # Get camera configuration
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
        
        if camera_id == -1:  # If named camera not found, use free camera
            cam_pos = [1.0, 1.0, 1.0]
            cam_target = [0.0, 0.0, 0.3]
            mujoco.mjv_defaultCamera(self.model, self.data, viewport, camera_id)
        
        # Render scene and get image
        mujoco.mjr_render(viewport, self.model, self.data, mujoco.MjrContext())
        mujoco.mjr_readPixels(rgb_arr, depth_arr, viewport, self.model, self.data, mujoco.MjrContext())
        
        return rgb_arr
    
    def get_ee_pose(self):
        """Get end effector pose"""
        # Get world position and orientation of the end effector
        pos = self.data.xpos[self.ee_body_id]
        quat = self.data.xquat[self.ee_body_id]
        
        return np.array(pos), np.array(quat)
    
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
        
        # Set up IK problem in MuJoCo
        # This is a simplified approach; a more robust solution would use mujoco.IK
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if ee_site_id < 0:
            print("Warning: end effector site not found in model")
            return False
            
        # Current approach: use the MuJoCo built-in IK solver
        # Store current state
        qpos_before = self.data.qpos.copy()
        
        # Set up IK target
        self.data.mocap_pos[0] = target_pos
        self.data.mocap_quat[0] = target_orn
        
        # Run IK
        for _ in range(max_iters):
            mujoco.mj_step(self.model, self.data)
        
        # Get resulting joint positions
        joint_positions = []
        for i, jnt_id in enumerate(self.joint_names):
            if jnt_id >= 0 and i < 7:  # Only the first 7 joints (arm, not gripper)
                joint_positions.append(self.data.qpos[jnt_id])
        
        # Apply the joint positions using position control
        for i, jnt_id in enumerate(self.joint_names):
            if jnt_id >= 0 and i < 7:
                self.data.ctrl[jnt_id] = joint_positions[i]
        
        # Check if the target was reached
        mujoco.mj_forward(self.model, self.data)
        pos, _ = self.get_ee_pose()
        distance = np.linalg.norm(pos - target_pos)
        
        # If target wasn't reached accurately enough, restore original state
        if distance > 0.01:  # 1 cm threshold
            self.data.qpos[:] = qpos_before
            mujoco.mj_forward(self.model, self.data)
            return False
            
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
        
        # Find gripper joint indices
        finger_joint1_id = self.joint_names[-2]
        finger_joint2_id = self.joint_names[-1]
        
        if finger_joint1_id >= 0 and finger_joint2_id >= 0:
            # Set gripper joints
            self.data.ctrl[finger_joint1_id] = target_pos
            self.data.ctrl[finger_joint2_id] = target_pos
    
    def close(self):
        """Close environment and release resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
