// 3D visualization for V-JEPA 2 Robot Control
// Uses Three.js for robot arm visualization

class RobotVisualization {
    constructor(containerElement) {
        this.container = containerElement;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.robot = null;
        this.controls = null;
        this.isInitialized = false;
        this.robotParts = {};
        this.jointPositions = [];
        this.linkMeshes = [];
    }

    init() {
        if (this.isInitialized) return;
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        // Create camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(1, 1, 1);
        this.camera.lookAt(0, 0, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.25;
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Add a grid for reference
        const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
        
        // Create robot placeholder
        this.createRobotModel();
        
        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(0.5);
        this.scene.add(axesHelper);
        
        // Start animation loop
        this.animate();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        this.isInitialized = true;
    }
    
    createRobotModel() {
        // Create a placeholder robot model
        // This will be replaced with actual robot data when available
        this.robot = new THREE.Group();
        this.scene.add(this.robot);
        
        // Create a simple base
        const baseGeometry = new THREE.CylinderGeometry(0.2, 0.2, 0.1, 32);
        const baseMaterial = new THREE.MeshStandardMaterial({ color: 0x7777ff });
        const base = new THREE.Mesh(baseGeometry, baseMaterial);
        base.position.y = 0.05;
        this.robot.add(base);
        this.robotParts.base = base;
        
        // Set up initial joint positions for a 7-DOF robot arm
        // These positions will be updated with actual data from the backend
        this.jointPositions = [
            new THREE.Vector3(0, 0.1, 0),  // Base/shoulder joint
            new THREE.Vector3(0, 0.3, 0),  // Shoulder
            new THREE.Vector3(0, 0.5, 0),  // Upper arm
            new THREE.Vector3(0, 0.7, 0),  // Elbow
            new THREE.Vector3(0, 0.9, 0),  // Lower arm
            new THREE.Vector3(0, 1.1, 0),  // Wrist
            new THREE.Vector3(0, 1.3, 0),  // End effector
        ];
        
        // Create primitive links between joints
        this.updateRobotLinks();
    }
    
    updateRobotLinks() {
        // Remove existing links
        this.linkMeshes.forEach(mesh => {
            this.robot.remove(mesh);
        });
        this.linkMeshes = [];
        
        // Create new links between each joint position
        for (let i = 0; i < this.jointPositions.length - 1; i++) {
            const start = this.jointPositions[i];
            const end = this.jointPositions[i + 1];
            
            // Create a joint sphere
            const jointGeometry = new THREE.SphereGeometry(0.03, 16, 16);
            const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xff3333 });
            const joint = new THREE.Mesh(jointGeometry, jointMaterial);
            joint.position.copy(start);
            this.robot.add(joint);
            this.linkMeshes.push(joint);
            
            // Create a cylinder between joints
            const direction = new THREE.Vector3().subVectors(end, start);
            const length = direction.length();
            const cylinderGeometry = new THREE.CylinderGeometry(0.02, 0.02, length, 8);
            const cylinderMaterial = new THREE.MeshStandardMaterial({ color: 0xcccccc });
            const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);
            
            // Position and rotate cylinder to connect joints
            cylinder.position.copy(start);
            cylinder.position.addScaledVector(direction, 0.5);
            
            // Orient cylinder along direction vector
            const quaternion = new THREE.Quaternion();
            const upVector = new THREE.Vector3(0, 1, 0);
            quaternion.setFromUnitVectors(upVector, direction.clone().normalize());
            cylinder.quaternion.copy(quaternion);
            
            this.robot.add(cylinder);
            this.linkMeshes.push(cylinder);
        }
        
        // Add end effector
        const endEffectorGeometry = new THREE.BoxGeometry(0.05, 0.08, 0.12);
        const endEffectorMaterial = new THREE.MeshStandardMaterial({ color: 0x33ff33 });
        const endEffector = new THREE.Mesh(endEffectorGeometry, endEffectorMaterial);
        endEffector.position.copy(this.jointPositions[this.jointPositions.length - 1]);
        this.robot.add(endEffector);
        this.linkMeshes.push(endEffector);
    }
    
    updateRobotState(robotState) {
        if (!this.isInitialized || !robotState) return;
        
        // Update end effector position and orientation
        if (robotState.end_effector) {
            const pos = robotState.end_effector.position;
            const orn = robotState.end_effector.orientation;
            
            // Update the end effector position (last joint)
            if (this.jointPositions.length > 0) {
                const lastIdx = this.jointPositions.length - 1;
                this.jointPositions[lastIdx].set(pos[0], pos[1], pos[2]);
            }
            
            // If we have detailed joint data, update all joints
            if (robotState.joints && robotState.joints.length > 0) {
                // In a real implementation, use the full kinematic chain
                // For now, we'll just fake it by interpolating positions
                // between base and end-effector
                const numJoints = Math.min(this.jointPositions.length - 2, robotState.joints.length);
                
                for (let i = 1; i <= numJoints; i++) {
                    const t = i / (numJoints + 1);
                    const basePos = this.jointPositions[0];
                    const endPos = this.jointPositions[this.jointPositions.length - 1];
                    
                    this.jointPositions[i].x = basePos.x + t * (endPos.x - basePos.x);
                    this.jointPositions[i].y = basePos.y + t * (endPos.y - basePos.y);
                    this.jointPositions[i].z = basePos.z + t * (endPos.z - basePos.z);
                    
                    // Add some bend based on joint positions
                    const jointAngle = robotState.joints[i-1].position;
                    const bendFactor = 0.1;
                    
                    // Apply a simple bend in the xz plane based on joint angle
                    if (i % 2 === 0) {
                        this.jointPositions[i].x += Math.sin(jointAngle) * bendFactor;
                    } else {
                        this.jointPositions[i].z += Math.sin(jointAngle) * bendFactor;
                    }
                }
            }
            
            // Update the visual representation
            this.updateRobotLinks();
        }
    }
    
    onWindowResize() {
        if (!this.isInitialized) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    animate() {
        if (!this.isInitialized) return;
        
        requestAnimationFrame(() => this.animate());
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Global robot visualization instance
let robotViz = null;

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    const vizContainer = document.getElementById('robot-3d-container');
    if (vizContainer) {
        robotViz = new RobotVisualization(vizContainer);
        
        // Initialize button
        const viz3dButton = document.getElementById('toggle-3d-btn');
        if (viz3dButton) {
            viz3dButton.addEventListener('click', () => {
                if (!robotViz.isInitialized) {
                    robotViz.init();
                    viz3dButton.textContent = 'Hide 3D View';
                    vizContainer.style.display = 'block';
                } else {
                    viz3dButton.textContent = 'Show 3D View';
                    vizContainer.style.display = viz3dButton.textContent.includes('Show') ? 'none' : 'block';
                }
            });
        }
    }
});

// Function to update robot state from API
async function updateRobotStateFrom3DAPI() {
    if (!robotViz || !robotViz.isInitialized) return;
    
    try {
        const response = await fetch(`${API_URL}/api/robot_state`);
        if (response.ok) {
            const robotState = await response.json();
            robotViz.updateRobotState(robotState);
        }
    } catch (error) {
        console.error('Error fetching robot state:', error);
    }
}

// Function to start periodic robot state updates
function startRobot3DUpdates() {
    if (robotViz && robotViz.isInitialized) {
        // Update every 100ms (10Hz)
        setInterval(updateRobotStateFrom3DAPI, 100);
    }
}
