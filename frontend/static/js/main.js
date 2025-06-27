// Constants and global variables
const API_URL = window.location.origin;

// Global variables
let socket;
let polling = false;
let robot3DUpdateInterval = null;
let robotViz = null; // 3D robot visualization object

// Application state
let isInitialized = false;
let isRunning = false;
let goalSet = false;

// DOM Elements
const statusElements = {
    environment: document.getElementById('env-status'),
    model: document.getElementById('model-status'),
    controller: document.getElementById('controller-status'),
    goal: document.getElementById('goal-status')
};

const buttons = {
    initialize: document.getElementById('initialize-btn'),
    reset: document.getElementById('reset-btn'),
    shutdown: document.getElementById('shutdown-btn'),
    upload: document.getElementById('upload-btn'),
    camera: document.getElementById('camera-btn'),
    run: document.getElementById('run-btn'),
    stop: document.getElementById('stop-btn'),
    clearLog: document.getElementById('clear-log-btn')
};

const images = {
    observation: document.getElementById('observation-image'),
    goal: document.getElementById('goal-image')
};

const controls = {
    planningHorizon: document.getElementById('planning-horizon'),
    distanceThreshold: document.getElementById('distance-threshold'),
    thresholdValue: document.getElementById('threshold-value'),
    progress: document.getElementById('progress'),
    progressText: document.getElementById('progress-text'),
    distanceText: document.getElementById('distance-text'),
    goalUpload: document.getElementById('goal-upload'),
    goalOverlay: document.getElementById('goal-overlay'),
    loading: document.getElementById('loading'),
    statusMessage: document.getElementById('status-message'),
    logEntries: document.getElementById('log-entries'),
    eePosition: document.getElementById('ee-position'),
    gripperState: document.getElementById('gripper-state')
};

// Socket.io event handlers
socket = io(API_URL);

socket.on('connect', () => {
    logMessage('Connected to server');
    fetchStatus();
});

socket.on('disconnect', () => {
    logMessage('Disconnected from server', 'error');
    updateStatus({
        environment_initialized: false,
        model_initialized: false,
        controller_initialized: false,
        running: false
    });
});

socket.on('observation', (data) => {
    // Update observation image
    images.observation.src = `data:image/jpeg;base64,${data.image}`;
});

socket.on('control_result', (data) => {
    isRunning = false;
    buttons.run.disabled = false;
    buttons.stop.disabled = true;

    if (data.success) {
        showStatusMessage(`Goal reached in ${data.steps} steps with distance ${data.final_distance.toFixed(4)}`, 'success');
        logMessage(`Goal reached in ${data.steps} steps (${data.time_taken.toFixed(2)}s)`);
    } else {
        showStatusMessage(`Failed to reach goal: ${data.reason}`, 'error');
        logMessage(`Failed to reach goal: ${data.reason}`, 'warning');
    }

    // Update progress
    updateProgress(100);
});

socket.on('control_progress', (data) => {
    // Update progress bar
    const progressPercent = (data.step / data.max_steps) * 100;
    updateProgress(progressPercent);

    // Update distance text
    controls.distanceText.textContent = `Distance to goal: ${data.distance.toFixed(4)}`;

    // Update robot state information
    if (data.robot_state) {
        controls.eePosition.textContent = `[${data.robot_state.position.map(p => p.toFixed(3)).join(', ')}]`;
        controls.gripperState.textContent = data.robot_state.gripper.toFixed(2);
    }
});

// API calls
async function fetchStatus() {
    try {
        const response = await fetch(`${API_URL}/api/status`);
        const data = await response.json();
        updateStatus(data);
    } catch (error) {
        console.error('Error fetching status:', error);
        showStatusMessage('Failed to connect to server', 'error');
    }
}

async function initializeSystem() {
    showLoading(true);
    try {
        const response = await fetch(`${API_URL}/api/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                gui: true
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to initialize system');
        }

        const result = await response.json();

        logMessage('System initialized successfully');
        showStatusMessage('System initialized', 'success');
        fetchStatus();

        // Start 3D robot state updates if 3D view is visible
        const container = document.getElementById('robot-3d-container');
        if (container && container.style.display !== 'none' && robotViz && robotViz.isInitialized) {
            setupRobot3DStateUpdates();
        }
    } catch (error) {
        console.error('Error initializing system:', error);
        logMessage(`Error initializing system: ${error.message}`, 'error');
        showStatusMessage('Initialization failed', 'error');
    } finally {
        showLoading(false);
    }
}

async function resetEnvironment() {
    showLoading(true);
    try {
        const response = await fetch(`${API_URL}/api/reset`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showStatusMessage('Environment reset successfully', 'success');
            logMessage('Environment reset');

            // Reset goal status
            goalSet = false;
            statusElements.goal.textContent = 'Not set';
            statusElements.goal.className = 'status-value';

            // Reset goal image
            images.goal.src = 'static/images/placeholder.png';
            controls.goalOverlay.classList.remove('hidden');

            // Reset progress
            updateProgress(0);
            controls.distanceText.textContent = 'Distance to goal: -';

            // Update buttons
            buttons.run.disabled = true;
        } else {
            showStatusMessage(`Failed to reset: ${data.error}`, 'error');
            logMessage(`Reset failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatusMessage('Failed to connect to server', 'error');
        console.error('Error resetting environment:', error);
    }

    showLoading(false);
}

async function shutdownSystem() {
    if (!confirm('Are you sure you want to shutdown the system?')) {
        return;
    }

    showLoading(true);
    try {
        const response = await fetch(`${API_URL}/api/shutdown`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showStatusMessage('System shutdown successfully', 'success');
            logMessage('System shutdown');
            updateStatus({
                environment_initialized: false,
                model_initialized: false,
                controller_initialized: false,
                running: false
            });
        } else {
            showStatusMessage(`Failed to shutdown: ${data.error}`, 'error');
            logMessage(`Shutdown failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatusMessage('Failed to connect to server', 'error');
        console.error('Error shutting down system:', error);
    }

    showLoading(false);
}

async function setGoalImage(imageData) {
    showLoading(true);
    goalSet = false;

    try {
        const response = await fetch(`${API_URL}/api/set_goal`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to set goal');
        }

        const result = await response.json();

        goalSet = true;
        updateStatusElement(statusElements.goal, true);
        logMessage('Goal image set successfully');
        showStatusMessage('Goal image set', 'success');
        buttons.run.disabled = false;
    } catch (error) {
        console.error('Error setting goal:', error);
        logMessage(`Error setting goal: ${error.message}`, 'error');
        showStatusMessage('Failed to set goal image', 'error');
        updateStatusElement(statusElements.goal, false);
    } finally {
        showLoading(false);
    }
}

async function runControlLoop() {
    if (!isInitialized || !goalSet) {
        showStatusMessage('System must be initialized and goal set before running', 'error');
        return;
    }

    if (isRunning) return;

    showLoading(true);
    isRunning = true;
    buttons.run.disabled = true;
    buttons.stop.disabled = false;

    try {
        // Get control parameters
        const planningHorizon = parseInt(controls.planningHorizon.value);
        const distanceThreshold = parseFloat(controls.distanceThreshold.value);

        const response = await fetch(`${API_URL}/api/run_control`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                planning_horizon: planningHorizon,
                distance_threshold: distanceThreshold
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to start control');
        }

        const result = await response.json();

        logMessage('Control loop started');
        showStatusMessage('Control loop running...', 'info');

        // Request 3D robot state updates if WebSocket is connected
        if (wsConnected) {
            setupRobot3DStateUpdates();
        }

    } catch (error) {
        console.error('Error starting control:', error);
        logMessage(`Error starting control: ${error.message}`, 'error');
        showStatusMessage('Failed to start control', 'error');
        isRunning = false;
        buttons.run.disabled = false;
        buttons.stop.disabled = true;
    } finally {
        showLoading(false);
    }
}

// Function to setup regular 3D robot state updates 
function setupRobot3DStateUpdates() {
    // Clear previous interval if it exists
    if (robot3DUpdateInterval) {
        clearInterval(robot3DUpdateInterval);
    }

    // Start periodic updates (10Hz)
    robot3DUpdateInterval = setInterval(async () => {
        // Only fetch if 3D viz is initialized
        if (robotViz && robotViz.isInitialized) {
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
    }, 100); // 10Hz
}

// Socket.IO is used for all communication
// No direct WebSocket connection needed

async function stopExecution() {
    showLoading(true);
    try {
        const response = await fetch(`${API_URL}/api/stop_control`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            isRunning = false;
            showStatusMessage('Control loop stopped', 'success');
            logMessage('Control loop stopped');

            // Update buttons
            buttons.run.disabled = false;
            buttons.stop.disabled = true;
        } else {
            showStatusMessage(`Failed to stop control: ${data.error}`, 'error');
            logMessage(`Control stop failed: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatusMessage('Failed to connect to server', 'error');
        console.error('Error stopping control loop:', error);
    }

    showLoading(false);
}

// Helper functions
function updateStatus(data) {
    isInitialized = data.environment_initialized && data.model_initialized && data.controller_initialized;
    isRunning = data.running;

    // Update status indicators
    updateStatusElement(statusElements.environment, data.environment_initialized);
    updateStatusElement(statusElements.model, data.model_initialized);
    updateStatusElement(statusElements.controller, data.controller_initialized);

    // Update buttons based on status
    buttons.initialize.disabled = isInitialized;
    buttons.reset.disabled = !isInitialized || isRunning;
    buttons.shutdown.disabled = !isInitialized;
    buttons.run.disabled = !isInitialized || !goalSet || isRunning;
    buttons.stop.disabled = !isRunning;
}

function updateStatusElement(element, isActive) {
    if (isActive) {
        element.textContent = 'Active';
        element.className = 'status-value active';
    } else {
        element.textContent = 'Not initialized';
        element.className = 'status-value';
    }
}

function showLoading(show) {
    if (show) {
        controls.loading.classList.add('active');
    } else {
        controls.loading.classList.remove('active');
    }
}

function showStatusMessage(message, type) {
    controls.statusMessage.textContent = message;
    controls.statusMessage.className = 'status-message ' + type;

    // Hide message after 5 seconds
    setTimeout(() => {
        controls.statusMessage.className = 'status-message';
    }, 5000);
}

function updateProgress(percent) {
    const clampedPercent = Math.min(100, Math.max(0, percent));
    controls.progress.style.width = `${clampedPercent}%`;
    controls.progressText.textContent = `${Math.round(clampedPercent)}%`;
}

function logMessage(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;

    const timestampElement = document.createElement('div');
    timestampElement.className = 'log-timestamp';
    timestampElement.textContent = timestamp;

    const messageElement = document.createElement('div');
    messageElement.className = 'log-message';
    messageElement.textContent = message;

    logEntry.appendChild(timestampElement);
    logEntry.appendChild(messageElement);

    controls.logEntries.appendChild(logEntry);
    controls.logEntries.scrollTop = controls.logEntries.scrollHeight;
}

function clearLog() {
    controls.logEntries.innerHTML = '';
    logMessage('Log cleared');
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (event) {
        const imageData = event.target.result.split(',')[1];  // Get base64 part
        images.goal.src = `data:${file.type};base64,${imageData}`;
        controls.goalOverlay.classList.add('hidden');

        // Send to server
        setGoalImage(imageData);
    };
    reader.readAsDataURL(file);
}

function captureCurrentView() {
    // Use the current observation as goal
    const currentImage = images.observation.src;
    if (currentImage && !currentImage.includes('placeholder.png')) {
        const imageData = currentImage.split(',')[1];  // Get base64 part
        images.goal.src = currentImage;
        controls.goalOverlay.classList.add('hidden');

        // Send to server
        setGoalImage(imageData);
    } else {
        showStatusMessage('No observation available to capture', 'error');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function () {
    // Connect to Socket.IO for all communication
    socket = io(API_URL);

    // Setup event handlers
    buttons.initialize.addEventListener('click', initializeSystem);
    buttons.reset.addEventListener('click', resetEnvironment);
    buttons.shutdown.addEventListener('click', shutdownSystem);
    buttons.upload.addEventListener('click', () => controls.goalUpload.click());
    buttons.camera.addEventListener('click', captureCurrentView);
    buttons.run.addEventListener('click', runControlLoop);
    buttons.stop.addEventListener('click', stopExecution);
    buttons.clearLog.addEventListener('click', clearLog);

    // Add 3D visualization toggle button handler
    const toggleBtn = document.getElementById('toggle-3d-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function () {
            const container = document.getElementById('robot-3d-container');

            // If robot visualization exists but not initialized, initialize it
            if (robotViz && !robotViz.isInitialized) {
                robotViz.init();
                toggleBtn.textContent = 'Hide 3D View';
                container.style.display = 'block';
                setupRobot3DStateUpdates(); // Start robot state updates
            } else if (robotViz) {
                // Toggle visibility
                if (container.style.display === 'none') {
                    container.style.display = 'block';
                    toggleBtn.textContent = 'Hide 3D View';
                    setupRobot3DStateUpdates(); // Restart robot state updates
                } else {
                    container.style.display = 'none';
                    toggleBtn.textContent = 'Show 3D View';
                    // Stop updates to save resources
                    if (robot3DUpdateInterval) {
                        clearInterval(robot3DUpdateInterval);
                        robot3DUpdateInterval = null;
                    }
                }
            }
        });
    }

    // File upload handler
    controls.goalUpload.addEventListener('change', handleFileUpload);

    // Goal overlay click handler
    controls.goalOverlay.addEventListener('click', () => controls.goalUpload.click());

    // Range input handler
    controls.distanceThreshold.addEventListener('input', (e) => {
        controls.thresholdValue.textContent = e.target.value;
    });

    // Drag and drop for goal image
    const goalContainer = document.querySelector('.image-container');

    goalContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        goalContainer.classList.add('dragover');
    });

    goalContainer.addEventListener('dragleave', () => {
        goalContainer.classList.remove('dragover');
    });

    goalContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        goalContainer.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function (event) {
                const imageData = event.target.result.split(',')[1];
                images.goal.src = `data:${file.type};base64,${imageData}`;
                controls.goalOverlay.classList.add('hidden');

                // Send to server
                setGoalImage(imageData);
            };
            reader.readAsDataURL(file);
        }
    });
});
