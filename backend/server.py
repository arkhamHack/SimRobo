import os
import io
import asyncio
import contextlib
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import socketio
import uvicorn
from typing import List

from backend.mujoco_environment import FrankaPandaEnv
from backend.controller import RobotController
from ml.vjepa2 import VJEPA2Handler

# Configure Socket.IO with permissive CORS settings to allow WebSocket connections
from starlette.middleware.cors import CORSMiddleware


app = FastAPI(title="V-JEPA 2 Robot Control API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*", logger=True, engineio_logger=True)


socket_app = socketio.ASGIApp(sio, app, socketio_path='/socket.io')

env = None
vjepa_handler = None
controller = None
simulation_task = None
running = False

class InitializeRequest(BaseModel):
    gui: bool = True

class InitializeRequestData(BaseModel):
    pass
    
class GoalImageRequest(BaseModel):
    image: str
    
class ControlRequest(BaseModel):
    planning_horizon: int = 5
    distance_threshold: float = 0.05
    
class RobotState(BaseModel):
    position: List[float]
    orientation: List[float]
    gripper: float


async def initialize_environment(gui=True):
    """Initialize simulation environment and V-JEPA 2 model"""
    global env, vjepa_handler, controller
    
    # Initialize environment
    env = FrankaPandaEnv(gui=gui, hz=20)
    
    # Initialize V-JEPA 2 model
    vjepa_handler = VJEPA2Handler()
    
    # Initialize controller
    controller = RobotController(
        env=env,
        vjepa_handler=vjepa_handler,
        planning_horizon=5,
        control_hz=10,
        distance_threshold=0.05,
        max_steps=100
    )
    
    print("Environment and model initialized successfully")


async def simulation_loop():
    """Main simulation loop using asyncio"""
    global running, env
    
    while running:
        if env:
            # Step simulation
            env.step_simulation()
            
            # Get current observation and send to clients
            try:
                image = env.get_observation()
                await send_observation(image)
            except Exception as e:
                print(f"Error in simulation loop: {str(e)}")
        
        # Sleep to maintain loop rate
        await asyncio.sleep(0.05)  # 20 Hz


async def send_observation(image):
    """Send observation to connected clients asynchronously"""
    # Convert numpy array to base64 string
    pil_img = Image.fromarray(image.astype('uint8'))
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Send image through Socket.IO
    await sio.emit('observation', {'image': img_str})


@app.get("/api/status")
async def get_status():
    """Get system status"""
    global env, vjepa_handler, controller, running
    
    status = {
        'environment_initialized': env is not None,
        'model_initialized': vjepa_handler is not None,
        'controller_initialized': controller is not None,
        'running': running
    }
    
    return status


@app.post('/api/initialize')
async def initialize(request_data: InitializeRequestData = None):
    """Initialize system"""
    global running, simulation_task
    
    try:
        # Initialize environment and models
        gui = True  # Default to GUI mode
        await initialize_environment(gui)
        
        # Start simulation task if not running
        if not running:
            running = True
            simulation_task = asyncio.create_task(simulation_loop())
        
        return {'success': True}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/set_goal')
async def set_goal(request: GoalImageRequest):
    """Set goal state from image"""
    global controller
    
    if not controller:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        # Decode base64 image
        img_data = base64.b64decode(request.image)
        goal_image = np.array(Image.open(io.BytesIO(img_data)))
        
        # Set goal
        controller.set_goal(goal_image)
        
        return {'success': True}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/run_control')
async def run_control(request: ControlRequest):
    """Run control loop to reach goal state"""
    global controller
    
    if not controller:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        # Update controller parameters if provided
        if request.planning_horizon:
            controller.planning_horizon = request.planning_horizon
        
        if request.distance_threshold:
            controller.distance_threshold = request.distance_threshold
        
        # Run control loop in a separate task to avoid blocking
        async def control_task():
            result = controller.run_control_loop()
            await sio.emit('control_result', result)
            
            # Send progress updates during execution
            for step in range(controller.max_steps):
                if not controller.is_running:
                    break
                    
                robot_state = {
                    'position': controller.env.get_ee_pose()[0].tolist(), 
                    'orientation': controller.env.get_ee_pose()[1].tolist(),
                    'gripper': 0.5  # Placeholder, would need proper measurement
                }
                
                await sio.emit('control_progress', {
                    'step': step,
                    'max_steps': controller.max_steps,
                    'distance': controller.compute_goal_distance(),
                    'robot_state': robot_state
                })
                
                await asyncio.sleep(0.1)  # Update rate
        
        asyncio.create_task(control_task())
        
        return {'success': True, 'message': 'Control loop started'}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/reset')
async def reset():
    """Reset environment"""
    global env
    
    if not env:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        # Reset robot and scene
        env._reset_robot()
        env._setup_scene()
        
        return {'success': True}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/shutdown')
async def shutdown():
    """Shutdown system"""
    global running, env, simulation_task
    
    try:
        # Stop simulation loop
        running = False
        
        # Cancel simulation task if running
        if simulation_task and not simulation_task.done():
            simulation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await simulation_task
        
        # Disconnect from physics server
        if env:
            env.close()
        
        return {'success': True}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add a shutdown event handler for the WebSocket
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


# Frontend routes
@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


@app.get("/{path:path}")
async def read_path(path: str):
    file_path = os.path.join("frontend", path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    else:
        return FileResponse("frontend/index.html")


# Add WebSocket endpoint for direct WebSocket communication

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection requested")
    
    try:
        # Accept the WebSocket connection
        await websocket.accept()
        print("WebSocket connection accepted successfully")
        
        while True:
            try:
                data = await websocket.receive_json()
                print(f"Received WebSocket data: {data}")
                
                # Process WebSocket messages
                if "command" in data:
                    if data["command"] == "get_observation":
                        if env:
                            try:
                                image = env.get_observation()
                                pil_img = Image.fromarray(image.astype('uint8'))
                                buffer = io.BytesIO()
                                pil_img.save(buffer, format='JPEG')
                                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                await websocket.send_json({"observation": img_str})
                            except Exception as e:
                                print(f"Error getting observation: {str(e)}")
                                await websocket.send_json({"error": str(e)})
                        else:
                            await websocket.send_json({"error": "Environment not initialized"})
            except WebSocketDisconnect:
                print("WebSocket client disconnected")
                break
            except Exception as e:
                print(f"WebSocket error: {str(e)}")
                break
    except Exception as e:
        print(f"WebSocket connection error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass


# Add 3D visualization endpoints
@app.get("/api/robot_state")
async def get_robot_state():
    """Get current robot state for 3D visualization"""
    if not env:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        # Get current robot state
        pos, orn = env.get_ee_pose()
        
        # Get joint states
        joint_states = []
        for i, joint_id in enumerate(env.joint_names):
            if joint_id >= 0:  # Valid joint
                position = env.data.qpos[joint_id]
                velocity = env.data.qvel[joint_id]
                joint_states.append({
                    "position": float(position),
                    "velocity": float(velocity),
                    "joint_reaction_forces": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Not directly accessible in MuJoCo
                    "applied_joint_motor_torque": float(env.data.ctrl[i]) if i < len(env.data.ctrl) else 0.0
                })
        
        return {
            "end_effector": {
                "position": pos.tolist(),
                "orientation": orn.tolist()
            },
            "joints": joint_states
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Start server
    uvicorn.run("backend.server:socket_app", host="0.0.0.0", port=8080, reload=True)
