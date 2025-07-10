import os
import io
import base64
from typing import Optional
import tempfile

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import socketio
import uvicorn
import cv2

from ml.vjepa2 import VJEPA2Handler
from backend.trajectory_predictor import TrajectoryPredictor


app = FastAPI(title="V-JEPA 2 Trajectory Prediction API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Socket.IO with specific allowed origins
sio = socketio.AsyncServer(
    async_mode="asgi",     
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "*"  # Allow all origins for testing
    ],
    logger=True,
    engineio_logger=True,
    ping_timeout=20000,
    ping_interval=5000,
    max_http_buffer_size=1e6,
)

# Mount Socket.IO to FastAPI
app.mount("/socket.io", socketio.ASGIApp(sio))

# Keep the socket_app for backwards compatibility
socket_app = app


# Global variables
vjepa_handler = None
trajectory_predictor = None
video_frames = {}  # Dictionary to store video frames by ID


# Models for API requests and responses
class TextQueryRequest(BaseModel):
    text: str
    video_id: str


class TrajectoryPredictionRequest(BaseModel):
    text_query: str
    video_id: str
    planning_horizon: int = 10
    confidence_threshold: float = 0.7


class VideoInfo(BaseModel):
    video_id: str
    num_frames: int
    width: int
    height: int
    fps: Optional[float] = None


async def initialize_vjepa():
    """Initialize V-JEPA 2 handler"""
    global vjepa_handler, trajectory_predictor
    vjepa_handler = VJEPA2Handler()
    trajectory_predictor = TrajectoryPredictor(
        vjepa_handler=vjepa_handler,
        planning_horizon=10,
        confidence_threshold=0.7,
        max_frames=30
    )
    print("V-JEPA 2 handler initialized successfully")


def extract_video_frames(video_path, max_frames=100):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit frames
    num_frames = min(total_frames, max_frames)
    
    # Extract frames
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    return frames, width, height, fps, num_frames


def encode_frame_to_base64(frame):
    """Convert frame to base64 encoded JPEG"""
    # Convert numpy array to PIL Image
    image = Image.fromarray(frame)
    
    # Save to buffer
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


@app.on_event("startup")
async def startup_event():
    await initialize_vjepa()


@app.get("/api/status")
async def get_status():
    return {
        'vjepa_initialized': vjepa_handler is not None,
        'trajectory_predictor_initialized': trajectory_predictor is not None,
        'num_videos_loaded': len(video_frames)
    }


@app.post('/api/upload_video', response_model=VideoInfo)
async def upload_video(file: UploadFile = File(...), max_frames: int = Form(100)):
    """Upload video file for trajectory prediction"""
    if not vjepa_handler:
        await initialize_vjepa()
    
    try:
        # Save uploaded file to temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file_path = temp_file.name
        
        # Write uploaded file to temporary file
        with open(temp_file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Extract frames from video
        frames, width, height, fps, num_frames = extract_video_frames(temp_file_path, max_frames)
        
        if not frames:
            os.unlink(temp_file_path)  # Clean up
            raise HTTPException(status_code=400, detail="Failed to extract frames from video")
        
        # Generate unique ID for this video
        import uuid
        video_id = str(uuid.uuid4())
        
        # Store frames for later use
        video_frames[video_id] = frames
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return VideoInfo(
            video_id=video_id,
            num_frames=len(frames),
            width=width,
            height=height,
            fps=fps
        )
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/predict_trajectory')
async def predict_trajectory(request: TrajectoryPredictionRequest):
    """Predict object trajectory based on text description"""
    if not vjepa_handler:
        await initialize_vjepa()
    
    if request.video_id not in video_frames:
        raise HTTPException(status_code=404, detail="Video not found. Upload a video first.")
    
    try:
        # Update trajectory predictor parameters
        trajectory_predictor.planning_horizon = request.planning_horizon
        trajectory_predictor.confidence_threshold = request.confidence_threshold
        
        # Get frames for the video
        frames = video_frames[request.video_id]
        
        # Predict trajectory
        result = trajectory_predictor.predict_trajectory(frames, request.text_query)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Encode visualization frames to base64
        viz_frames_base64 = []
        for frame in result["visualization_frames"]:
            viz_frames_base64.append(encode_frame_to_base64(frame))
        
        # Format response
        response = {
            "success": True,
            "text_query": request.text_query,
            "observed_trajectory": result["observed_trajectory"],
            "predicted_trajectory": result["predicted_trajectory"],
            "visualization_frames": viz_frames_base64
        }
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/video_frame/{video_id}/{frame_index}')
async def get_video_frame(video_id: str, frame_index: int):
    """Get a specific frame from a video"""
    if video_id not in video_frames:
        raise HTTPException(status_code=404, detail="Video not found")
    
    frames = video_frames[video_id]
    
    if frame_index < 0 or frame_index >= len(frames):
        raise HTTPException(status_code=400, detail="Invalid frame index")
    
    # Get the requested frame
    frame = frames[frame_index]
    
    # Convert to base64
    img_str = encode_frame_to_base64(frame)
    
    return {"frame": img_str}


# Socket.IO events
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


@app.get("/{path:path}")
async def read_path(path: str):
    file_path = os.path.join("frontend", path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse("frontend/index.html")


# Uvicorn entry
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server_trajectory:socket_app", host="0.0.0.0", port=8080, reload=True)
