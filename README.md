# V-JEPA2 Trajectory Prediction System

This project implements an advanced trajectory prediction system using Meta's V-JEPA2 vision model with language alignment capabilities. The system enables text-based object selection and tracking, followed by trajectory prediction across video frames.

## Project Structure

```
project/
├── backend/          # Server API, trajectory prediction, and video processing
├── frontend/         # Modern React UI with Material UI components
├── ml/               # V-JEPA2 and LLM integration for vision-language alignment
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## Features

- **Text-to-Vision Alignment**: Integrates V-JEPA2 vision model with Mistral LLM for precise text-based object selection
- **Object Tracking**: Identifies and tracks objects based on natural language descriptions
- **Trajectory Prediction**: Uses Cross-Entropy Method (CEM) to predict future object trajectories
- **Modern UI**: Sleek dark-themed interface with intuitive controls and real-time visualization
- **Video Processing**: Handles video uploads and extracts frames for processing
- **Real-time Updates**: Provides frame-by-frame playback of observed and predicted trajectories

## Technical Stack

### Backend
- **V-JEPA2**: Facebook's state-of-the-art vision model (`facebook/vjepa2-vitl-fpc64-256`)
- **Mistral LLM**: Advanced language model for text encoding and semantically rich embeddings
- **PyTorch**: Deep learning framework for model inference
- **HuggingFace Transformers**: For model loading and processing
- **OpenCV**: For video processing and object tracking
- **Flask/SocketIO**: For API endpoints and real-time communication

### Frontend
- **React**: Modern component-based UI framework
- **Material UI v5**: Responsive design components with dark theme
- **Axios**: For API communication

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```
   cd frontend
   npm install
   ```

## Running the System

1. Start the backend server:
   ```
   python backend/server_trajectory.py
   ```
2. In a separate terminal, start the frontend development server:
   ```
   cd frontend
   npm start
   ```
3. Access the web interface at `http://localhost:3000`

## Usage

1. Upload a video through the UI
2. Enter a text description of the object you want to track (e.g., "red ball", "person in blue shirt")
3. Configure prediction parameters (horizon length, confidence threshold)
4. Click "Predict Trajectory"
5. Use playback controls to view frame-by-frame results
6. View the predicted trajectory overlaid on the video

## How It Works

1. The system encodes the user's text description using the Mistral LLM
2. Text embeddings are projected to align with V-JEPA2's visual embedding space
3. Each video frame is processed through V-JEPA2 to extract visual embeddings
4. Similarity maps between text and visual embeddings identify the target object
5. The object is tracked across frames using both visual features and traditional tracking
6. CEM optimization predicts the object's future trajectory based on observed motion
7. Results are visualized in the UI with both observed and predicted paths
