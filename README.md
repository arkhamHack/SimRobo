# Zero-Shot Robotic Arm Control Using V-JEPA 2

This project implements a closed-loop model-predictive control system for a simulated 7-DoF robotic arm (Franka Emika Panda) using Meta's V-JEPA 2 model. The system enables zero-shot robotic control through visual prediction without any task-specific training.

## Project Structure

```
project/
├── backend/          # Simulation and control backend using PyBullet
├── frontend/         # Web interface for visualization and goal setting
├── ml/               # V-JEPA 2 model integration and prediction
├── requirements.txt  # Dependencies
└── README.md         # Project documentation
```

## Features

- Zero-shot robotic arm control using visual goal images
- Physics-based simulation of a Franka Panda robot arm
- Cross-Entropy Method (CEM) for action optimization
- V-JEPA 2 for visual embedding and future state prediction
- Web interface for monitoring and goal specification

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the System

1. Start the server:
   ```
   python backend/server.py
   ```
2. Open the web interface:
   ```
   python frontend/app.py
   ```

## System Architecture

The system consists of three main components:

1. **Backend**: Handles the physics simulation, robot control, and planning logic
2. **ML Module**: Incorporates the V-JEPA 2 model for visual embeddings and future state prediction
3. **Frontend**: Provides a user interface to specify goals and monitor progress

## Components Explanation

- **Simulation Environment**: PyBullet physics engine simulating a Franka Panda arm
- **V-JEPA 2 Model**: Pretrained encoder converts RGB frames to latent embeddings
- **Action Optimization**: Cross-Entropy Method (CEM) optimizes actions to reach visual goals
- **Planning Loop**: Predicts future states based on candidate actions and selects optimal trajectory

## License

This project is for research and demonstration purposes only.

## Acknowledgments

- Meta AI Research for the V-JEPA 2 model
- PyBullet team for the physics simulation framework
