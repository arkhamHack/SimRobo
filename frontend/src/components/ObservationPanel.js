import React from 'react';

function ObservationPanel({ observationImage, robotState }) {
  return (
    <div className="panel view-panel">
      <h2>Current Observation</h2>
      <div className="image-container">
        <img src={observationImage} alt="Current observation" />
      </div>
      <div className="info-container">
        <div className="info-item">
          <span className="info-label">End Effector Position:</span>
          <span className="info-value">
            [{robotState.position.map(p => p.toFixed(3)).join(', ')}]
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">Gripper State:</span>
          <span className="info-value">{robotState.gripper.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}

export default ObservationPanel;
