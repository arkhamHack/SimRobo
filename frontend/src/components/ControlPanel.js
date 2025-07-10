import React, { useState } from 'react';

function ControlPanel({ status, progress, distance, onRun, onStop }) {
  const [planningHorizon, setPlanningHorizon] = useState(5);
  const [distanceThreshold, setDistanceThreshold] = useState(0.05);
  
  const handleDistanceThresholdChange = (e) => {
    setDistanceThreshold(e.target.value);
  };
  
  const handlePlanningHorizonChange = (e) => {
    setPlanningHorizon(e.target.value);
  };
  
  return (
    <div className="panel control-panel">
      <h2>Control</h2>
      <div className="control-options">
        <div className="option">
          <label htmlFor="planning-horizon">Planning Horizon:</label>
          <input 
            type="number" 
            id="planning-horizon" 
            min="1" 
            max="10" 
            value={planningHorizon}
            onChange={handlePlanningHorizonChange}
          />
        </div>
        <div className="option">
          <label htmlFor="distance-threshold">Distance Threshold:</label>
          <input 
            type="range" 
            id="distance-threshold" 
            min="0.01" 
            max="0.2" 
            step="0.01" 
            value={distanceThreshold}
            onChange={handleDistanceThresholdChange}
          />
          <span id="threshold-value">{distanceThreshold}</span>
        </div>
      </div>
      <div className="button-container">
        <button 
          id="run-btn" 
          className="btn primary" 
          onClick={onRun}
          disabled={!status.environment || !status.model || !status.controller || !status.goal || status.running}
        >
          Run Control Loop
        </button>
        <button 
          id="stop-btn" 
          className="btn danger" 
          onClick={onStop}
          disabled={!status.running}
        >
          Stop Execution
        </button>
      </div>
      <div className="progress-container">
        <div className="progress-bar" id="progress-bar">
          <div 
            className="progress" 
            id="progress"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="progress-info">
          <span id="progress-text">{Math.round(progress)}%</span>
          <span id="distance-text">Distance to goal: {distance}</span>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;
