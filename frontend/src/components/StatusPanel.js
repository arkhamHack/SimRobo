import React from 'react';

function StatusPanel({ 
  status, 
  loading, 
  statusMessage, 
  statusType, 
  onInitialize, 
  onReset, 
  onShutdown 
}) {
  return (
    <div className="panel status-panel">
      <h2>System Status</h2>
      <div className="status-container">
        <div className="status-item">
          <span className="status-label">Environment:</span>
          <span className={`status-value ${status.environment ? 'active' : ''}`}>
            {status.environment ? 'Initialized' : 'Not initialized'}
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">V-JEPA 2 Model:</span>
          <span className={`status-value ${status.model ? 'active' : ''}`}>
            {status.model ? 'Loaded' : 'Not loaded'}
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Controller:</span>
          <span className={`status-value ${status.controller ? 'active' : ''}`}>
            {status.controller ? 'Initialized' : 'Not initialized'}
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Goal:</span>
          <span className={`status-value ${status.goal ? 'active' : ''}`}>
            {status.goal ? 'Set' : 'Not set'}
          </span>
        </div>
      </div>
      <div className="button-container">
        <button 
          className="btn primary" 
          onClick={onInitialize}
          disabled={status.running}
        >
          Initialize System
        </button>
        <button 
          className="btn secondary" 
          onClick={onReset}
          disabled={!status.environment || status.running}
        >
          Reset Environment
        </button>
        <button 
          className="btn danger" 
          onClick={onShutdown}
          disabled={!status.environment || status.running}
        >
          Shutdown
        </button>
      </div>
      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <span>Processing...</span>
        </div>
      )}
      {statusMessage && (
        <div className={`status-message ${statusType}`}>
          {statusMessage}
        </div>
      )}
    </div>
  );
}

export default StatusPanel;
