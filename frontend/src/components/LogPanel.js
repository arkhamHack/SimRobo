import React from 'react';

function LogPanel({ logEntries, onClearLog }) {
  return (
    <div className="panel log-panel">
      <h2>Execution Log</h2>
      <div className="log-container">
        <div className="log-entry" id="log-entries">
          {logEntries.map((entry, index) => (
            <div key={index} className={`log-message ${entry.type}`}>
              <div className="log-timestamp">{entry.timestamp}</div>
              <div className="log-message">{entry.message}</div>
            </div>
          ))}
        </div>
      </div>
      <button 
        id="clear-log-btn" 
        className="btn secondary"
        onClick={onClearLog}
      >
        Clear Log
      </button>
    </div>
  );
}

export default LogPanel;
