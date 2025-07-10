import React, { useState, useEffect, useRef } from 'react';

function Visualization3D() {
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef(null);
  const robotVizRef = useRef(null);
  const interval3DRef = useRef(null);
  
  const toggleVisibility = () => {
    // If 3D visualization doesn't exist, initialize it
    if (!robotVizRef.current) {
      // This would be initialized with your 3D visualization code
      // For now, we'll just toggle visibility
      setIsVisible(!isVisible);
    } else {
      setIsVisible(!isVisible);
      
      if (!isVisible) {
        // Setup robot state updates when visible
        setupRobot3DStateUpdates();
      } else {
        // Stop updates to save resources
        if (interval3DRef.current) {
          clearInterval(interval3DRef.current);
          interval3DRef.current = null;
        }
      }
    }
  };
  
  const setupRobot3DStateUpdates = () => {
    // This would connect to your backend to get robot state updates
    // For now, just a placeholder
    if (interval3DRef.current) {
      clearInterval(interval3DRef.current);
    }
    
    interval3DRef.current = setInterval(() => {
      // This would update the 3D visualization with latest robot state
      // For now, it's just a placeholder
    }, 100); // Update every 100ms
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (interval3DRef.current) {
        clearInterval(interval3DRef.current);
      }
    };
  }, []);

  return (
    <div className="panel viz3d-panel">
      <h2>
        3D Robot Visualization 
        <button 
          id="toggle-3d-btn" 
          className="btn secondary"
          onClick={toggleVisibility}
        >
          {isVisible ? 'Hide 3D View' : 'Show 3D View'}
        </button>
      </h2>
      <div 
        id="robot-3d-container" 
        ref={containerRef}
        className="robot-3d-container" 
        style={{ display: isVisible ? 'block' : 'none' }}
      >
        {/* 3D visualization will be rendered here by three.js */}
      </div>
    </div>
  );
}

export default Visualization3D;
