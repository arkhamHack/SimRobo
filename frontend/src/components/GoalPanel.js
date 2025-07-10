import React, { useRef } from 'react';

function GoalPanel({ goalImage, goalSet, onFileUpload, onCapture }) {
  const fileInputRef = useRef(null);
  
  const handleOverlayClick = () => {
    fileInputRef.current.click();
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
  };
  
  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('dragover');
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageData = event.target.result.split(',')[1];
        onFileUpload({ target: { files: [file] } });
      };
      reader.readAsDataURL(file);
    }
  };
  
  return (
    <div className="panel view-panel">
      <h2>Goal Image</h2>
      <div 
        className="image-container"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <img src={goalImage} alt="Goal image" />
        {!goalSet && (
          <div className="overlay" onClick={handleOverlayClick}>
            <span>Drop image here or click to upload</span>
          </div>
        )}
      </div>
      <div className="button-container">
        <input 
          type="file" 
          ref={fileInputRef} 
          accept="image/*" 
          hidden
          onChange={onFileUpload}
        />
        <button 
          className="btn secondary" 
          onClick={() => fileInputRef.current.click()}
        >
          Upload Goal Image
        </button>
        <button 
          className="btn secondary"
          onClick={onCapture}
        >
          Capture Current View
        </button>
      </div>
    </div>
  );
}

export default GoalPanel;
