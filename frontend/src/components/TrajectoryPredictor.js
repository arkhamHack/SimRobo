import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  LinearProgress,
  Slider,
  TextField,
  Typography,
  Paper,
  Fade,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  alpha,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import TimelineIcon from '@mui/icons-material/Timeline';
import FastForwardIcon from '@mui/icons-material/FastForward';
import FastRewindIcon from '@mui/icons-material/FastRewind';
import TuneIcon from '@mui/icons-material/Tune';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const TrajectoryPredictor = () => {
  const [videoId, setVideoId] = useState(null);
  const [textQuery, setTextQuery] = useState('');
  const [planningHorizon, setPlanningHorizon] = useState(10);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [videoInfo, setVideoInfo] = useState(null);
  const [frames, setFrames] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [logs, setLogs] = useState([]);
  
  const canvasRef = useRef(null);
  const playIntervalRef = useRef(null);
  
  // Initialize on mount
  useEffect(() => {
    checkStatus();
    
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, []);
  
  // Draw frame when current frame index changes
  useEffect(() => {
    if (frames.length > 0 && currentFrameIndex < frames.length) {
      drawFrame(currentFrameIndex);
    }
  }, [currentFrameIndex, frames]);
  
  // Handle play/pause
  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = setInterval(() => {
        setCurrentFrameIndex(prevIndex => {
          const nextIndex = (prevIndex + 1) % frames.length;
          if (nextIndex === 0) {
            setIsPlaying(false);
          }
          return nextIndex;
        });
      }, 200);
    } else if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
    }
    
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, frames.length]);
  
  const checkStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      addLog(`V-JEPA2 status: ${response.data.vjepa_initialized ? 'Initialized' : 'Not Initialized'}`);
    } catch (error) {
      addLog('Error connecting to server');
    }
  };
  
  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prevLogs => [...prevLogs, `[${timestamp}] ${message}`]);
  };
  
  const handleUpload = async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (!file) {
      addLog('No file selected');
      return;
    }
    
    setIsLoading(true);
    setMessage('Uploading video...');
    addLog(`Uploading video: ${file.name}`);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('max_frames', 30);
    
    try {
      const response = await axios.post('/api/upload_video', formData);
      setVideoId(response.data.video_id);
      setVideoInfo(response.data);
      setMessage('Video uploaded successfully!');
      addLog(`Upload successful! Video ID: ${response.data.video_id}`);
      addLog(`Frames: ${response.data.num_frames}, Resolution: ${response.data.width}x${response.data.height}`);
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
      addLog(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handlePredict = async (event) => {
    event.preventDefault();
    
    if (!videoId) {
      addLog('Please upload a video first');
      return;
    }
    
    if (!textQuery) {
      addLog('Please enter an object description');
      return;
    }
    
    setIsLoading(true);
    setMessage('Predicting trajectory...');
    addLog(`Predicting trajectory for "${textQuery}"`);
    
    try {
      const response = await axios.post('/api/predict_trajectory', {
        text_query: textQuery,
        video_id: videoId,
        planning_horizon: planningHorizon,
        confidence_threshold: confidenceThreshold
      });
      
      if (response.data.success) {
        setPredictionResult(response.data);
        
        // Convert base64 images to Image objects for canvas drawing
        const loadedFrames = await Promise.all(
          response.data.visualization_frames.map(base64 => {
            return new Promise((resolve) => {
              const img = new Image();
              img.onload = () => resolve(img);
              img.src = `data:image/jpeg;base64,${base64}`;
            });
          })
        );
        
        setFrames(loadedFrames);
        setCurrentFrameIndex(0);
        
        addLog(`Prediction complete: ${response.data.observed_trajectory.length} observed frames, ${response.data.predicted_trajectory.length} predicted frames`);
        setMessage('Trajectory prediction complete');
        
        // Initialize canvas size
        if (loadedFrames.length > 0) {
          const canvas = canvasRef.current;
          canvas.width = loadedFrames[0].width;
          canvas.height = loadedFrames[0].height;
        }
      } else {
        setMessage(`Error: ${response.data.message}`);
        addLog(`Error: ${response.data.message}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
      addLog(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const drawFrame = (index) => {
    if (frames.length === 0 || index >= frames.length) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw the frame
    ctx.drawImage(frames[index], 0, 0);
  };
  
  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };
  
  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Fade in={true} timeout={800}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card 
              elevation={4} 
              sx={{ 
                p: 1, 
                background: theme => `linear-gradient(145deg, ${alpha(theme.palette.background.paper, 0.8)}, ${theme.palette.background.paper})`,
                backdropFilter: 'blur(10px)',
                position: 'relative',
                overflow: 'visible'
              }}
            >
              <Box sx={{ 
                position: 'absolute', 
                top: -12, 
                left: 24, 
                px: 2, 
                py: 0.5, 
                bgcolor: 'primary.main',
                borderRadius: '12px',
                boxShadow: '0 3px 5px rgba(0,0,0,0.2)',
                zIndex: 5
              }}>
                <Typography variant="h6" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TimelineIcon fontSize="small" />
                  V-JEPA2 Trajectory Prediction
                </Typography>
              </Box>

              <CardContent sx={{ pt: 3, mt: 1 }}>
                {/* Video Upload Section */}
                <Paper 
                  elevation={2}
                  sx={{ 
                    p: 3, 
                    mt: 2, 
                    borderRadius: 2,
                    background: theme => alpha(theme.palette.background.default, 0.5),
                    backdropFilter: 'blur(20px)',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                    <Chip 
                      icon={<CloudUploadIcon />} 
                      label="Step 1: Upload" 
                      color="primary" 
                      variant="filled"
                      sx={{ fontWeight: 500, px: 1 }}
                    />
                    <Tooltip title="Upload a short video (5-15 seconds) for best results">  
                      <IconButton size="small">
                        <InfoOutlinedIcon fontSize="small" />  
                      </IconButton>
                    </Tooltip>  
                  </Box>

                  <Box component="form" onSubmit={handleUpload} sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 2 }}>
                    <input
                      accept="video/*"
                      id="videoFile"
                      type="file"
                      style={{ display: 'none' }}
                      onChange={(e) => e.target.files[0] && setMessage(`Selected: ${e.target.files[0].name}`)}
                    />
                    <label htmlFor="videoFile">
                      <Button
                        variant="outlined"
                        component="span"
                        startIcon={<UploadFileIcon />}
                        sx={{ 
                          borderWidth: 2,
                          '&:hover': { borderWidth: 2 }
                        }}
                      >
                        Select Video
                      </Button>
                    </label>
                    <Typography variant="body2" sx={{ 
                      px: 2,
                      py: 1,
                      borderRadius: 1,
                      bgcolor: alpha('#ffffff', 0.05),
                      minWidth: 200
                    }}>
                      {document.getElementById('videoFile')?.files[0]?.name || 'No file selected'}
                    </Typography>
                    {document.getElementById('videoFile')?.files?.length > 0 && (
                      <Button
                        type="submit"
                        variant="contained"
                        startIcon={<CloudUploadIcon />}
                        disabled={isLoading}
                        sx={{ 
                          px: 3,
                          boxShadow: '0 4px 8px rgba(0,0,0,0.3)', 
                          '&:hover': { transform: 'translateY(-2px)' },
                          transition: 'transform 0.2s'
                        }}
                      >
                        Upload
                      </Button>
                    )}
                  </Box>

                  {videoInfo && (
                    <Fade in={!!videoInfo} timeout={500}>
                      <Box sx={{ mt: 2 }}>
                        <Chip 
                          label={`${videoInfo.num_frames} frames · ${videoInfo.width}x${videoInfo.height}`} 
                          size="small" 
                          color="secondary" 
                          variant="outlined"
                          sx={{ fontSize: '0.75rem' }}
                        />
                      </Box>
                    </Fade>
                  )}
                </Paper>

                <Divider sx={{ my: 3, opacity: 0.6 }} />
                
                {/* Trajectory Prediction Section */}
                <Paper 
                  elevation={2}
                  sx={{ 
                    p: 3, 
                    mb: 2,
                    borderRadius: 2,
                    background: theme => alpha(theme.palette.background.default, 0.5),
                    backdropFilter: 'blur(20px)',
                    opacity: videoId ? 1 : 0.7,
                    transition: 'opacity 0.3s'
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                    <Chip 
                      icon={<TuneIcon />} 
                      label="Step 2: Configure & Predict" 
                      color="secondary" 
                      variant="filled"
                      sx={{ fontWeight: 500, px: 1 }}
                    />
                    <Tooltip title="Describe the object you want to track and predict">  
                      <IconButton size="small">
                        <InfoOutlinedIcon fontSize="small" />  
                      </IconButton>
                    </Tooltip>
                  </Box>

                  <Box component="form" onSubmit={handlePredict}>
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6} md={4} lg={3}>
                        <TextField
                          fullWidth
                          label="Object to Track"
                          value={textQuery}
                          onChange={(e) => setTextQuery(e.target.value)}
                          placeholder="E.g., red ball, person walking"
                          disabled={isLoading}
                          variant="outlined"
                          error={videoId === null && textQuery !== ''}
                          helperText={videoId === null && textQuery !== '' ? 'Upload a video first' : ''}
                          InputProps={{
                            sx: { borderRadius: 2 }
                          }}
                        />
                      </Grid>

                      <Grid item xs={12} sm={6} md={8} lg={6}>
                        <Box sx={{ 
                          p: 2, 
                          border: theme => `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                          borderRadius: 2,
                          bgcolor: alpha('#ffffff', 0.03)
                        }}>
                          <Typography variant="body2" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Chip label="Planning Horizon" size="small" color="primary" variant="outlined" />
                            <span>{planningHorizon} frames</span>
                          </Typography>
                          <Slider
                            value={planningHorizon}
                            onChange={(e, newValue) => setPlanningHorizon(newValue)}
                            valueLabelDisplay="auto"
                            step={1}
                            marks
                            min={5}
                            max={15}
                            disabled={isLoading}
                            sx={{ 
                              '& .MuiSlider-thumb': {
                                width: 12,
                                height: 12,
                                transition: '0.2s',
                                '&:hover': {
                                  boxShadow: '0 0 0 8px rgba(144, 202, 249, 0.16)'
                                }
                              }
                            }}
                          />

                          <Typography variant="body2" sx={{ mt: 2, mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Chip label="Confidence Threshold" size="small" color="secondary" variant="outlined" />
                            <span>{confidenceThreshold.toFixed(2)}</span>
                          </Typography>
                          <Slider
                            value={confidenceThreshold}
                            onChange={(e, newValue) => setConfidenceThreshold(newValue)}
                            valueLabelDisplay="auto"
                            step={0.05}
                            marks
                            min={0.5}
                            max={0.9}
                            disabled={isLoading}
                            color="secondary"
                            sx={{ 
                              '& .MuiSlider-thumb': {
                                width: 12,
                                height: 12,
                                transition: '0.2s',
                                '&:hover': {
                                  boxShadow: '0 0 0 8px rgba(244, 143, 177, 0.16)'
                                }
                              }
                            }}
                          />
                        </Box>
                      </Grid>

                      <Grid item xs={12} sm={6} md={4} lg={3}>
                        <Button
                          type="submit"
                          variant="contained"
                          color="secondary"
                          fullWidth
                          size="large"
                          startIcon={<TimelineIcon />}
                          disabled={!videoId || !textQuery || isLoading}
                          sx={{ 
                            height: '100%', 
                            minHeight: '56px',
                            borderRadius: 2,
                            boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
                            '&:hover': { transform: 'translateY(-2px)' },
                            transition: 'transform 0.2s'
                          }}
                        >
                          Predict Trajectory
                        </Button>
                      </Grid>
                    </Grid>
                  </Box>
                </Paper>
                
                {isLoading && (
                  <Fade in={isLoading} timeout={300}>
                    <Box sx={{ width: '100%', mt: 2 }}>
                      <LinearProgress sx={{ height: 6, borderRadius: 3 }} />
                      <Typography variant="body2" sx={{ mt: 1, textAlign: 'center', fontWeight: 500 }}>
                        {message}
                      </Typography>
                    </Box>
                  </Fade>
                )}
                
                {message && !isLoading && (
                  <Fade in={!!message && !isLoading} timeout={500}>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        mt: 2, 
                        textAlign: 'center',
                        color: message.includes('Error') ? 'error.main' : 'success.main',
                        fontWeight: 500,
                      }}
                    >
                      {message}
                    </Typography>
                  </Fade>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          {frames.length > 0 && (
            <>
              <Grid item xs={12} md={8} lg={9}>
                <Fade in={frames.length > 0} timeout={800}>
                  <Card 
                    elevation={4}
                    sx={{ 
                      overflow: 'hidden',
                      background: theme => `linear-gradient(145deg, ${alpha(theme.palette.background.paper, 0.8)}, ${theme.palette.background.paper})`,
                    }}
                  >
                    <CardContent>
                      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="h6" sx={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TimelineIcon fontSize="small" color="primary" />
                          Trajectory Visualization
                        </Typography>
                        
                        {predictionResult && (
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <Chip 
                              size="small" 
                              label={`Object: ${predictionResult.text_query}`} 
                              color="primary" 
                              variant="outlined" 
                            />
                            <Chip 
                              size="small" 
                              label={`Frames: ${predictionResult.observed_trajectory.length + predictionResult.predicted_trajectory.length}`} 
                              color="secondary" 
                              variant="outlined" 
                            />
                          </Box>
                        )}
                      </Box>

                      <Box 
                        sx={{ 
                          position: 'relative', 
                          mb: 2, 
                          borderRadius: 2,
                          overflow: 'hidden',
                          boxShadow: '0 4px 20px rgba(0,0,0,0.2)', 
                        }}
                      >
                        <canvas 
                          ref={canvasRef} 
                          style={{ 
                            display: 'block',
                            width: '100%',
                            maxHeight: '60vh',
                            objectFit: 'contain',
                            backgroundColor: '#111'
                          }} 
                        />
                      </Box>

                      <Box sx={{ mt: 3, px: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="body2" fontWeight={500}>
                            Frame: {currentFrameIndex + 1} / {frames.length}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Chip 
                              size="small" 
                              label="Observed" 
                              sx={{ 
                                bgcolor: 'success.dark',
                                color: 'white',
                                fontWeight: 500,
                                borderRadius: '4px'
                              }} 
                            />
                            <Chip 
                              size="small" 
                              label="Predicted" 
                              sx={{ 
                                bgcolor: 'primary.dark', 
                                color: 'white',
                                fontWeight: 500,
                                borderRadius: '4px'
                              }} 
                            />
                          </Box>
                        </Box>

                        <Slider
                          value={currentFrameIndex}
                          onChange={(e, value) => setCurrentFrameIndex(value)}
                          min={0}
                          max={frames.length - 1}
                          step={1}
                          sx={{ 
                            '& .MuiSlider-thumb': {
                              width: 12,
                              height: 12,
                              transition: '0.2s',
                              '&:hover': {
                                boxShadow: '0 0 0 8px rgba(144, 202, 249, 0.16)'
                              }
                            }
                          }}
                        />

                        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
                          <IconButton onClick={() => setCurrentFrameIndex(0)}>
                            <FastRewindIcon />
                          </IconButton>
                          <Button
                            variant="contained"
                            startIcon={isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                            onClick={togglePlayPause}
                            sx={{ 
                              px: 3,
                              borderRadius: 6,
                              minWidth: '120px',
                              boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
                              '&:hover': { transform: 'translateY(-2px)' },
                              transition: 'transform 0.2s'
                            }}
                          >
                            {isPlaying ? 'Pause' : 'Play'}
                          </Button>
                          <IconButton onClick={() => setCurrentFrameIndex(frames.length - 1)}>
                            <FastForwardIcon />
                          </IconButton>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Fade>
              </Grid>
              
              <Grid item xs={12} md={4} lg={3}>
                <Fade in={frames.length > 0} timeout={1000}>
                  <Card 
                    elevation={4}
                    sx={{ 
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      background: theme => `linear-gradient(145deg, ${alpha(theme.palette.background.paper, 0.8)}, ${theme.palette.background.paper})`,
                    }}
                  >
                    <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="h6" sx={{ mb: 2, fontWeight: 500, display: 'flex', alignItems: 'center', gap: 1 }}>
                        <InfoOutlinedIcon fontSize="small" color="secondary" />
                        System Logs
                      </Typography>
                      
                      <Paper 
                        sx={{ 
                          flex: 1, 
                          p: 1.5, 
                          overflow: 'auto', 
                          bgcolor: alpha('#000', 0.3), 
                          fontFamily: 'monospace',
                          borderRadius: 2,
                          minHeight: '200px',
                          border: theme => `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                        }}
                      >
                        {logs.length === 0 ? (
                          <Typography variant="body2" sx={{ opacity: 0.6, fontStyle: 'italic' }}>
                            No logs yet. Upload a video and make predictions to see logs.
                          </Typography>
                        ) : (
                          logs.map((log, index) => (
                            <Typography 
                              key={index} 
                              variant="body2" 
                              sx={{ 
                                fontSize: '0.8rem', 
                                mb: 0.5,
                                color: log.includes('Error') ? 'error.light' : 
                                       log.includes('success') ? 'success.light' : 'text.primary',
                              }}
                            >
                              {log}
                            </Typography>
                          ))
                        )}
                      </Paper>
                    </CardContent>
                  </Card>
                </Fade>
              </Grid>
            </>
          )}
        </Grid>
      </Fade>
    </Container>
  );
};

export default TrajectoryPredictor;
