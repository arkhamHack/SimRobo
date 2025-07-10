import React from 'react';
import { AppBar, Toolbar, Typography, Box, Button, useTheme } from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import GitHubIcon from '@mui/icons-material/GitHub';

function Header() {
  const theme = useTheme();
  
  return (
    <AppBar 
      position="static" 
      color="transparent" 
      elevation={0}
      sx={{ 
        backdropFilter: 'blur(10px)',
        borderBottom: `1px solid ${theme.palette.divider}`,
        background: `linear-gradient(180deg, ${theme.palette.background.paper} 0%, ${theme.palette.background.default} 100%)`,
      }}
    >
      <Toolbar>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center',
            gap: 1.5,
            flexGrow: 1
          }}
        >
          <TimelineIcon color="primary" fontSize="large" />
          <Typography variant="h5" component="h1" sx={{ fontWeight: 600 }}>
            V-JEPA2 Trajectory Prediction
          </Typography>
        </Box>
        
        <Button 
          href="https://github.com/facebookresearch/jepa" 
          target="_blank" 
          rel="noopener noreferrer"
          startIcon={<GitHubIcon />}
          color="inherit"
          sx={{ textTransform: 'none' }}
        >
          GitHub
        </Button>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
