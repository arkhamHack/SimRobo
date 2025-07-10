import React from 'react';
import { Box, Typography, Link, Divider, useTheme, alpha } from '@mui/material';

function Footer() {
  const theme = useTheme();
  const currentYear = new Date().getFullYear();
  
  return (
    <Box 
      component="footer" 
      sx={{ 
        py: 3, 
        textAlign: 'center',
        mt: 'auto',
        borderTop: `1px solid ${alpha(theme.palette.divider, 0.3)}`,
        background: `linear-gradient(0deg, ${theme.palette.background.paper} 0%, ${theme.palette.background.default} 100%)`,
      }}
    >
      <Typography variant="body2" color="text.secondary">
        V-JEPA2 Trajectory Prediction System
      </Typography>
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1, mb: 1 }}>
        <Link href="https://github.com/facebookresearch/jepa" target="_blank" color="inherit" underline="hover">
          V-JEPA2 Project
        </Link>
        <Divider orientation="vertical" flexItem sx={{ mx: 1, opacity: 0.5 }} />
        <Link href="https://ai.meta.com/research/" target="_blank" color="inherit" underline="hover">
          Meta AI Research
        </Link>
      </Box>
      <Typography variant="caption" color="text.disabled" sx={{ mt: 1, display: 'block', opacity: 0.7 }}>
        © {currentYear} • Built with V-JEPA2, React, and Material UI
      </Typography>
    </Box>
  );
}

export default Footer;
