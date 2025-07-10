import React from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import Header from './components/Header';
import Footer from './components/Footer';
import TrajectoryPredictor from './components/TrajectoryPredictor';

// API URL for backend communication
const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8080';

function App() {
  // Create modern dark theme
  const theme = createTheme({
    palette: {
      mode: 'dark',
      primary: {
        main: '#90caf9',
      },
      secondary: {
        main: '#f48fb1',
      },
      background: {
        default: '#121212',
        paper: '#1e1e1e',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h6: {
        fontWeight: 500,
      },
      button: {
        textTransform: 'none',
        fontWeight: 500,
      },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: '0 8px 16px 0 rgba(0,0,0,0.2)',
          },
        },
      },
    },
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        <Header />

        <main>
          <TrajectoryPredictor />
        </main>

        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default App;
