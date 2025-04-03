import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { 
  ThemeProvider, 
  CssBaseline, 
  Box, 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton,
  useTheme,
  useMediaQuery
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { lightTheme, darkTheme } from './theme';
import Dashboard from './components/Dashboard';
import PositionHistory from './components/PositionHistory';
import Sidebar from './components/Sidebar';

const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('darkMode');
    return savedMode ? JSON.parse(savedMode) : true;
  });
  
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    const savedState = localStorage.getItem('sidebarOpen');
    return savedState ? JSON.parse(savedState) : true;
  });

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('sidebarOpen', JSON.stringify(sidebarOpen));
  }, [sidebarOpen]);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <AppBar 
            position="fixed" 
            elevation={0}
            sx={(theme) => ({ 
              zIndex: theme.zIndex.drawer + 1,
              backgroundColor: theme.palette.mode === 'dark' ? '#121212' : '#ffffff',
              borderBottom: `1px solid ${theme.palette.divider}`,
              '& .MuiToolbar-root, & .MuiIconButton-root, & .MuiTypography-root': {
                color: theme.palette.mode === 'dark' ? '#ffffff' : '#121212',
              }
            })}
          >
            <Toolbar sx={{ minHeight: { xs: 56, sm: 64 } }}>
              {isMobile && (
                <IconButton
                  edge="start"
                  onClick={toggleSidebar}
                  aria-label="menu"
                  sx={{ 
                    mr: 2,
                    '&:hover': {
                      backgroundColor: theme.palette.mode === 'dark' 
                        ? 'rgba(255, 255, 255, 0.08)' 
                        : 'rgba(0, 0, 0, 0.04)',
                    },
                  }}
                >
                  <MenuIcon />
                </IconButton>
              )}
              <Typography 
                variant="h5" 
                component="div" 
                sx={{ 
                  flexGrow: 1,
                  fontWeight: 600,
                  letterSpacing: '-0.5px',
                  fontSize: { xs: '1.15rem', sm: '1.5rem' },
                  fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
                }}
              >
                Hummingbird
              </Typography>
            </Toolbar>
          </AppBar>

          <Sidebar
            open={sidebarOpen}
            onToggle={toggleSidebar}
            darkMode={darkMode}
            onThemeChange={toggleTheme}
          />
          
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              ml: { sm: `${sidebarOpen ? 240 : 0}px` },
              mt: '64px', // Height of AppBar
              transition: 'margin 225ms cubic-bezier(0.4, 0, 0.6, 1) 0ms',
              p: { xs: 2, sm: 3 },
            }}
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/position-history" element={<PositionHistory />} />
              {/* Add more routes here */}
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;
