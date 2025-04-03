import React from 'react';
import { AppBar, Tabs, Tab, Box } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';

const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleChange = (event: React.SyntheticEvent, newValue: string) => {
    navigate(newValue);
  };

  const getCurrentTab = () => {
    switch (location.pathname) {
      case '/':
        return 0;
      case '/positions':
        return 1;
      default:
        return 0;
    }
  };

  return (
    <AppBar position="static" color="default">
      <Tabs
        value={getCurrentTab()}
        onChange={handleChange}
        indicatorColor="primary"
        textColor="primary"
        variant="fullWidth"
      >
        <Tab label="Dashboard" />
        <Tab label="Position History" />
      </Tabs>
    </AppBar>
  );
};

export default Navigation; 