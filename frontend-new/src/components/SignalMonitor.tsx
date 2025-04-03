import React from 'react';
import { Card, CardContent, Typography, Box, Chip, CircularProgress, Alert } from '@mui/material';
import { useWebSocket } from '../hooks/useWebSocket';

interface Signal {
  signal: string;
  confidence: number;
  timeframe: string;
  reason: string;
  timestamp: string;
}

interface SignalData {
  latest_signal: Signal;
}

const SignalMonitor: React.FC = () => {
  const { data, isConnected, error, loading } = useWebSocket<SignalData>(process.env.REACT_APP_WS_URL || 'ws://backend:8000/ws/dashboard');

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">Error: {error.message}</Alert>
        </CardContent>
      </Card>
    );
  }

  if (!isConnected) {
    return (
      <Card>
        <CardContent>
          <Alert severity="warning">Connecting to server...</Alert>
        </CardContent>
      </Card>
    );
  }

  if (!data?.latest_signal) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Signal Monitor</Typography>
          <Typography color="text.secondary">No signals available</Typography>
        </CardContent>
      </Card>
    );
  }

  const signal = data.latest_signal;
  const getSignalColor = (type: string) => {
    switch (type?.toUpperCase()) {
      case 'BUY':
      case 'LONG':
        return 'success';
      case 'SELL':
      case 'SHORT':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Signal Monitor</Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" color="text.secondary">Signal Type</Typography>
          <Chip 
            label={signal.signal || 'NEUTRAL'} 
            color={getSignalColor(signal.signal)}
            sx={{ mt: 1 }}
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" color="text.secondary">Confidence</Typography>
          <Typography variant="body1">
            {(signal.confidence * 100).toFixed(1)}%
          </Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" color="text.secondary">Timeframe</Typography>
          <Typography variant="body1">{signal.timeframe}</Typography>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" color="text.secondary">Reasoning</Typography>
          <Typography variant="body2">{signal.reason || 'No reasoning provided'}</Typography>
        </Box>

        <Box>
          <Typography variant="subtitle2" color="text.secondary">Last Updated</Typography>
          <Typography variant="caption">
            {new Date(signal.timestamp).toLocaleString()}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SignalMonitor; 