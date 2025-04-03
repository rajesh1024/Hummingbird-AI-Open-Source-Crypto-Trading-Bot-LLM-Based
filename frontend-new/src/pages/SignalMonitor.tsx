import React, { useState, useEffect } from 'react';
import {
  Card,
  Typography,
  Box,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  Paper,
  Divider
} from '@mui/material';

interface Signal {
  symbol: string;
  type: 'BUY' | 'SELL';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  risk_reward: number;
  timeframe: string;
  reasoning: string;
  timestamp: string;
}

const SignalMonitor: React.FC = () => {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const apiUrl = process.env.REACT_APP_API_URL || 'http://backend:8000';

  const fetchSignals = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/signals`);
      if (!response.ok) throw new Error('Failed to fetch signals');
      const data = await response.json();
      setSignals(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getSignalColor = (type: 'BUY' | 'SELL'): 'success' | 'error' => {
    return type === 'BUY' ? 'success' : 'error';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>;
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>Signal Monitor</Typography>
      
      {signals.length === 0 ? (
        <Alert severity="info" sx={{ m: 2 }}>No signals available at the moment.</Alert>
      ) : (
        <Grid container spacing={3}>
          {signals.map((signal, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Card sx={{ p: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">{signal.symbol}</Typography>
                  <Chip
                    label={signal.type}
                    color={getSignalColor(signal.type)}
                    size="small"
                  />
                </Box>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" color="text.secondary">Confidence</Typography>
                    <Typography variant="body1">
                      {signal.confidence.toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" color="text.secondary">Timeframe</Typography>
                    <Typography variant="body1">{signal.timeframe}</Typography>
                  </Grid>
                </Grid>

                <Divider sx={{ my: 2 }} />

                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Typography variant="subtitle2" color="text.secondary">Entry</Typography>
                    <Typography variant="body1">${signal.entry_price}</Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="subtitle2" color="text.secondary">Stop Loss</Typography>
                    <Typography variant="body1" color="error.main">
                      ${signal.stop_loss}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="subtitle2" color="text.secondary">Take Profit</Typography>
                    <Typography variant="body1" color="success.main">
                      ${signal.take_profit}
                    </Typography>
                  </Grid>
                </Grid>

                <Box mt={2}>
                  <Typography variant="subtitle2" color="text.secondary">Risk:Reward</Typography>
                  <Typography variant="body1">{signal.risk_reward.toFixed(2)}</Typography>
                </Box>

                <Paper variant="outlined" sx={{ p: 2, mt: 2, bgcolor: 'background.default' }}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Reasoning
                  </Typography>
                  <Typography variant="body2">{signal.reasoning}</Typography>
                </Paper>

                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                  Generated at: {new Date(signal.timestamp).toLocaleString()}
                </Typography>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default SignalMonitor; 