import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  Typography,
  Alert,
  CircularProgress,
  Chip,
  Divider
} from '@mui/material';

interface MarketData {
  symbol: string;
  price: number;
  change_24h: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
}

interface ActivePosition {
  symbol: string;
  type: 'LONG' | 'SHORT';
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percentage: number;
  size: number;
  opened_at: string;
}

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

const Dashboard: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [activePositions, setActivePositions] = useState<ActivePosition[]>([]);
  const [latestSignal, setLatestSignal] = useState<Signal | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const apiUrl = process.env.REACT_APP_API_URL || 'http://backend:8000';

  const fetchDashboardData = async () => {
    try {
      const [marketResponse, positionsResponse, signalResponse] = await Promise.all([
        fetch(`${apiUrl}/api/market-data`),
        fetch(`${apiUrl}/api/positions/active`),
        fetch(`${apiUrl}/api/signals/latest`)
      ]);

      if (!marketResponse.ok || !positionsResponse.ok || !signalResponse.ok) {
        throw new Error('Failed to fetch dashboard data');
      }

      const [marketData, positions, signal] = await Promise.all([
        marketResponse.json(),
        positionsResponse.json(),
        signalResponse.json()
      ]);

      setMarketData(marketData);
      setActivePositions(positions);
      setLatestSignal(signal);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getChangeColor = (change: number): 'success' | 'error' => {
    return change >= 0 ? 'success' : 'error';
  };

  const getPositionColor = (type: 'LONG' | 'SHORT'): 'success' | 'error' => {
    return type === 'LONG' ? 'success' : 'error';
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
      <Typography variant="h5" gutterBottom>Dashboard</Typography>

      {/* Market Overview */}
      <Card sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>Market Overview</Typography>
        <Grid container spacing={2}>
          {marketData.map((market) => (
            <Grid item xs={12} sm={6} md={3} key={market.symbol}>
              <Box>
                <Typography variant="subtitle1">{market.symbol}</Typography>
                <Typography variant="h6">${market.price.toFixed(2)}</Typography>
                <Chip
                  label={`${market.change_24h.toFixed(2)}%`}
                  color={getChangeColor(market.change_24h)}
                  size="small"
                  sx={{ mt: 1 }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Vol: ${market.volume_24h.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  H: ${market.high_24h.toFixed(2)} L: ${market.low_24h.toFixed(2)}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Card>

      {/* Active Positions */}
      <Card sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>Active Positions</Typography>
        {activePositions.length === 0 ? (
          <Alert severity="info">No active positions</Alert>
        ) : (
          <Grid container spacing={2}>
            {activePositions.map((position) => (
              <Grid item xs={12} sm={6} md={4} key={`${position.symbol}-${position.type}`}>
                <Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="subtitle1">{position.symbol}</Typography>
                    <Chip
                      label={position.type}
                      color={getPositionColor(position.type)}
                      size="small"
                    />
                  </Box>
                  <Typography variant="h6" color={position.pnl >= 0 ? 'success.main' : 'error.main'}>
                    ${position.pnl.toFixed(2)} ({position.pnl_percentage.toFixed(2)}%)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Entry: ${position.entry_price.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Current: ${position.current_price.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Size: {position.size}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        )}
      </Card>

      {/* Latest Signal */}
      <Card sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>Latest Signal</Typography>
        {latestSignal ? (
          <Box>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="subtitle1">{latestSignal.symbol}</Typography>
              <Chip
                label={latestSignal.type}
                color={latestSignal.type === 'BUY' ? 'success' : 'error'}
                size="small"
              />
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="subtitle2" color="text.secondary">Confidence</Typography>
                <Typography variant="body1">{latestSignal.confidence.toFixed(1)}%</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2" color="text.secondary">Timeframe</Typography>
                <Typography variant="body1">{latestSignal.timeframe}</Typography>
              </Grid>
            </Grid>
            <Divider sx={{ my: 2 }} />
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Typography variant="subtitle2" color="text.secondary">Entry</Typography>
                <Typography variant="body1">${latestSignal.entry_price}</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="subtitle2" color="text.secondary">Stop Loss</Typography>
                <Typography variant="body1" color="error.main">
                  ${latestSignal.stop_loss}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="subtitle2" color="text.secondary">Take Profit</Typography>
                <Typography variant="body1" color="success.main">
                  ${latestSignal.take_profit}
                </Typography>
              </Grid>
            </Grid>
            <Box mt={2}>
              <Typography variant="subtitle2" color="text.secondary">Risk:Reward</Typography>
              <Typography variant="body1">{latestSignal.risk_reward.toFixed(2)}</Typography>
            </Box>
            <Box mt={2}>
              <Typography variant="subtitle2" color="text.secondary">Reasoning</Typography>
              <Typography variant="body2">{latestSignal.reasoning}</Typography>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
              Generated at: {new Date(latestSignal.timestamp).toLocaleString()}
            </Typography>
          </Box>
        ) : (
          <Alert severity="info">No signals available</Alert>
        )}
      </Card>
    </Box>
  );
};

export default Dashboard; 