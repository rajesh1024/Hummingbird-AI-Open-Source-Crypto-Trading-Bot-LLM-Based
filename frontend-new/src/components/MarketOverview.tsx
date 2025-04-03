import React from 'react';
import { 
  Box, 
  Grid, 
  Typography, 
  Paper, 
  Divider,
  Tooltip,
  LinearProgress,
  Card,
  CardContent,
  useTheme
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import { useWebSocket } from '../hooks/useWebSocket';

interface TechnicalIndicators {
  RSI: number;
  MACD: {
    MACD: number;
    Signal: number;
    Histogram: number;
  };
  EMA: {
    EMA8: number;
    EMA21: number;
    EMA50: number;
    EMA200: number;
  };
  BB: {
    Upper: number;
    Middle: number;
    Lower: number;
  };
  Volume: {
    '1h': number;
    '4h': number;
    '1d': number;
  };
  ATR: number;
}

interface MarketData {
  symbol: string;
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  technical_indicators: TechnicalIndicators;
  trading_mode: string;
}

interface DashboardData {
  type: string;
  data: {
    positions: any[];
    market_data: MarketData;
    signal: any;
  };
}

const MarketOverview: React.FC = () => {
  const theme = useTheme();
  const wsUrl = process.env.REACT_APP_WS_URL || 'ws://backend:8000/ws/dashboard';
  const { data, error, loading, isConnected } = useWebSocket<DashboardData>(wsUrl);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <Typography>Loading market data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <Typography color="error">Error loading market data</Typography>
      </Box>
    );
  }

  if (!data?.data?.market_data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <Typography>No market data available</Typography>
      </Box>
    );
  }

  const { 
    symbol, 
    current_price, 
    price_change_24h, 
    volume_24h, 
    technical_indicators,
    trading_mode 
  } = data.data.market_data;

  const isPriceUp = price_change_24h > 0;

  const getRSIColor = (value: number) => {
    if (value >= 70) return 'error';
    if (value <= 30) return 'success';
    return 'warning';
  };

  const getMACDStatus = (macd: number, signal: number, histogram: number) => {
    if (histogram > 0 && macd > signal) return 'success';
    if (histogram < 0 && macd < signal) return 'error';
    return 'warning';
  };

  const getEMAStatus = (price: number, ema: number) => {
    return price > ema ? 'success' : 'error';
  };

  const formatLargeNumber = (num: number) => {
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
  };

  return (
    <Box>
      {/* Primary Metrics */}
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box display="flex" alignItems="center">
              <ShowChartIcon sx={{ mr: 1 }} />
              <Typography variant="h6">{symbol}</Typography>
            </Box>
            <Box display="flex" alignItems="center">
              <Typography variant="h5" sx={{ mr: 2 }}>
                ${current_price.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                {trading_mode.toUpperCase()}
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      {/* Technical Indicators Grid */}
      <Grid container spacing={2}>
        {/* Price Change and Volume */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              24h Change & Volume
            </Typography>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Box display="flex" alignItems="center">
                {isPriceUp ? (
                  <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                ) : (
                  <TrendingDownIcon color="error" sx={{ mr: 1 }} />
                )}
                <Typography
                  variant="h6"
                  color={isPriceUp ? 'success.main' : 'error.main'}
                >
                  {Math.abs(price_change_24h).toFixed(2)}%
                </Typography>
              </Box>
              <Typography variant="h6">
                ${formatLargeNumber(volume_24h)}
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* RSI */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              RSI (14)
            </Typography>
            <Box>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <TimelineIcon sx={{ mr: 1 }} color={getRSIColor(technical_indicators.RSI)} />
                <Typography variant="h6" color={`${getRSIColor(technical_indicators.RSI)}.main`}>
                  {technical_indicators.RSI.toFixed(1)}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={technical_indicators.RSI}
                color={getRSIColor(technical_indicators.RSI)}
                sx={{ mt: 1, height: 6, borderRadius: 3 }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* MACD */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              MACD
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={4}>
                <Tooltip title="MACD Line">
                  <Typography variant="body2" color="textSecondary">
                    MACD: {technical_indicators.MACD.MACD.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={4}>
                <Tooltip title="Signal Line">
                  <Typography variant="body2" color="textSecondary">
                    Signal: {technical_indicators.MACD.Signal.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={4}>
                <Tooltip title="Histogram">
                  <Typography 
                    variant="body2" 
                    color={technical_indicators.MACD.Histogram > 0 ? 'success.main' : 'error.main'}
                  >
                    Hist: {technical_indicators.MACD.Histogram.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Moving Averages */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              Moving Averages
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Tooltip title="8 EMA">
                  <Typography 
                    variant="body2" 
                    color={getEMAStatus(current_price, technical_indicators.EMA.EMA8) + '.main'}
                  >
                    EMA8: {technical_indicators.EMA.EMA8.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={6}>
                <Tooltip title="21 EMA">
                  <Typography 
                    variant="body2" 
                    color={getEMAStatus(current_price, technical_indicators.EMA.EMA21) + '.main'}
                  >
                    EMA21: {technical_indicators.EMA.EMA21.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={6}>
                <Tooltip title="50 EMA">
                  <Typography 
                    variant="body2" 
                    color={getEMAStatus(current_price, technical_indicators.EMA.EMA50) + '.main'}
                  >
                    EMA50: {technical_indicators.EMA.EMA50.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={6}>
                <Tooltip title="200 EMA">
                  <Typography 
                    variant="body2" 
                    color={getEMAStatus(current_price, technical_indicators.EMA.EMA200) + '.main'}
                  >
                    EMA200: {technical_indicators.EMA.EMA200.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Multi-timeframe Volume */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              Multi-timeframe Volume
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">1H</Typography>
                    <Typography variant="body1">
                      ${formatLargeNumber(technical_indicators.Volume['1h'])}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">4H</Typography>
                    <Typography variant="body1">
                      ${formatLargeNumber(technical_indicators.Volume['4h'])}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">1D</Typography>
                    <Typography variant="body1">
                      ${formatLargeNumber(technical_indicators.Volume['1d'])}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Bollinger Bands */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              Bollinger Bands
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={4}>
                <Tooltip title="Upper Band">
                  <Typography variant="body2" color="textSecondary">
                    Upper: {technical_indicators.BB.Upper.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={4}>
                <Tooltip title="Middle Band (20 SMA)">
                  <Typography variant="body2" color="textSecondary">
                    Middle: {technical_indicators.BB.Middle.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
              <Grid item xs={4}>
                <Tooltip title="Lower Band">
                  <Typography variant="body2" color="textSecondary">
                    Lower: {technical_indicators.BB.Lower.toFixed(2)}
                  </Typography>
                </Tooltip>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* ATR */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              Average True Range (14)
            </Typography>
            <Typography variant="h6">
              {technical_indicators.ATR.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MarketOverview; 