import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  CircularProgress, 
  Alert, 
  Divider,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  useTheme,
  Button,
  AlertTitle,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  CardHeader,
  useMediaQuery
} from '@mui/material';
import { useWebSocket } from '../hooks/useWebSocket';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TimelineIcon from '@mui/icons-material/Timeline';
import SignalWifiStatusbar4BarIcon from '@mui/icons-material/SignalWifiStatusbar4Bar';
import SignalWifiOffIcon from '@mui/icons-material/SignalWifiOff';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import HistoryIcon from '@mui/icons-material/History';
import UpdateIcon from '@mui/icons-material/Update';

interface Position {
  id: number;
  symbol: string;
  position_type: string;
  status: string;
  entry_price: number;
  current_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  size: number;
  pnl: number;
  created_at: string | null;
  closed_at: string | null;
}

interface MarketData {
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  volume_4h: number;
  volume_1h: number;
  volume_15m: number;
  macd: number;
  ema: number;
  sma: number;
  rsi: number;
  symbol: string;
}

interface Signal {
  signal: string;
  confidence: number;
  symbol: string;
  timeframe: string;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  reason: string;
  timestamp: string;
  position_management: {
    action: string;
    stop_loss_adjustment: string;
    take_profit_adjustment: string;
    risk_reward_ratio: number;
  };
}

interface TechnicalIndicators {
  RSI: number;
  MACD: number;
  MACD_Signal: number;
  MACD_Hist: number;
  EMA_8: number;
  EMA_21: number;
}

interface SMCData {
  order_blocks: any[];
  liquidity_levels: any[];
  fair_value_gaps: any[];
  supply_zones: any[];
  demand_zones: any[];
  smart_money_traps: any[];
}

interface MarketContext {
  symbol: string;
  timeframe: string;
  current_price: number;
  technical_indicators: TechnicalIndicators;
  market_structure: string;
  smc_data: SMCData;
  active_positions: Position[];
  config: {
    trading: {
      modes: any;
      risk_management: any;
    };
    llm: {
      confidence_threshold: number;
      model_name: string;
    };
  };
}

interface DashboardData {
  type: string;
  data: {
    positions: Position[];
    market_data: MarketData;
    signal: Signal;
  };
}

interface Analysis {
  timestamp: string;
  symbol: string;
  analysis: string;
  confidence: number;
}

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const wsUrl = process.env.NODE_ENV === 'production'
    ? (process.env.REACT_APP_WS_URL || `ws://${window.location.hostname}:8000/ws/dashboard`)
    : `ws://${window.location.hostname}:8000/ws/dashboard`;
  const { data, error, loading, isConnected, reconnect } = useWebSocket<DashboardData>(wsUrl);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [analysisHistory, setAnalysisHistory] = useState<Array<Analysis>>([]);

  useEffect(() => {
    if (data?.data?.signal) {
      setAnalysisHistory(prevHistory => {
        const newAnalysis = {
          timestamp: data.data.signal.timestamp,
          symbol: data.data.signal.symbol,
          analysis: data.data.signal.reason,
          confidence: data.data.signal.confidence
        };
        
        // Check if this analysis is already in history to prevent duplicates
        const isDuplicate = prevHistory.some(
          item => item.timestamp === newAnalysis.timestamp && 
                 item.analysis === newAnalysis.analysis
        );
        
        if (!isDuplicate) {
          const newHistory = [newAnalysis, ...prevHistory];
          return newHistory.slice(0, 10); // Keep only last 10 analyses
        }
        return prevHistory;
      });
      setLastUpdate(new Date().toLocaleTimeString());
    }
  }, [data?.data?.signal]);

  // Add debug logging
  useEffect(() => {
    if (data) {
      console.log('Received WebSocket data:', data);
    }
    if (error) {
      console.error('WebSocket error:', error);
    }
  }, [data, error]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 50) return 'warning';
    return 'error';
  };

  const formatPrice = (price: number | null | undefined) => {
    if (price === null || price === undefined) return 'N/A';
    return `$${price.toLocaleString()}`;
  };

  const getSignalColor = (signal: string | undefined) => {
    switch (signal) {
      case 'BUY':
        return 'success';
      case 'SELL':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getActionColor = (action: string | undefined) => {
    switch (action) {
      case 'MAINTAIN':
        return 'success';
      case 'CLOSE':
        return 'error';
      default:
        return 'warning';
    }
  };

  const formatIndicator = (value: number | undefined | null) => {
    if (value === undefined || value === null) return 'N/A';
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

  const formatDuration = (createdAt: string | null) => {
    if (!createdAt) return null;
    const created = new Date(createdAt);
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - created.getTime()) / (1000 * 60));
    return diffInMinutes;
  };

  const renderAnalysisHistory = () => {
    return (
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
          Analysis History
        </Typography>
        
        {/* Fixed section for latest 2 analyses
        <Box sx={{ mb: 2 }}>
          {analysisHistory.slice(0, 2).map((analysis, index) => (
            <Box
              key={`${analysis.timestamp}-${index}`}
              sx={{
                p: 2,
                mb: 2,
                borderRadius: 1,
                bgcolor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider',
                position: 'relative',
              }}
            >
              {index === 0 && (
                <Chip
                  label="Latest"
                  size="small"
                  sx={{
                    position: 'absolute',
                    left: 16,
                    top: 16,
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                  }}
                />
              )}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1, mt: index === 0 ? 3 : 0 }}>
                <Typography variant="caption" color="text.secondary">
                  {new Date(analysis.timestamp).toLocaleString()}
                </Typography>
                <Chip
                  label={`${Math.round(analysis.confidence * 100)}%`}
                  size="small"
                  color="success"
                  sx={{ height: 20 }}
                />
              </Box>
              <Typography variant="body2">{analysis.analysis}</Typography>
            </Box>
          ))}
        </Box> */}

        {/* Scrollable section for older analyses */}
        {analysisHistory.length > 2 && (
          <Box
            sx={{
              maxHeight: '250px',
              overflowY: 'auto',
              mt: 2,
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                background: (theme) => theme.palette.divider,
                borderRadius: '4px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: (theme) => theme.palette.action.hover,
              },
            }}
          >
            {analysisHistory.map((analysis, index) => (
              <Box
                key={`${analysis.timestamp}-${index}`}
                sx={{
                  p: 2,
                  mb: 2,
                  borderRadius: 1,
                  bgcolor: 'background.paper',
                  border: '1px solid',
                  borderColor: 'divider',
                  '&:last-child': {
                    mb: 0,
                  },
                }}
              >
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {new Date(analysis.timestamp).toLocaleString()}
                  </Typography>
                  <Chip
                    label={`${Math.round(analysis.confidence * 100)}%`}
                    size="small"
                    color="success"
                    sx={{ height: 20 }}
                  />
                  {index === 0 && (
                  <Chip
                    label="Latest"
                    size="small"
                    sx={{
                      position: 'absolute',
                      left: 16,
                      top: 16,
                      bgcolor: 'primary.main',
                      color: 'primary.contrastText',
                    }}
                  />
                )}
                </Box>
                <Typography variant="body2">{analysis.analysis}</Typography>
              </Box>
            ))}
          </Box>
        )}
      </Box>
    );
  };

  const renderMobileMarketOverview = () => {
    if (!data?.data?.market_data) return null;
    
    return (
      <Card sx={{ mb: 2, bgcolor: 'background.paper' }}>
        <CardContent sx={{ p: 2 }}>
          <Box display="flex" alignItems="center" mb={2}>
            <TimelineIcon sx={{ mr: 1, fontSize: '1.2rem' }} />
            <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
              Market Overview
            </Typography>
          </Box>

          {/* Price and Change */}
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ 
              fontSize: '1.5rem',
              fontWeight: 'bold',
              wordBreak: 'break-word'
            }}>
              ${data.data.market_data.current_price.toLocaleString()}
            </Typography>
            <Box display="flex" alignItems="center" mt={0.5}>
              {data.data.market_data.price_change_24h >= 0 ? (
                <TrendingUpIcon color="success" sx={{ fontSize: '1.2rem' }} />
              ) : (
                <TrendingDownIcon color="error" sx={{ fontSize: '1.2rem' }} />
              )}
              <Typography 
                variant="body2" 
                color={data.data.market_data.price_change_24h >= 0 ? 'success.main' : 'error.main'}
                sx={{ ml: 0.5 }}
              >
                {data.data.market_data.price_change_24h.toFixed(2)}%
              </Typography>
            </Box>
          </Box>

          {/* Volume Grid */}
          <Grid container spacing={1} sx={{ mb: 1 }}>
            <Grid item xs={6}>
              <Box sx={{ p: 1, bgcolor: theme.palette.background.default, borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" noWrap>24h Volume</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  ${data.data.market_data.volume_24h.toLocaleString()}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ p: 1, bgcolor: theme.palette.background.default, borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" noWrap>4h Volume</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  ${data.data.market_data.volume_4h.toLocaleString()}
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {/* Indicators Grid */}
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Box sx={{ p: 1, bgcolor: theme.palette.background.default, borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" noWrap>RSI</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  {formatIndicator(data.data.market_data.rsi)}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ p: 1, bgcolor: theme.palette.background.default, borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" noWrap>EMA 8</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  {formatIndicator(data.data.market_data.ema)}
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  const renderMobileSignalAnalysis = () => {
    return (
      <Card sx={{ mb: 2, bgcolor: 'background.paper' }}>
        <CardContent sx={{ p: 2 }}>
          <Box display="flex" alignItems="center" mb={2}>
            <UpdateIcon sx={{ mr: 1, fontSize: '1.2rem' }} />
            <Box sx={{ display: 'flex', flexDirection: 'column',}}>
              <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
                Signal & Analysis
              </Typography>
              <Typography variant="caption" color="text.secondary">
                      Last updated: {lastUpdate}
                </Typography>
            </Box>
            
          </Box>

          {/* Signal Details */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              mb: 1.5,
              gap: 1,
              flexWrap: 'wrap'
            }}>
              <Chip
                label={data?.data?.signal?.signal || 'HOLD'}
                color={getSignalColor(data?.data?.signal?.signal)}
                size="small"
              />
              <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                {data?.data?.signal?.symbol || 'BTC/USDT'} • {data?.data?.signal?.timeframe || '5m'}
              </Typography>
              <Chip
                label={`${Math.round((data?.data?.signal?.confidence || 0) * 100)}% Confidence`}
                color="success"
                size="small"
              />
            </Box>

            {/* Entry/Stop/Target */}
            <Grid container spacing={1} sx={{ mb: 1.5 }}>
              <Grid item xs={4}>
                <Typography variant="caption" color="text.secondary">Entry</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  {data?.data?.signal?.entry_price ? `$${data.data.signal.entry_price.toLocaleString()}` : 'None'}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption" color="text.secondary">Stop Loss</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  {data?.data?.signal?.stop_loss ? `$${data.data.signal.stop_loss.toLocaleString()}` : 'None'}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption" color="text.secondary">Take Profit</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  {data?.data?.signal?.take_profit ? `$${data.data.signal.take_profit.toLocaleString()}` : 'None'}
                </Typography>
              </Grid>
            </Grid>

            {/* Position Management */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                Position Management
              </Typography>
              <Chip
                label={data?.data?.signal?.position_management?.action || 'MAINTAIN'}
                color={getActionColor(data?.data?.signal?.position_management?.action)}
                size="small"
                sx={{ fontSize: '0.7rem', mr: 1 }}
              />
              <Typography variant="caption" sx={{ ml: 1 }}>
                R/R: {data?.data?.signal?.position_management?.risk_reward_ratio?.toFixed(2) || '0.00'}
              </Typography>
            </Box>
            
            <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary"> Trailing Stop Loss</Typography>
                <Typography variant="body2" sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  wordBreak: 'break-word'
                }}>
                  {data?.data?.signal?.position_management?.stop_loss_adjustment || 'None'}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary"> Trailing Take Profit</Typography>
                <Typography variant="body2" sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  wordBreak: 'break-word'
                }}>
                  {data?.data?.signal?.position_management?.take_profit_adjustment || 'None'}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Started on</Typography>
                <Typography variant="body2" sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  wordBreak: 'break-word'
                }}>
                  {data?.data?.positions?.[0]?.created_at ? 
                    `${new Date(data.data.positions[0].created_at).toLocaleString()} (${formatDuration(data.data.positions[0].created_at)}m)` 
                    : 'None'}
                </Typography>
              </Grid>
              
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Analysis History */}
          {analysisHistory.length > 0 ? (
            renderAnalysisHistory()
          ) : (
            <Alert severity="info" sx={{ fontSize: '0.8rem' }}>No analysis history available</Alert>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderMobilePositions = () => {
    if (!data?.data?.positions || data.data.positions.length === 0) {
      return (
        <Alert severity="info" sx={{ fontSize: '0.8rem' }}>No active positions</Alert>
      );
    }

    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {data.data.positions.map((position) => (
          <Card key={position.id} sx={{ bgcolor: theme.palette.background.default }}>
            <CardContent sx={{ p: 2 }}>
              {/* Header - Symbol and Type */}
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                mb: 1.5 
              }}>
                <Typography sx={{ 
                  fontSize: '1rem',
                  fontWeight: 'bold' 
                }}>
                  {position.symbol}
                </Typography>
                <Chip
                  label={position.position_type}
                  color={position.position_type === 'LONG' ? 'success' : 'error'}
                  size="small"
                  sx={{ fontSize: '0.7rem' }}
                />
              </Box>

              {/* Prices Grid */}
              <Grid container spacing={1} sx={{ mb: 1.5 }}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Entry Price</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                    {formatPrice(position.entry_price)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Current Price</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                    {formatPrice(position.current_price)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Stop Loss</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                    {formatPrice(position.stop_loss)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Take Profit</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                    {formatPrice(position.take_profit)}
                  </Typography>
                </Grid>
              </Grid>

              {/* Footer - Size, P&L, Status */}
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                alignItems: 'center',
                pt: 1,
                borderTop: 1,
                borderColor: 'divider'
              }}>
                <Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Size
                  </Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                    {position.size}
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary" display="block">
                    P&L
                  </Typography>
                  <Typography sx={{ 
                    fontSize: '0.75rem',
                    color: position.pnl >= 0 ? theme.palette.success.main : theme.palette.error.main,
                    fontWeight: 'bold'
                  }}>
                    ${Math.abs(position.pnl).toLocaleString()}
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'right' }}>
                  <Chip
                    label={position.status}
                    color={position.status === 'ACTIVE' ? 'success' : 'default'}
                    size="small"
                    sx={{ fontSize: '0.7rem' }}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 } }}>
      {/* Connection Status */}
      {!isConnected && (
        <Alert 
          severity={error ? "error" : "warning"}
          action={
            <Button color="inherit" size="small" onClick={reconnect}>
              Retry
            </Button>
          }
          sx={{ mb: 2, fontSize: isMobile ? '0.8rem' : 'inherit' }}
        >
          {error ? error.message : "Connecting..."}
        </Alert>
      )}

      {/* Dashboard Content */}
      {isMobile ? (
        // Mobile Layout
        <>
          {renderMobileMarketOverview()}
          {renderMobileSignalAnalysis()}
          
          {/* Active Positions */}
          <Card sx={{ bgcolor: 'background.paper' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem' }}>
                Active Positions
              </Typography>
              {renderMobilePositions()}
            </CardContent>
          </Card>
        </>
      ) : (
        // Desktop Layout - Keep existing grid layout
        <Grid container spacing={{ xs: 2, md: 3 }}>
          {/* Market Overview */}
          <Grid item xs={12} lg={6}>
            <Card sx={{ height: '100%', bgcolor: 'background.paper', boxShadow: 3 }}>
              <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
                <Box display="flex" alignItems="center" mb={2}>
                  <TimelineIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ fontSize: { xs: '1.1rem', sm: '1.25rem' } }}>
                    Market Overview
                  </Typography>
                </Box>
                {data?.data?.market_data && (
                  <Grid container spacing={{ xs: 1, sm: 2 }}>
                    <Grid item xs={12}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="h4" sx={{ 
                          fontWeight: 'bold',
                          fontSize: { xs: '1.5rem', sm: '2rem', md: '2.5rem' },
                          wordBreak: 'break-word'
                        }}>
                          ${data.data.market_data.current_price.toLocaleString()}
                        </Typography>
                        <Box display="flex" alignItems="center" mt={1}>
                          {data.data.market_data.price_change_24h >= 0 ? (
                            <TrendingUpIcon color="success" sx={{ fontSize: { xs: '1.2rem', sm: '1.5rem' } }} />
                          ) : (
                            <TrendingDownIcon color="error" sx={{ fontSize: { xs: '1.2rem', sm: '1.5rem' } }} />
                          )}
                          <Typography 
                            variant="body1" 
                            color={data.data.market_data.price_change_24h >= 0 ? 'success.main' : 'error.main'}
                            sx={{ ml: 1, fontSize: { xs: '0.9rem', sm: '1rem' } }}
                          >
                            {data.data.market_data.price_change_24h.toFixed(2)}%
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>

                    {/* Volume Indicators */}
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          24h Volume
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' },
                          wordBreak: 'break-word'
                        }}>
                          ${data.data.market_data.volume_24h.toLocaleString()}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          4h Volume
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' },
                          wordBreak: 'break-word'
                        }}>
                          ${data.data.market_data.volume_4h.toLocaleString()}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          1h Volume
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' },
                          wordBreak: 'break-word'
                        }}>
                          ${data.data.market_data.volume_1h.toLocaleString()}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          15m Volume
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' },
                          wordBreak: 'break-word'
                        }}>
                          ${data.data.market_data.volume_15m.toLocaleString()}
                        </Typography>
                      </Box>
                    </Grid>

                    {/* Technical Indicators */}
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          RSI
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' }
                        }}>
                          {formatIndicator(data?.data?.market_data?.rsi)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          MACD
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' }
                        }}>
                          {formatIndicator(data?.data?.market_data?.macd)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          EMA 8
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' }
                        }}>
                          {formatIndicator(data?.data?.market_data?.ema)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ 
                        p: { xs: 1, sm: 2 }, 
                        bgcolor: theme.palette.background.default,
                        borderRadius: 1,
                        minHeight: { xs: '80px', sm: '100px' }
                      }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom noWrap>
                          SMA 8
                        </Typography>
                        <Typography variant="body2" sx={{ 
                          fontSize: { xs: '0.75rem', sm: '0.875rem' }
                        }}>
                          {formatIndicator(data?.data?.market_data?.sma)}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Signal and Analysis Section */}
          <Grid item xs={12} lg={6}>
            <Card sx={{ height: '100%', bgcolor: 'background.paper', boxShadow: 3 }}>
              <CardHeader
                title={
                  <Box display="flex" alignItems="center">
                    <UpdateIcon sx={{ mr: 1 }} />
                    <Typography variant="h6" sx={{ fontSize: { xs: '1.1rem', sm: '1.25rem' } }}>
                      Signal & Analysis
                    </Typography>
                  </Box>
                }
                action={
                  <Typography variant="caption" color="text.secondary">
                    Last updated: {lastUpdate}
                  </Typography>
                }
                sx={{ p: { xs: 2, sm: 3 } }}
              />
              <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
                {/* Signal Details */}
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    mb: 2,
                    flexWrap: 'wrap',
                    gap: 1
                  }}>
                    <Chip
                      label={data?.data?.signal?.signal || 'HOLD'}
                      color={getSignalColor(data?.data?.signal?.signal)}
                      size="small"
                    />
                    <Typography variant="body2" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
                      {data?.data?.signal?.symbol || 'BTC/USDT'} • {data?.data?.signal?.timeframe || '5m'}
                    </Typography>
                    <Chip
                      label={`${Math.round((data?.data?.signal?.confidence || 0) * 100)}% Confidence`}
                      color="success"
                      size="small"
                      sx={{ ml: { xs: 0, sm: 'auto' } }}
                    />
                  </Box>

                  {/* Signal Management Grid */}
                  <Grid container spacing={{ xs: 1, sm: 2 }} sx={{ mb: 2 }}>
                    <Grid item xs={6} sm={4}>
                      <Typography variant="caption" color="text.secondary">Entry Price</Typography>
                      <Typography variant="body2" sx={{ 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        wordBreak: 'break-word'
                      }}>
                        {data?.data?.signal?.entry_price ? `$${data.data.signal.entry_price.toLocaleString()}` : 'None'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Typography variant="caption" color="text.secondary">Take Profit</Typography>
                      <Typography variant="body2" sx={{ 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        wordBreak: 'break-word'
                      }}>
                        {data?.data?.signal?.take_profit ? `$${data.data.signal.take_profit.toLocaleString()}` : 'None'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={4}>
                      <Typography variant="caption" color="text.secondary">Stop Loss</Typography>
                      <Typography variant="body2" sx={{ 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        wordBreak: 'break-word'
                      }}>
                        {data?.data?.signal?.stop_loss ? `$${data.data.signal.stop_loss.toLocaleString()}` : 'None'}
                      </Typography>
                    </Grid>
                  </Grid>

                  {/* Position Management Grid */}
                  <Grid container spacing={{ xs: 1, sm: 2 }}>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">Action</Typography>
                      <Typography variant="body2">
                        <Chip
                          label={data?.data?.signal?.position_management?.action || 'MAINTAIN'}
                          color={getActionColor(data?.data?.signal?.position_management?.action)}
                          size="small"
                          sx={{ fontSize: { xs: '0.7rem', sm: '0.8rem' } }}
                        />
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary"> Trailing Stop Loss</Typography>
                      <Typography variant="body2" sx={{ 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        wordBreak: 'break-word'
                      }}>
                        {data?.data?.signal?.position_management?.stop_loss_adjustment || 'None'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary"> Trailing Take Profit</Typography>
                      <Typography variant="body2" sx={{ 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        wordBreak: 'break-word'
                      }}>
                        {data?.data?.signal?.position_management?.take_profit_adjustment || 'None'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">Risk/Reward</Typography>
                      <Typography variant="body2" sx={{ 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' }
                      }}>
                        {data?.data?.signal?.position_management?.risk_reward_ratio?.toFixed(2) || '0.00'}
                      </Typography>
                    </Grid>
                    
              <Grid item md={6}>
                <Typography variant="caption" color="text.secondary">Started on</Typography>
                <Typography variant="body2" sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' }
                }}>
                  {data?.data?.positions?.[0]?.created_at ? 
                    `${new Date(data.data.positions[0].created_at).toLocaleString()} (${formatDuration(data.data.positions[0].created_at)}m)` 
                    : 'None'}
                </Typography>
              </Grid>
                  </Grid>
                </Box>

                <Divider sx={{ my: 2 }} />

                {/* Analysis History */}
                {analysisHistory.length > 0 ? (
                  renderAnalysisHistory()
                ) : (
                  <Alert severity="info">No analysis history available</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Active Positions */}
          <Grid item xs={12}>
            <Card sx={{ bgcolor: 'background.paper', boxShadow: 3 }}>
              <CardContent sx={{ p: { xs: 1, sm: 2, md: 3 } }}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: { xs: '1.1rem', sm: '1.25rem' } }}>
                  Active Positions
                </Typography>
                {data?.data?.positions && data.data.positions.length > 0 ? (
                  <Box sx={{ overflowX: 'auto' }}>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Symbol</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Type</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Entry</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Current</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Stop</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Target</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Size</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>P&L</TableCell>
                            <TableCell sx={{ whiteSpace: 'nowrap', fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Status</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {data.data.positions.map((position) => (
                            <TableRow key={position.id}>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' }
                              }}>
                                {position.symbol}
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={position.position_type}
                                  color={position.position_type === 'LONG' ? 'success' : 'error'}
                                  size="small"
                                  sx={{ fontSize: { xs: '0.7rem', sm: '0.8rem' } }}
                                />
                              </TableCell>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' }
                              }}>
                                {formatPrice(position.entry_price)}
                              </TableCell>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' }
                              }}>
                                {formatPrice(position.current_price)}
                              </TableCell>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' }
                              }}>
                                {formatPrice(position.stop_loss)}
                              </TableCell>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' }
                              }}>
                                {formatPrice(position.take_profit)}
                              </TableCell>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' }
                              }}>
                                {position.size}
                              </TableCell>
                              <TableCell sx={{ 
                                whiteSpace: 'nowrap', 
                                fontSize: { xs: '0.75rem', sm: '0.875rem' },
                                color: position.pnl >= 0 ? theme.palette.success.main : theme.palette.error.main
                              }}>
                                ${Math.abs(position.pnl).toLocaleString()}
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={position.status}
                                  color={position.status === 'ACTIVE' ? 'success' : 'default'}
                                  size="small"
                                  sx={{ fontSize: { xs: '0.7rem', sm: '0.8rem' } }}
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                ) : (
                  <Alert severity="info">No active positions</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default Dashboard; 