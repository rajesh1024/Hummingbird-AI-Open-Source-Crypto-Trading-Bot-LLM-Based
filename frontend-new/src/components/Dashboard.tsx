import React, { useEffect, useState, useRef, useCallback } from 'react';
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
  useMediaQuery,
  Tabs,
  Tab
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
import TradingViewWidget from './TradingViewWidget';
import { convertUTCToIST, formatDuration } from '../utils/dateUtils';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import ShowChartIcon from '@mui/icons-material/ShowChart';

interface Position {
  id: number;
  symbol: string;
  position_type: string;
  status: string;
  entry_price: number;
  current_price?: number;
  stop_loss?: number;
  take_profit?: number;
  size: number;
  pnl: number;
  created_at: string;
  closed_at?: string;
}

interface MarketData {
  symbol: string;
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  technical_indicators: {
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
  };
  trading_mode: string;
}

interface PositionManagement {
  position_type: string;
  entry_price: string;
  stop_loss: string;
  take_profit: string;
  action: string;
  confidence: number;
  [key: string]: string | number;
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
  position_management?: PositionManagement;
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

interface WebSocketMessage {
  type: 'initial' | 'update' | 'error';
  data?: {
    market_data: MarketData;
    positions: Position[];
    signal: Signal;
  };
  message?: string;
}

interface DashboardProps {
  // Add any props if needed
}

const getSignalProperty = (signal: Signal | null, property: string): string | number | null => {
  if (!signal?.position_management) return null;
  return signal.position_management[property] ?? null;
};

const getChangeColor = (value: number | string | null | undefined): string => {
  if (value === null || value === undefined) return 'text.primary';
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return numValue >= 0 ? 'success.main' : 'error.main';
};

const formatNumber = (value: number | string | null | undefined): number | null => {
  if (value === null || value === undefined) return null;
  return typeof value === 'string' ? parseFloat(value) : value;
};

const formatPrice = (value: number | string | null | undefined): string => {
  const numValue = formatNumber(value);
  if (numValue === null) return 'N/A';
  return numValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const formatPercentage = (value: number | string | null | undefined): string => {
  const numValue = formatNumber(value);
  if (numValue === null) return 'N/A';
  return `${numValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
};

const formatSignalValue = (value: string | number | null): string => {
  if (value === null) return 'N/A';
  const numValue = formatNumber(value);
  if (numValue !== null) {
    return numValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return String(value);
};

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [currentSymbol, setCurrentSymbol] = useState<string>('BTC/USDT');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [signal, setSignal] = useState<Signal | null>(null);
  const [isSymbolActive, setIsSymbolActive] = useState<boolean>(false);
  const [symbolStatus, setSymbolStatus] = useState<string>('');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [analysisHistory, setAnalysisHistory] = useState<Array<Analysis>>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [wsError, setWsError] = useState<Error | null>(null);
  const [lastDataReceived, setLastDataReceived] = useState<number>(Date.now());
  const [connectionStatus, setConnectionStatus] = useState<string>('Connecting...');
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const handleWebSocketMessage = useCallback((data: any) => {
    if (!data) return;

    const hasMarketData = data.data?.market_data;
    const hasPositions = data.data?.positions;
    const hasSignal = data.data?.signal;

    console.log('Processing WebSocket message:', {
      type: data.type,
      hasMarketData,
      hasPositions,
      hasSignal
    });

    if (data.type === 'error') {
      setError(data.message);
      setIsLoading(false);
      return;
    }

    if (hasMarketData) {
      console.log('Updating market data:', data.data.market_data);
      setMarketData(data.data.market_data);
    }

    if (hasPositions) {
      console.log(`Updating positions: ${data.data.positions.length} positions:`, data.data.positions);
      setPositions(data.data.positions);
    }

    if (hasSignal) {
      console.log('Updating signal:', data.data.signal);
      setSignal(data.data.signal);
    }

    if (data.type === 'initial' || data.type === 'update') {
      setLastUpdate(new Date());
      setError(null);
      setIsLoading(false);
    }
  }, []);

  const handleWebSocketClose = useCallback((event: CloseEvent) => {
    console.log('WebSocket closed for', currentSymbol);
    console.log('WebSocket close event - Code:', event.code, 'Clean:', event.wasClean, 'Reason:', event.reason);
    
    setConnectionStatus('Disconnected');
    
    if (event.code === 1000) {
      // Normal closure
      setConnectionStatus('Connection closed');
    } else {
      console.log('Reconnecting after close event. Code:', event.code, 'Clean:', event.wasClean);
      setConnectionStatus('Reconnecting...');
    }
  }, [currentSymbol]);

  const handleWebSocketError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
    setError('Connection error occurred');
    setConnectionStatus('Error');
    setIsLoading(false);
  }, []);

  const handleWebSocketOpen = useCallback(() => {
    console.log('WebSocket connected');
    setConnectionStatus('Connected');
    setError(null);
    setReconnectAttempts(0);
  }, []);

  const { isConnected } = useWebSocket(
    `ws://localhost:8000/ws/dashboard?symbol=${currentSymbol}`,
    {
      onMessage: handleWebSocketMessage,
      onClose: handleWebSocketClose,
      onError: handleWebSocketError,
      onOpen: handleWebSocketOpen,
      shouldReconnect: true,
      reconnectAttempts: 5,
      reconnectInterval: 10000
    }
  );

  // Update connection status based on isConnected
  useEffect(() => {
    if (isConnected) {
      setConnectionStatus('Connected');
    } else {
      setConnectionStatus('Disconnected');
      setIsLoading(true);
    }
  }, [isConnected]);

  // Reset states when symbol changes
  useEffect(() => {
    setMarketData(null);
    setPositions([]);
    setSignal(null);
    setError(null);
    setConnectionStatus('Connecting...');
    setReconnectAttempts(0);
    setIsLoading(true);
  }, [currentSymbol]);

  const getChangeColor = (value: number | undefined) => {
    if (!value) return 'text.primary';
    return value >= 0 ? 'success.main' : 'error.main';
  };

  // Handle symbol change
  const handleSymbolChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentSymbol(newValue === 0 ? 'ETH/USDT' : 'BTC/USDT');
    setIsLoading(true);
  };

  useEffect(() => {
    if (signal) {
      setAnalysisHistory(prevHistory => {
        const newAnalysis = {
          timestamp: signal.timestamp,
          symbol: signal.symbol,
          analysis: signal.reason,
          confidence: signal.confidence
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
      setLastUpdate(new Date());
    }
  }, [signal]);

  const handleLoadingChange = (loading: boolean) => {
    setIsLoading(loading);
  };

  // Add debug logging
  useEffect(() => {
    if (marketData) {
      console.log('Received market data:', marketData);
    }
    if (signal) {
      console.log('Received signal:', signal);
    }
  }, [marketData, signal]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 50) return 'warning';
    return 'error';
  };

  const getSignalColor = (signal: string | undefined) => {
    switch (signal) {
      case 'BUY':
        return 'success';
      case 'SELL':
        return 'error';
      case 'LONG':
        return 'success';
      case 'SHORT':
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
              maxHeight: '380px',
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
    if (!marketData) return null;
    
    // Extract and provide default values for nested properties
    const currentPrice = marketData?.current_price ?? 0;
    const priceChange = marketData?.price_change_24h ?? 0;
    const volume24h = marketData?.volume_24h ?? 0;
    const rsi = marketData?.technical_indicators?.RSI ?? 0;
    const ema8 = marketData?.technical_indicators?.EMA?.EMA8 ?? 0;
    
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          ${currentPrice.toLocaleString()}
        </Typography>
        <Box display="flex" alignItems="center" mt={0.5}>
          {priceChange >= 0 ? (
            <TrendingUpIcon color="success" sx={{ fontSize: '1.2rem' }} />
          ) : (
            <TrendingDownIcon color="error" sx={{ fontSize: '1.2rem' }} />
          )}
          <Typography 
            variant="body2" 
            color={priceChange >= 0 ? 'success.main' : 'error.main'}
            sx={{ ml: 0.5 }}
          >
            {priceChange.toFixed(2)}%
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
          <Box>
            <Typography variant="caption" color="text.secondary" noWrap>24h Volume</Typography>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              ${volume24h.toLocaleString()}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary" noWrap>RSI</Typography>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {rsi.toFixed(2)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary" noWrap>EMA 8</Typography>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              ${ema8.toLocaleString()}
            </Typography>
          </Box>
        </Box>
      </Box>
    );
  };

  const renderMobileSignalAnalysis = () => {
    if (!signal) return null;

    // Extract and provide default values for nested properties
    const signalType = signal?.signal ?? 'N/A';
    const confidence = signal?.confidence ?? 0;
    const symbol = signal?.symbol ?? 'N/A';
    const timeframe = signal?.timeframe ?? 'N/A';
    const entryPrice = signal?.entry_price ?? 0;
    const stopLoss = signal?.stop_loss ?? 0;
    const takeProfit = signal?.take_profit ?? 0;
    const positionAction = signal?.position_management?.action ?? 'N/A';
    const riskReward = signal?.position_management?.risk_reward_ratio ?? 0;
    const stopLossAdjustment = signal?.position_management?.stop_loss_adjustment ?? 'N/A';
    const takeProfitAdjustment = signal?.position_management?.take_profit_adjustment ?? 'N/A';

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
                Last updated: {lastUpdate.toLocaleTimeString()}
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
                label={signalType}
                color={getSignalColor(signalType)}
                size="small"
              />
              <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                {symbol} • {timeframe}
              </Typography>
              <Chip
                label={`${Math.round(confidence * 100)}% Confidence`}
                color="success"
                size="small"
              />
            </Box>

            {/* Entry/Stop/Target */}
            <Grid container spacing={1} sx={{ mb: 1.5 }}>
              <Grid item xs={4}>
                <Typography variant="caption" color="text.secondary">Entry</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  ${entryPrice.toLocaleString()}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption" color="text.secondary">Stop Loss</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  ${stopLoss.toLocaleString()}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="caption" color="text.secondary">Take Profit</Typography>
                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                  ${takeProfit.toLocaleString()}
                </Typography>
              </Grid>
            </Grid>

            {/* Position Management */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                Position Management
              </Typography>
              <Chip
                label={positionAction}
                color={getActionColor(positionAction)}
                size="small"
                sx={{ fontSize: '0.7rem', mr: 1 }}
              />
              <Typography variant="caption" sx={{ ml: 1 }}>
                R/R: {riskReward.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </Typography>
            </Box>
            
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary"> Trailing Stop Loss</Typography>
              <Typography variant="body2" sx={{ 
                fontSize: { xs: '0.75rem', sm: '0.875rem' },
                wordBreak: 'break-word'
              }}>
                {stopLossAdjustment}
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="caption" color="text.secondary"> Trailing Take Profit</Typography>
              <Typography variant="body2" sx={{ 
                fontSize: { xs: '0.75rem', sm: '0.875rem' },
                wordBreak: 'break-word'
              }}>
                {takeProfitAdjustment}
              </Typography>
            </Grid>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const renderMobilePositions = () => {
    if (positions.length === 0) {
      return (
        <Alert severity="info" sx={{ fontSize: '0.8rem' }}>No active positions</Alert>
      );
    }

    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {positions.map((position) => (
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

  const renderMarketOverview = (): JSX.Element => {
    const currentPrice = marketData?.current_price;
    const priceChange24h = marketData?.price_change_24h;

    return (
      <Box sx={{ transition: 'opacity 0.3s ease-in-out' }}>
        <CardHeader
          title={
            <Box display="flex" alignItems="center">
              <ShowChartIcon sx={{ mr: 1 }} />
              <Typography variant="h6">Market Overview</Typography>
            </Box>
          }
        />
        <Box p={2}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">Current Price</Typography>
                <Typography variant="h6">${formatPrice(currentPrice)}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">24h Change</Typography>
                <Typography variant="h6" color={getChangeColor(priceChange24h)}>
                  {formatPercentage(priceChange24h)}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Box>
    );
  };

  const renderSignalAnalysis = (): JSX.Element => {
    return (
      <Box sx={{ transition: 'opacity 0.3s ease-in-out' }}>
        {signal && (
          <>
            <CardHeader
              title={
                <Box display="flex" alignItems="center">
                  <SignalCellularAltIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Signal Analysis</Typography>
                </Box>
              }
            />
            <Box p={2}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary">Position</Typography>
                    <Typography variant="h6">{formatSignalValue(getSignalProperty(signal, 'position_type'))}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary">Entry Price</Typography>
                    <Typography variant="h6">${formatSignalValue(getSignalProperty(signal, 'entry_price'))}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary">Stop Loss</Typography>
                    <Typography variant="h6">${formatSignalValue(getSignalProperty(signal, 'stop_loss'))}</Typography>
                  </Paper>
                </Grid>
                {getSignalProperty(signal, 'confidence') !== null && (
                  <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary">Confidence</Typography>
                      <Typography variant="h6">
                        {formatPercentage(getSignalProperty(signal, 'confidence') as number)}
                      </Typography>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            </Box>
          </>
        )}
      </Box>
    );
  };

  const renderPositions = (): JSX.Element => {
    return (
      <Box sx={{ transition: 'opacity 0.3s ease-in-out' }}>
        {positions.length > 0 ? (
          <Box sx={{ overflowX: 'auto' }}>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell>Entry Price</TableCell>
                    <TableCell>Current Price</TableCell>
                    <TableCell>Stop Loss</TableCell>
                    <TableCell>Take Profit</TableCell>
                    <TableCell>PnL</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {positions.map((position) => (
                    <TableRow key={position.id}>
                      <TableCell>{position.position_type}</TableCell>
                      <TableCell>${formatPrice(position.entry_price)}</TableCell>
                      <TableCell>${formatPrice(position.current_price)}</TableCell>
                      <TableCell>${formatPrice(position.stop_loss)}</TableCell>
                      <TableCell>${formatPrice(position.take_profit)}</TableCell>
                      <TableCell>
                        <Typography color={Number(position.pnl) >= 0 ? 'success.main' : 'error.main'}>
                          ${formatPrice(position.pnl)}
                        </Typography>
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
      </Box>
    );
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 } }}>
      {/* Static Symbol Tabs */}
      <Tabs
        value={currentSymbol === 'ETH/USDT' ? 0 : 1}
        onChange={handleSymbolChange}
        variant="fullWidth"
        sx={{ mb: 3 }}
      >
        <Tab label="ETH/USDT" />
        <Tab label="BTC/USDT" />
      </Tabs>

      {/* Always show connection status */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Connection Status</Typography>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Last Update: {lastUpdate.toLocaleTimeString()}
            </Typography>
          </Box>
        </Box>
        <Grid container spacing={2} mt={1}>
          <Grid item xs={12} sm={4}>
            <Typography variant="subtitle2" color="text.secondary">Status</Typography>
            <Typography variant="body1">{connectionStatus}</Typography>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="subtitle2" color="text.secondary">Symbol Status</Typography>
            <Typography variant="body1">{isConnected ? 'Running' : 'Connection closed'}</Typography>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="subtitle2" color="text.secondary">Reconnect Attempts</Typography>
            <Typography variant="body1">{reconnectAttempts} / 5</Typography>
          </Grid>
        </Grid>
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>

      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
          <Typography variant="body1" sx={{ ml: 2 }}>
            Loading dashboard data...
          </Typography>
        </Box>
      ) : (
        <>
          {!isConnected && !wsError && (
            <Box sx={{ mb: 2 }}>
              <Alert severity="warning">
                {symbolStatus || 'Connecting to dashboard data stream...'}
              </Alert>
            </Box>
          )}

          {isConnected && !isSymbolActive && (
            <Box sx={{ mb: 2 }}>
              <Alert severity="info">
                {symbolStatus}
              </Alert>
            </Box>
          )}

          {isConnected && (
            <Grid container spacing={3}>
              {/* Market Overview */}
              <Grid item xs={12} md={8}>
                <Card sx={{ bgcolor: 'background.paper' }}>
                  <CardContent>
                    {marketData ? renderMarketOverview() : (
                      <Alert severity="info">Waiting for market data...</Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Signal Analysis */}
              <Grid item xs={12} md={4}>
                <Card sx={{ bgcolor: 'background.paper' }}>
                  <CardContent>
                    {signal ? renderSignalAnalysis() : (
                      <Alert severity="info">Waiting for signal data...</Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Active Positions */}
              <Grid item xs={12}>
                <Card sx={{ bgcolor: 'background.paper' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Active Positions
                    </Typography>
                    {renderPositions()}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
        </>
      )}
    </Box>
  );
};

export default Dashboard; 