// import React, { useState } from 'react';
// import { 
//   Box, 
//   Grid, 
//   Typography, 
//   Paper, 
//   Divider,
//   Tooltip,
//   LinearProgress,
//   Card,
//   CardContent,
//   useTheme,
//   Chip,
//   CircularProgress,
//   Alert,
//   Button
// } from '@mui/material';
// import TrendingUpIcon from '@mui/icons-material/TrendingUp';
// import TrendingDownIcon from '@mui/icons-material/TrendingDown';
// import ShowChartIcon from '@mui/icons-material/ShowChart';
// import TimelineIcon from '@mui/icons-material/Timeline';
// import { useWebSocket } from '../hooks/useWebSocket';

// interface TechnicalIndicators {
//   RSI: number;
//   MACD: {
//     MACD: number;
//     Signal: number;
//     Histogram: number;
//   };
//   EMA: {
//     EMA8: number;
//     EMA21: number;
//     EMA50: number;
//     EMA200: number;
//   };
//   BB: {
//     Upper: number;
//     Middle: number;
//     Lower: number;
//   };
//   Volume: {
//     '1h': number;
//     '4h': number;
//     '1d': number;
//   };
//   ATR: number;
// }

// interface MarketData {
//   symbol: string;
//   current_price: number;
//   price_change_24h: number;
//   volume_24h: number;
//   technical_indicators: TechnicalIndicators;
//   trading_mode: string;
// }

// interface DashboardData {
//   type: string;
//   data: {
//     positions: any[];
//     market_data: MarketData;
//     signal: any;
//   };
// }

// const MarketOverview: React.FC = () => {
//   const [isInitialLoad, setIsInitialLoad] = useState(true);
//   const theme = useTheme();
//   const wsUrl = process.env.REACT_APP_WS_URL || 'ws://backend:8000/ws/dashboard';
//   const { data, error, isConnected, reconnect } = useWebSocket<DashboardData>(wsUrl);

//   if (isInitialLoad) {
//     return (
//       <Card>
//         <CardContent>
//           <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
//             <CircularProgress />
//           </Box>
//         </CardContent>
//       </Card>
//     );
//   }

//   if (error) {
//     return (
//       <Card>
//         <CardContent>
//           <Alert 
//             severity="error"
//             action={
//               <Button color="inherit" size="small" onClick={reconnect}>
//                 Retry
//               </Button>
//             }
//           >
//             {error.message}
//           </Alert>
//         </CardContent>
//       </Card>
//     );
//   }

//   if (!isConnected) {
//     return (
//       <Card>
//         <CardContent>
//           <Alert 
//             severity="warning"
//             action={
//               <Button color="inherit" size="small" onClick={reconnect}>
//                 Retry
//               </Button>
//             }
//           >
//             Connecting to market data...
//           </Alert>
//         </CardContent>
//       </Card>
//     );
//   }

//   if (!data?.data?.market_data) {
//     return (
//       <Card>
//         <CardContent>
//           <Typography variant="h6" gutterBottom>Market Overview</Typography>
//           <Typography color="text.secondary">No market data available</Typography>
//         </CardContent>
//       </Card>
//     );
//   }

//   const { 
//     symbol, 
//     current_price, 
//     price_change_24h, 
//     volume_24h, 
//     technical_indicators,
//     trading_mode 
//   } = data.data.market_data;

//   const isPriceUp = price_change_24h > 0;

//   const getRSIColor = (value: number) => {
//     if (value >= 70) return 'error';
//     if (value <= 30) return 'success';
//     return 'warning';
//   };

//   const getMACDStatus = (macd: number, signal: number, histogram: number) => {
//     if (histogram > 0 && macd > signal) return 'success';
//     if (histogram < 0 && macd < signal) return 'error';
//     return 'warning';
//   };

//   const getEMAStatus = (price: number, ema: number) => {
//     return price > ema ? 'success' : 'error';
//   };

//   const formatLargeNumber = (num: number) => {
//     if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
//     if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
//     if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
//     return num.toFixed(2);
//   };

//   return (
//     <Card>
//       <CardContent>
//         <Typography variant="h6" gutterBottom>Market Overview</Typography>
        
//         <Box sx={{ mb: 2 }}>
//           <Typography variant="subtitle2" color="text.secondary">Current Price</Typography>
//           <Typography variant="h4">
//             ${current_price.toLocaleString()}
//           </Typography>
//           <Box display="flex" alignItems="center" mt={1}>
//             <Chip
//               label={`${price_change_24h >= 0 ? '+' : ''}${price_change_24h.toFixed(2)}%`}
//               color={price_change_24h >= 0 ? 'success' : 'error'}
//               size="small"
//             />
//           </Box>
//         </Box>

//         <Box sx={{ mb: 2 }}>
//           <Typography variant="subtitle2" color="text.secondary">Volume (24h)</Typography>
//           <Typography variant="body1">
//             ${formatLargeNumber(volume_24h)}
//           </Typography>
//         </Box>

//         <Box sx={{ mb: 2 }}>
//           <Typography variant="subtitle2" color="text.secondary">Technical Indicators</Typography>
//           <Box display="flex" flexWrap="wrap" gap={2} mt={1}>
//             <Box>
//               <Typography variant="caption" color="text.secondary">RSI</Typography>
//               <Typography variant="body2">{technical_indicators.RSI.toFixed(2)}</Typography>
//             </Box>
//             <Box>
//               <Typography variant="caption" color="text.secondary">MACD</Typography>
//               <Typography variant="body2">{technical_indicators.MACD.MACD.toFixed(2)}</Typography>
//             </Box>
//             <Box>
//               <Typography variant="caption" color="text.secondary">EMA</Typography>
//               <Typography variant="body2">{technical_indicators.EMA.EMA8.toFixed(2)}</Typography>
//             </Box>
//             <Box>
//               <Typography variant="caption" color="text.secondary">SMA</Typography>
//               <Typography variant="body2">{technical_indicators.BB.Middle.toFixed(2)}</Typography>
//             </Box>
//           </Box>
//         </Box>

//         <Box>
//           <Typography variant="subtitle2" color="text.secondary">Volume Breakdown</Typography>
//           <Box display="flex" flexWrap="wrap" gap={2} mt={1}>
//             <Box>
//               <Typography variant="caption" color="text.secondary">4h</Typography>
//               <Typography variant="body2">${formatLargeNumber(technical_indicators.Volume['4h'])}</Typography>
//             </Box>
//             <Box>
//               <Typography variant="caption" color="text.secondary">1h</Typography>
//               <Typography variant="body2">${formatLargeNumber(technical_indicators.Volume['1h'])}</Typography>
//             </Box>
//             <Box>
//               <Typography variant="caption" color="text.secondary">1d</Typography>
//               <Typography variant="body2">${formatLargeNumber(technical_indicators.Volume['1d'])}</Typography>
//             </Box>
//           </Box>
//         </Box>
//       </CardContent>
//     </Card>
//   );
// };

// export default MarketOverview; 