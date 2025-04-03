import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  TextField,
  MenuItem,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
  useTheme,
  useMediaQuery,
  Card,
  Stack,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import FilterListIcon from '@mui/icons-material/FilterList';
import RefreshIcon from '@mui/icons-material/Refresh';
import { usePositionHistory } from '../hooks/usePositionHistory';

const PositionHistory: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  const [symbolFilter, setSymbolFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');

  const { positions, loading, error, refetch } = usePositionHistory();

  const uniqueSymbols = Array.from(new Set(positions.map(p => p.symbol)));
  const positionTypes = ['LONG', 'SHORT'];

  // Apply filters when they change
  useEffect(() => {
    refetch({
      startDate,
      endDate,
      symbol: symbolFilter || undefined,
      type: typeFilter || undefined
    });
  }, [startDate, endDate, symbolFilter, typeFilter]);

  const totalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0);
  const winningTrades = positions.filter(pos => pos.pnl > 0).length;
  const losingTrades = positions.filter(pos => pos.pnl < 0).length;
  const winRate = positions.length > 0 
    ? ((winningTrades / positions.length) * 100).toFixed(1)
    : '0.0';

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">Error loading position history: {error.message}</Alert>
      </Box>
    );
  }

  const renderMobilePosition = (position: any, index: number) => (
    <Card key={index} sx={{ mb: 2, p: 2 }}>
      <Stack spacing={1}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">{position.symbol}</Typography>
          <Typography
            color={position.type === 'LONG' ? 'success.main' : 'error.main'}
          >
            {position.type}
          </Typography>
        </Box>
        <Box display="flex" justifyContent="space-between">
          <Typography color="text.secondary">Entry</Typography>
          <Typography>${position.entry_price.toLocaleString()}</Typography>
        </Box>
        <Box display="flex" justifyContent="space-between">
          <Typography color="text.secondary">Exit</Typography>
          <Typography>${position.exit_price.toLocaleString()}</Typography>
        </Box>
        <Box display="flex" justifyContent="space-between">
          <Typography color="text.secondary">P&L</Typography>
          <Typography color={position.pnl >= 0 ? 'success.main' : 'error.main'}>
            ${Math.abs(position.pnl).toLocaleString()}
          </Typography>
        </Box>
        <Box display="flex" justifyContent="space-between">
          <Typography color="text.secondary">Duration</Typography>
          <Typography>{position.duration}</Typography>
        </Box>
        <Box display="flex" justifyContent="space-between">
          <Typography color="text.secondary">Reason</Typography>
          <Typography>{position.closed_reason}</Typography>
        </Box>
      </Stack>
    </Card>
  );

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 } }}>
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>Position History</Typography>
      
      {/* Filters */}
      <Paper sx={{ p: { xs: 1.5, sm: 2 }, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="Start Date"
                value={startDate}
                onChange={(newValue) => setStartDate(newValue)}
                slotProps={{ 
                  textField: { 
                    fullWidth: true, 
                    size: "small",
                    sx: { mb: { xs: 1, sm: 0 } }
                  } 
                }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="End Date"
                value={endDate}
                onChange={(newValue) => setEndDate(newValue)}
                slotProps={{ 
                  textField: { 
                    fullWidth: true, 
                    size: "small",
                    sx: { mb: { xs: 1, sm: 0 } }
                  } 
                }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={6} md={2}>
            <TextField
              select
              fullWidth
              size="small"
              label="Symbol"
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              {uniqueSymbols.map(symbol => (
                <MenuItem key={symbol} value={symbol}>{symbol}</MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={6} md={2}>
            <TextField
              select
              fullWidth
              size="small"
              label="Type"
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              {positionTypes.map(type => (
                <MenuItem key={type} value={type}>{type}</MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} md={2}>
            <Box display="flex" justifyContent="flex-end">
              <Tooltip title="Reset Filters">
                <IconButton 
                  onClick={() => {
                    setStartDate(null);
                    setEndDate(null);
                    setSymbolFilter('');
                    setTypeFilter('');
                  }}
                  size="small"
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Stats Summary */}
      <Paper sx={{ p: { xs: 1.5, sm: 2 }, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={2}>
            <Typography variant="subtitle2" color="text.secondary">Total Trades</Typography>
            <Typography variant="h6">{positions.length}</Typography>
          </Grid>
          <Grid item xs={6} sm={2}>
            <Typography variant="subtitle2" color="text.secondary">Win Rate</Typography>
            <Typography variant="h6">{winRate}%</Typography>
          </Grid>
          <Grid item xs={6} sm={2}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>Profit Trades</Typography>
            <Typography variant="h6" color="success.main">{winningTrades}</Typography>
          </Grid>
          <Grid item xs={6} sm={2}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>Loss Trades</Typography>
            <Typography variant="h6" color="error.main">{losingTrades}</Typography>
          </Grid>
          <Grid item xs={6} sm={2}>
            <Typography variant="subtitle2" color="text.secondary">Total P&L</Typography>
            <Typography variant="h6" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
              ${totalPnL.toLocaleString()}
            </Typography>
          </Grid>
          <Grid item xs={6} sm={2}>
            <Typography variant="subtitle2" color="text.secondary">Avg P&L/Trade</Typography>
            <Typography variant="h6" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
              ${positions.length ? (totalPnL / positions.length).toLocaleString() : '0'}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Positions List/Table */}
      {isMobile ? (
        <Stack spacing={2}>
          {positions.map((position, index) => renderMobilePosition(position, index))}
          {positions.length === 0 && (
            <Alert severity="info">No positions found</Alert>
          )}
        </Stack>
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Type</TableCell>
                <TableCell align="right">Entry</TableCell>
                <TableCell align="right">Exit</TableCell>
                <TableCell align="right">P&L</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Reason</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {positions.map((position, index) => (
                <TableRow key={index}>
                  <TableCell>{position.symbol}</TableCell>
                  <TableCell>
                    <Typography
                      color={position.type === 'LONG' ? 'success.main' : 'error.main'}
                    >
                      {position.type}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    ${position.entry_price.toLocaleString()}
                  </TableCell>
                  <TableCell align="right">
                    ${position.exit_price.toLocaleString()}
                  </TableCell>
                  <TableCell align="right">
                    <Typography
                      color={position.pnl >= 0 ? 'success.main' : 'error.main'}
                    >
                      ${Math.abs(position.pnl).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>{position.duration}</TableCell>
                  <TableCell>{position.closed_reason}</TableCell>
                </TableRow>
              ))}
              {positions.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    <Typography color="text.secondary">No positions found</Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default PositionHistory; 