import React, { useState, useEffect } from 'react';
import { 
  Card, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, 
  Typography, Box, Chip, TextField, MenuItem, Button, Alert 
} from '@mui/material';
import { Grid } from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { Theme } from '@mui/material/styles';
import { SxProps } from '@mui/system';

interface Position {
  symbol: string;
  type: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  closed_reason: string;
  duration: string;
  closed_at: string;
}

interface Filters {
  startDate: Date | null;
  endDate: Date | null;
  symbol: string;
  outcome: 'all' | 'profit' | 'loss';
}

interface Stats {
  total: number;
  wins: number;
  losses: number;
  totalPnL: number;
}

const PositionHistory: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const apiUrl = process.env.REACT_APP_API_URL || 'http://backend:8000';
  const [filters, setFilters] = useState<Filters>({
    startDate: new Date(new Date().setHours(0, 0, 0, 0)),
    endDate: new Date(),
    symbol: '',
    outcome: 'all'
  });

  const fetchPositions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiUrl}/api/positions/history?` + new URLSearchParams({
        start_date: filters.startDate?.toISOString() || new Date().toISOString(),
        end_date: filters.endDate?.toISOString() || new Date().toISOString(),
        symbol: filters.symbol,
        outcome: filters.outcome
      }));
      
      if (!response.ok) throw new Error('Failed to fetch positions');
      
      const data = await response.json();
      setPositions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPositions();
  }, []);

  const getClosedReasonColor = (reason: string | undefined): 'success' | 'error' | 'warning' | 'default' => {
    switch (reason?.toLowerCase()) {
      case 'take_profit': return 'success';
      case 'stop_loss': return 'error';
      case 'manual': return 'warning';
      default: return 'default';
    }
  };

  const formatPnL = (pnl: number) => {
    const value = parseFloat(pnl.toString());
    const color = value >= 0 ? '#4caf50' : '#f44336';
    return <Typography component="span" style={{ color }}>${Math.abs(value).toFixed(2)}</Typography>;
  };

  const calculateStats = (): Stats => {
    if (!positions.length) return { total: 0, wins: 0, losses: 0, totalPnL: 0 };
    
    return positions.reduce((acc: Stats, pos) => {
      const pnl = parseFloat(pos.pnl.toString());
      return {
        total: acc.total + 1,
        wins: pnl >= 0 ? acc.wins + 1 : acc.wins,
        losses: pnl < 0 ? acc.losses + 1 : acc.losses,
        totalPnL: acc.totalPnL + pnl
      };
    }, { total: 0, wins: 0, losses: 0, totalPnL: 0 });
  };

  const stats = calculateStats();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>Position History</Typography>
      
      {/* Filters */}
      <Card sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="Start Date"
                value={filters.startDate}
                onChange={(newValue: Date | null) => setFilters(prev => ({ ...prev, startDate: newValue }))}
                slotProps={{ textField: { fullWidth: true } }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="End Date"
                value={filters.endDate}
                onChange={(newValue: Date | null) => setFilters(prev => ({ ...prev, endDate: newValue }))}
                slotProps={{ textField: { fullWidth: true } }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              fullWidth
              label="Symbol"
              value={filters.symbol}
              onChange={(e) => setFilters(prev => ({ ...prev, symbol: e.target.value }))}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              fullWidth
              select
              label="Outcome"
              value={filters.outcome}
              onChange={(e) => setFilters(prev => ({ ...prev, outcome: e.target.value as 'all' | 'profit' | 'loss' }))}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="profit">Profit</MenuItem>
              <MenuItem value="loss">Loss</MenuItem>
            </TextField>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button 
              fullWidth 
              variant="contained" 
              onClick={fetchPositions}
              disabled={loading}
            >
              Apply Filters
            </Button>
          </Grid>
        </Grid>
      </Card>

      {/* Stats Summary */}
      <Card sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={6} sm={3}>
            <Typography variant="subtitle2" color="text.secondary">Total Trades</Typography>
            <Typography variant="h6">{stats.total}</Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="subtitle2" color="text.secondary">Win Rate</Typography>
            <Typography variant="h6">
              {stats.total ? ((stats.wins / stats.total) * 100).toFixed(1) : 0}%
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="subtitle2" color="text.secondary">Total P&L</Typography>
            <Typography variant="h6" color={stats.totalPnL >= 0 ? 'success.main' : 'error.main'}>
              ${stats.totalPnL.toFixed(2)}
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="subtitle2" color="text.secondary">Avg. P&L per Trade</Typography>
            <Typography variant="h6" color={stats.totalPnL >= 0 ? 'success.main' : 'error.main'}>
              ${stats.total ? (stats.totalPnL / stats.total).toFixed(2) : '0.00'}
            </Typography>
          </Grid>
        </Grid>
      </Card>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Positions Table */}
      <Card sx={{ p: 2 }}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Entry</TableCell>
                <TableCell>Exit</TableCell>
                <TableCell>P&L</TableCell>
                <TableCell>Close Reason</TableCell>
                <TableCell>Duration</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {positions.map((position, index) => (
                <TableRow key={index}>
                  <TableCell>{new Date(position.closed_at).toLocaleString()}</TableCell>
                  <TableCell>{position.symbol}</TableCell>
                  <TableCell>
                    <Typography
                      component="span"
                      color={position.type.toLowerCase() === 'long' ? 'success.main' : 'error.main'}
                    >
                      {position.type}
                    </Typography>
                  </TableCell>
                  <TableCell>${position.entry_price}</TableCell>
                  <TableCell>${position.exit_price}</TableCell>
                  <TableCell>{formatPnL(position.pnl)}</TableCell>
                  <TableCell>
                    <Chip
                      label={position.closed_reason}
                      size="small"
                      color={getClosedReasonColor(position.closed_reason)}
                      sx={{ opacity: 0.9 }}
                    />
                  </TableCell>
                  <TableCell>{position.duration}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>
    </Box>
  );
};

export default PositionHistory; 