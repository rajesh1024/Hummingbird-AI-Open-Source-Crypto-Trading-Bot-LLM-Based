import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
} from '@mui/material';

interface Position {
  id: string;
  symbol: string;
  position_type: string;
  entry_price: number;
  current_price: number;
  pnl: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
  position_strength: number;
  status: string;
}

interface ActivePositionsProps {
  positions: Position[];
}

const ActivePositions: React.FC<ActivePositionsProps> = ({ positions }) => {
  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell>Type</TableCell>
            <TableCell align="right">Entry</TableCell>
            <TableCell align="right">Current</TableCell>
            <TableCell align="right">PnL</TableCell>
            <TableCell align="right">RR</TableCell>
            <TableCell align="right">Strength</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((position) => (
            <TableRow key={position.id}>
              <TableCell>{position.symbol}</TableCell>
              <TableCell>
                <Typography
                  color={position.position_type === 'LONG' ? 'success.main' : 'error.main'}
                >
                  {position.position_type}
                </Typography>
              </TableCell>
              <TableCell align="right">
                ${position.entry_price.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </TableCell>
              <TableCell align="right">
                ${position.current_price.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </TableCell>
              <TableCell align="right">
                <Typography
                  color={position.pnl >= 0 ? 'success.main' : 'error.main'}
                >
                  ${Math.abs(position.pnl).toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </Typography>
              </TableCell>
              <TableCell align="right">
                {position.risk_reward_ratio.toFixed(2)}
              </TableCell>
              <TableCell align="right">
                {(position.position_strength * 100).toFixed(1)}%
              </TableCell>
            </TableRow>
          ))}
          {positions.length === 0 && (
            <TableRow>
              <TableCell colSpan={7} align="center">
                <Typography color="textSecondary">No active positions</Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ActivePositions; 