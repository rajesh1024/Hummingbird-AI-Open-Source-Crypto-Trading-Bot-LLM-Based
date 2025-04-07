import React, { useState, useEffect } from 'react';
import { Tabs, Tab, Box, Chip, Typography, Alert } from '@mui/material';

interface SymbolStatus {
  symbol: string;
  state: string;
  subscribers: number;
  last_update: string;
}

interface SymbolData {
  [key: string]: SymbolStatus;
}

interface SymbolTabsProps {
  onSymbolChange: (symbol: string) => void;
  onSymbolsAvailable: (hasSymbols: boolean) => void;
  onLoadingChange: (isLoading: boolean) => void;
}

const SymbolTabs: React.FC<SymbolTabsProps> = ({ onSymbolChange, onSymbolsAvailable, onLoadingChange }) => {
  const [currentSymbol, setCurrentSymbol] = useState<string>('');
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [hasRunningSymbols, setHasRunningSymbols] = useState<boolean>(false);
  const [statusData, setStatusData] = useState<SymbolData>({});
  const [error, setError] = useState<string | null>(null);
  
  // Initial fetch of symbol status - only once on component mount
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        onLoadingChange(true);
        const apiUrl = `http://${window.location.hostname}:8000/api/symbols/status`;
        
        const response = await fetch(apiUrl);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json() as SymbolData;
        
        setStatusData(data);
        
        // Get all symbols that are running
        const runningSymbols = Object.entries(data)
          .filter(([_, status]) => status.state === 'running')
          .map(([symbol]) => symbol);
        
        setAvailableSymbols(runningSymbols);
        const hasSymbols = runningSymbols.length > 0;
        setHasRunningSymbols(hasSymbols);
        
        // Notify parent about symbol availability
        onSymbolsAvailable(hasSymbols);
        
        // Set current symbol if we have running symbols
        if (runningSymbols.length > 0) {
          const newSymbol = runningSymbols[0];
          setCurrentSymbol(newSymbol);
          onSymbolChange(newSymbol);
        }
      } catch (error) {
        console.error('Error fetching symbol status:', error);
        setError('Failed to fetch symbol status');
        setHasRunningSymbols(false);
        onSymbolsAvailable(false);
      } finally {
        onLoadingChange(false);
      }
    };

    fetchStatus();
  }, [onSymbolChange, onSymbolsAvailable, onLoadingChange]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
    setCurrentSymbol(newValue);
    onSymbolChange(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'stopped':
        return 'error';
      case 'paused':
        return 'warning';
      default:
        return 'default';
    }
  };

  if (error) {
    return (
      <Box sx={{ mb: 2 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!hasRunningSymbols) {
    return (
      <Box sx={{ mb: 2 }}>
        <Alert severity="info">No symbols running</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
      <Tabs 
        value={currentSymbol} 
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        aria-label="symbol tabs"
      >
        {availableSymbols.map((symbol) => (
          <Tab
            key={symbol}
            value={symbol}
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2">{symbol}</Typography>
                {statusData?.[symbol] && (
                  <Chip
                    label={statusData[symbol].state}
                    color={getStatusColor(statusData[symbol].state)}
                    size="small"
                  />
                )}
              </Box>
            }
          />
        ))}
      </Tabs>
    </Box>
  );
};

export default SymbolTabs; 