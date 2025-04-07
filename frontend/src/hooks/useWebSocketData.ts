import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from 'sonner';

// Types for the WebSocket data
export interface TechnicalIndicators {
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
    "1h": number;
    "4h": number;
    "1d": number;
  };
  ATR: number;
}

export interface MarketData {
  symbol: string;
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  technical_indicators: TechnicalIndicators;
  trading_mode: string;
}

export interface Position {
  id: number;
  symbol: string;
  position_type: "LONG" | "SHORT";
  status: "OPEN" | "CLOSED";
  entry_price: number;
  current_price: number;
  stop_loss: number;
  take_profit: number;
  size: number;
  pnl: number;
  created_at: string;
  closed_at: string | null;
  duration: number;
}

export interface PositionManagement {
  action: string;
  take_profit_adjustment: number | null;
  stop_loss_adjustment: number | null;
}

export interface Signal {
  signal: "BUY" | "SELL" | "HOLD";
  confidence: number;
  symbol: string;
  timeframe: string;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  reasoning: string;
  timestamp: string;
  position_management?: PositionManagement;
}

export interface WebSocketData {
  type: "initial" | "update";
  symbol: string;
  data: {
    market_data: MarketData;
    positions: Position[];
    signal: Signal;
  };
}

interface UseWebSocketDataReturn {
  data: WebSocketData | null;
  isConnected: boolean;
  error: string | null;
  signalHistory: Signal[];
}

// Maximum number of signal history items to keep
const MAX_SIGNAL_HISTORY = 10;

const useWebSocketData = (url: string): UseWebSocketDataReturn => {
  const [data, setData] = useState<WebSocketData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [signalHistory, setSignalHistory] = useState<Signal[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const urlRef = useRef(url);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const currentSymbolRef = useRef<string | null>(null);
  const signalHistoryRef = useRef<Record<string, Signal[]>>({});

  // Extract symbol from URL
  const getSymbolFromUrl = (wsUrl: string): string | null => {
    try {
      const url = new URL(wsUrl);
      return url.searchParams.get('symbol');
    } catch (e) {
      console.error('Failed to parse WebSocket URL:', e);
      return null;
    }
  };

  // Update signal history for a specific symbol
  const updateSignalHistory = useCallback((symbol: string, newSignal: Signal) => {
    console.log(`Updating signal history for ${symbol}:`, newSignal);
    
    // Ensure the signal has a timestamp
    if (!newSignal.timestamp) {
      newSignal.timestamp = new Date().toISOString();
    }
    
    setSignalHistory(prevHistory => {
      // Get current history for the symbol from ref
      const currentHistory = signalHistoryRef.current[symbol] || [];
      
      // Check if signal already exists by comparing reasoning only
      // This allows multiple signals with the same reasoning but different timestamps
      const exists = currentHistory.some(
        item => item.reasoning === newSignal.reasoning &&
                item.symbol === symbol &&
                Math.abs(new Date(item.timestamp).getTime() - new Date(newSignal.timestamp).getTime()) < 1000 // Within 1 second
      );
      
      if (!exists) {
        console.log(`Adding new signal to history for ${symbol}:`, newSignal);
        if (newSignal.signal === 'BUY') {
          toast.success(`BUY signal for ${symbol}`);
        } else if (newSignal.signal === 'SELL') {
          toast.error(`SELL signal for ${symbol}`);
        } else if (newSignal.signal === 'HOLD') {
         
        }
        // Add new signal to the beginning and limit to MAX_SIGNAL_HISTORY
        const updatedHistory = [newSignal, ...currentHistory].slice(0, MAX_SIGNAL_HISTORY);
        
        // Sort history by timestamp in descending order (newest first)
        updatedHistory.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        
        // Update the ref with the new history
        signalHistoryRef.current = {
          ...signalHistoryRef.current,
          [symbol]: updatedHistory
        };
        
        // Return the updated history if this is the current symbol
        if (symbol === currentSymbolRef.current) {
          console.log(`Updating displayed history for ${symbol}`, updatedHistory);
          return updatedHistory;
        }
      } else {
        console.log(`Signal already exists in history for ${symbol}`, newSignal);
      }
      
      // Return current history for the active symbol
      return signalHistoryRef.current[currentSymbolRef.current || ''] || [];
    });
  }, []);

  // Check WebSocket connection status
  const checkConnection = useCallback(() => {
    if (!ws.current) {
      setIsConnected(false);
      return false;
    }
    const isWsConnected = ws.current.readyState === WebSocket.OPEN;
    setIsConnected(isWsConnected);
    return isWsConnected;
  }, []);

  // Cleanup function to reset state and close connections
  const cleanup = useCallback(() => {
    console.log('Cleaning up WebSocket connection');
    
    // Clear timeouts and intervals
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    
    // Close WebSocket if open
    if (ws.current) {
      if (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING) {
        ws.current.close();
      }
      ws.current = null;
    }
    
    // Only reset WebSocket related state, preserve signal history
    setData(null);
    setIsConnected(false);
    setError(null);
    reconnectAttempts.current = 0;
  }, []);

  useEffect(() => {
    const newSymbol = getSymbolFromUrl(url);
    console.log(`Setting up WebSocket for symbol: ${newSymbol}`);

    // If symbol has changed, update connection and signal history
    if (currentSymbolRef.current !== newSymbol) {
      console.log(`Symbol changed from ${currentSymbolRef.current} to ${newSymbol}`);
      
      // Clean up old connection
      cleanup();
      
      // Update current symbol
      currentSymbolRef.current = newSymbol;
      urlRef.current = url;
      
      // Load signal history for the new symbol
      if (newSymbol) {
        const symbolHistory = signalHistoryRef.current[newSymbol] || [];
        console.log(`Loading signal history for ${newSymbol}:`, symbolHistory);
        setSignalHistory(symbolHistory);
      }
    }

    // Initialize WebSocket connection
    const connectWebSocket = () => {
      try {
        // Ensure we have a valid symbol
        if (!newSymbol) {
          console.error('WebSocket URL must include a symbol parameter');
          setError('WebSocket URL must include a symbol parameter');
          setIsConnected(false);
          return;
        }

        // Create new WebSocket connection
        console.log(`Creating new WebSocket connection for symbol: ${newSymbol}`);
        ws.current = new WebSocket(url);

        // Set up connection status check interval
        const connectionCheckInterval = setInterval(checkConnection, 1000);

        ws.current.onopen = () => {
          console.log(`WebSocket connected for symbol: ${newSymbol}`);
          setIsConnected(true);
          setError(null);
          reconnectAttempts.current = 0;

          // Start ping interval
          if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
          }
          pingIntervalRef.current = setInterval(() => {
            if (checkConnection()) {
              ws.current?.send(JSON.stringify({ type: 'ping', symbol: newSymbol }));
            }
          }, 10000);
        };

        ws.current.onmessage = (event) => {
          try {
            // Check connection status on each message
            checkConnection();
            
            const parsedData = JSON.parse(event.data);
            console.log('Received WebSocket data:', parsedData);
            
            // Skip if message doesn't have a symbol
            if (!parsedData.symbol) {
              console.debug('Skipping message without symbol:', parsedData);
              return;
            }

            // Skip if symbol doesn't match current symbol
            if (parsedData.symbol !== newSymbol) {
              console.debug(`Skipping message for different symbol: ${parsedData.symbol}, current: ${newSymbol}`);
              return;
            }
            
            if (parsedData.type === 'error') {
              console.error('WebSocket error:', parsedData.message);
              setError(parsedData.message);
              return;
            }
            
            if (parsedData.type === 'pong') {
              console.debug(`Received pong response for ${newSymbol}`);
              return;
            }
            
            if (parsedData.type === 'initial' || parsedData.type === 'update') {
              if (!parsedData.data) {
                console.error('Skipping message missing data field:', parsedData);
                return;
              }
              
              const { market_data, positions, signal } = parsedData.data;
              
              // Skip if market data is missing or doesn't match current symbol
              if (!market_data || market_data.symbol !== newSymbol) {
                console.debug('Skipping update with invalid market data or mismatched symbol');
                return;
              }

              // Update signal history if we have a valid signal
              if (signal && signal.symbol === newSymbol) {
                console.log('Received new signal:', signal);
                updateSignalHistory(newSymbol, signal);
              }

              setData((prevData) => {
                // Only update if previous data is null or matches current symbol
                if (prevData && prevData.symbol !== newSymbol) {
                  console.debug('Skipping state update for different symbol');
                  return prevData;
                }

                const updatedData: WebSocketData = {
                  type: parsedData.type,
                  symbol: newSymbol,
                  data: {
                    market_data: market_data,
                    positions: positions || [],
                    signal: signal || null,
                  }
                };

                console.debug('Updating state for symbol:', {
                  symbol: newSymbol,
                  currentPrice: market_data.current_price,
                  prevPrice: prevData?.data.market_data?.current_price,
                  type: parsedData.type
                });

                return updatedData;
              });
            }
          } catch (err) {
            console.error('Error parsing WebSocket data:', err);
            setError('Failed to parse WebSocket data');
          }
        };

        ws.current.onclose = (event) => {
          console.log(`WebSocket disconnected for symbol ${newSymbol}:`, event);
          setIsConnected(false);
          clearInterval(connectionCheckInterval);
          
          // Only attempt to reconnect if this is still the current symbol
          if (!event.wasClean && currentSymbolRef.current === newSymbol) {
            const retryDelay = Math.min(1000 * (2 ** reconnectAttempts.current), 30000);
            console.log(`Reconnecting ${newSymbol} in ${retryDelay}ms...`);
            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttempts.current += 1;
              connectWebSocket();
            }, retryDelay);
          }
        };

        ws.current.onerror = (error) => {
          console.error(`WebSocket error for ${newSymbol}:`, error);
          setError('WebSocket connection error');
          setIsConnected(false);
        };

      } catch (err) {
        console.error(`Failed to establish WebSocket connection for ${newSymbol}:`, err);
        setError('Failed to establish WebSocket connection');
        setIsConnected(false);
      }
    };

    // Only connect if we have a valid symbol
    if (newSymbol) {
      connectWebSocket();
    }

    // Cleanup when unmounting or URL changes
    return () => {
      cleanup();
    };
  }, [url, cleanup, checkConnection, updateSignalHistory]);

  return { data, isConnected, error, signalHistory };
};

export default useWebSocketData;
