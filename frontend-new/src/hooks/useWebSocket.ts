import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketOptions {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onMessage?: (data: any) => void;
  onError?: (error: Event) => void;
  shouldReconnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export const useWebSocket = (url: string, options: WebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const maxReconnectAttempts = options.reconnectAttempts || 5;
  const reconnectInterval = options.reconnectInterval || 10000; // 10 seconds
  const optionsRef = useRef(options);
  const urlRef = useRef(url);

  // Update refs when options or url change
  useEffect(() => {
    optionsRef.current = options;
    urlRef.current = url;
  }, [options, url]);

  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }
    if (wsRef.current) {
      console.log(`Cleaning up WebSocket connection for ${url}`);
      wsRef.current.close(1000, 'Component unmounting');
      wsRef.current = null;
    }
    setIsConnected(false);
    setError(null);
  }, [url]);

  const connect = useCallback(() => {
    cleanup();

    try {
      console.log(`Initiating WebSocket connection to ${url}`);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`WebSocket connected to ${url}`);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
        optionsRef.current.onOpen?.();
      };

      ws.onclose = (event) => {
        console.log(`WebSocket closed for ${url}. Code: ${event.code}, Reason: ${event.reason}, Clean: ${event.wasClean}`);
        setIsConnected(false);
        optionsRef.current.onClose?.(event);

        // Don't reconnect if the connection was closed cleanly or we're unmounting
        if (event.code === 1000 || !optionsRef.current.shouldReconnect) {
          return;
        }

        // Handle reconnection
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          const delay = reconnectInterval;
          console.log(`Connection attempt ${reconnectAttemptsRef.current} of ${maxReconnectAttempts} for ${url} will start in ${delay / 1000} seconds`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (optionsRef.current.shouldReconnect) {
              connect();
            }
          }, delay);
        } else {
          setError(`Maximum reconnection attempts (${maxReconnectAttempts}) reached`);
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'heartbeat') {
            // Handle heartbeat silently
            ws.send(JSON.stringify({ type: 'pong' }));
            return;
          }
          console.log(`Received data for ${url}:`, data.type);
          optionsRef.current.onMessage?.(data);
        } catch (e) {
          console.error(`Error processing WebSocket message:`, e);
        }
      };

      ws.onerror = (event) => {
        console.error(`WebSocket error for ${url}:`, event);
        setError('WebSocket connection error');
        optionsRef.current.onError?.(event);
      };

      // Set up ping interval to keep connection alive
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000); // Send ping every 30 seconds

      return () => {
        clearInterval(pingInterval);
        cleanup();
      };
    } catch (error) {
      console.error(`Error creating WebSocket connection:`, error);
      setError('Failed to create WebSocket connection');
    }
  }, [url, cleanup, maxReconnectAttempts, reconnectInterval]);

  useEffect(() => {
    connect();
    return cleanup;
  }, [connect, cleanup]);

  // Reset connection when URL changes
  useEffect(() => {
    reconnectAttemptsRef.current = 0;
    connect();
  }, [url, connect]);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect]);

  return {
    isConnected,
    error,
    reconnect
  };
};

// Helper function to validate WebSocket data structure
function validateWebSocketData(data: any): boolean {
  if (!data || typeof data !== 'object') return false;

  try {
    // Validate message type
    if (typeof data.type !== 'string') {
      console.warn('Invalid message type:', data.type);
      return false;
    }

    // Validate data object
    if (!data.data || typeof data.data !== 'object') {
      console.warn('Invalid data object:', data.data);
      return false;
    }

    // Validate positions array
    if (data.data.positions) {
      if (!Array.isArray(data.data.positions)) {
        console.warn('Invalid positions array:', data.data.positions);
        return false;
      }
      // Validate each position
      for (const position of data.data.positions) {
        if (!validatePosition(position)) {
          console.warn('Invalid position:', position);
          return false;
        }
      }
    }

    // Validate market data
    if (data.data.market_data) {
      if (!validateMarketData(data.data.market_data)) {
        console.warn('Invalid market_data:', data.data.market_data);
        return false;
      }
    }

    // Validate signal
    if (data.data.signal) {
      if (!validateSignal(data.data.signal)) {
        console.warn('Invalid signal:', data.data.signal);
        return false;
      }
    }

    return true;
  } catch (err) {
    console.error('Error validating WebSocket data:', err);
    return false;
  }
}

function validatePosition(position: any): boolean {
  return (
    typeof position.id === 'number' &&
    typeof position.symbol === 'string' &&
    typeof position.position_type === 'string' &&
    typeof position.status === 'string' &&
    typeof position.entry_price === 'number' &&
    (position.current_price === null || typeof position.current_price === 'number') &&
    (position.stop_loss === null || typeof position.stop_loss === 'number') &&
    (position.take_profit === null || typeof position.take_profit === 'number') &&
    typeof position.size === 'number' &&
    typeof position.pnl === 'number' &&
    (position.created_at === null || typeof position.created_at === 'string') &&
    (position.closed_at === null || typeof position.closed_at === 'string')
  );
}

function validateMarketData(marketData: any): boolean {
  return (
    typeof marketData.current_price === 'number' &&
    typeof marketData.price_change_24h === 'number' &&
    typeof marketData.volume_24h === 'number' &&
    typeof marketData.rsi === 'number' &&
    typeof marketData.symbol === 'string'
  );
}

function validateSignal(signal: any): boolean {
  return (
    typeof signal.signal === 'string' &&
    typeof signal.confidence === 'number' &&
    typeof signal.symbol === 'string' &&
    typeof signal.timeframe === 'string' &&
    typeof signal.timestamp === 'string'
  );
} 