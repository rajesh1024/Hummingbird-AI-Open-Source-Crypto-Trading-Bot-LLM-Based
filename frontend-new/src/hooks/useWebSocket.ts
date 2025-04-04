import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketHook<T> {
  data: T | null;
  isConnected: boolean;
  error: Error | null;
  loading: boolean;
  reconnect: () => void;
}

export function useWebSocket<T>(url: string): WebSocketHook<T> {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatIntervalRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 10;
  const HEARTBEAT_INTERVAL = 30000; // 30 seconds
  const CONNECTION_TIMEOUT = 15000; // 15 seconds
  const connectionTimeoutRef = useRef<NodeJS.Timeout>();
  const isComponentMounted = useRef(true);

  const cleanup = useCallback(() => {
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      } catch (err) {
        console.warn('Failed to send heartbeat:', err);
        cleanup();
        reconnect();
      }
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    heartbeatIntervalRef.current = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL);
  }, [sendHeartbeat]);

  const connect = useCallback(() => {
    if (!isComponentMounted.current) return;

    try {
      cleanup();

      if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        setError(new Error('Maximum reconnection attempts reached. Please refresh the page.'));
        setLoading(false);
        return;
      }

      console.log('Attempting to connect to WebSocket:', url);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      // Set connection timeout
      connectionTimeoutRef.current = setTimeout(() => {
        console.warn('Connection timeout - closing socket');
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close();
        }
      }, CONNECTION_TIMEOUT);

      ws.onopen = () => {
        console.log('WebSocket Connected');
        if (connectionTimeoutRef.current) {
          clearTimeout(connectionTimeoutRef.current);
        }
        setIsConnected(true);
        setLoading(false);
        setError(null);
        reconnectAttemptsRef.current = 0;
        startHeartbeat();
      };

      ws.onclose = (event) => {
        console.log('WebSocket Disconnected:', event.code, event.reason);
        cleanup();
        setIsConnected(false);
        setLoading(false);
        
        // Handle different close codes
        switch (event.code) {
          case 1000: // Normal closure
            break;
          case 1006: // Abnormal closure
            console.warn('Abnormal closure - server might be down');
            setError(new Error('Connection lost. Server might be unavailable.'));
            break;
          default:
            console.warn(`WebSocket closed with code ${event.code}`);
            setError(new Error('Connection closed. Attempting to reconnect...'));
        }

        // Don't reconnect if it was a normal closure or component unmounted
        if (event.code !== 1000 && isComponentMounted.current) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);
          reconnectTimeoutRef.current = setTimeout(connect, delay);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('Connection error. Please check if the server is running.'));
      };

      ws.onmessage = (event) => {
        try {
          // Handle heartbeat response
          if (event.data === 'pong') {
            return;
          }

          const parsedData = JSON.parse(event.data);
          console.log('Received WebSocket data:', parsedData);
          
          // Validate the data structure
          if (!parsedData || typeof parsedData !== 'object') {
            console.warn('Invalid data format received:', parsedData);
            return;
          }

          // Type guard to ensure the data matches our expected structure
          if (validateWebSocketData(parsedData)) {
            setData(parsedData as T);
            setError(null);
          } else {
            console.warn('Invalid data structure received:', parsedData);
          }
        } catch (err) {
          console.error('Error processing WebSocket data:', err);
          console.warn('Failed to process message:', event.data);
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError(new Error('Unable to establish connection. Please check if the server is running.'));
      setLoading(false);
      
      // Attempt to reconnect if component is still mounted
      if (isComponentMounted.current) {
        reconnectAttemptsRef.current += 1;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      }
    }
  }, [url, startHeartbeat, cleanup]);

  const reconnect = useCallback(() => {
    if (!isComponentMounted.current) return;
    cleanup();
    reconnectAttemptsRef.current = 0;
    setLoading(true);
    connect();
  }, [connect, cleanup]);

  useEffect(() => {
    isComponentMounted.current = true;
    connect();
    return () => {
      isComponentMounted.current = false;
      cleanup();
    };
  }, [url, connect, cleanup]);

  return { data, isConnected, error, loading, reconnect };
}

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