import { useState, useEffect } from 'react';

interface Position {
  symbol: string;
  type: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  closed_reason: string;
  duration: string;
}

interface PositionHistoryFilters {
  startDate: Date | null;
  endDate: Date | null;
  symbol?: string;
  type?: string;
}

interface UsePositionHistoryResult {
  positions: Position[];
  loading: boolean;
  error: Error | null;
  refetch: (filters: PositionHistoryFilters) => Promise<void>;
}

export function usePositionHistory(): UsePositionHistoryResult {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const apiUrl = process.env.NODE_ENV === 'production' 
    ? process.env.REACT_APP_API_URL 
    : `http://${window.location.hostname}:8000`;

  const fetchPositions = async (filters: PositionHistoryFilters) => {
    try {
      setLoading(true);
      setError(null);

      // Build query parameters
      const params = new URLSearchParams();
      if (filters.startDate) {
        params.append('start_date', filters.startDate.toISOString());
      }
      if (filters.endDate) {
        params.append('end_date', filters.endDate.toISOString());
      }
      if (filters.symbol) {
        params.append('symbol', filters.symbol);
      }
      if (filters.type) {
        params.append('type', filters.type);
      }

      const response = await fetch(`${apiUrl}/api/positions/history?${params}`);
      if (!response.ok) {
        throw new Error('Failed to fetch position history');
      }

      const data = await response.json();
      setPositions(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('An error occurred'));
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchPositions({
      startDate: null,
      endDate: null
    });
  }, []);

  return {
    positions,
    loading,
    error,
    refetch: fetchPositions
  };
} 