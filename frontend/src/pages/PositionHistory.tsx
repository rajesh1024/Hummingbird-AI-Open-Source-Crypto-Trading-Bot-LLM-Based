import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { cn } from "@/lib/utils";

interface Position {
  id: string;
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
  tradeType: 'all' | 'LONG' | 'SHORT';
}

interface Stats {
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  avgPnLPerTrade: number;
}

interface PaginatedResponse {
  data: Position[];
  total: number;
  page: number;
  totalPages: number;
  stats: Stats;
}

const ITEMS_PER_PAGE = 10;

const PositionHistory = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [totalPositions, setTotalPositions] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [stats, setStats] = useState<Stats>({
    totalTrades: 0,
    winRate: 0,
    totalPnL: 0,
    avgPnLPerTrade: 0
  });
  const [filters, setFilters] = useState<Filters>({
    startDate: new Date(new Date().setDate(new Date().getDate() - 30)), // Last 30 days
    endDate: new Date(),
    symbol: 'all',
    outcome: 'all',
    tradeType: 'all'
  });

  const symbols = ['all', 'BTC/USDT', 'ETH/USDT']; // Add your trading pairs here

  const fetchPositions = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      
      if (filters.startDate) {
        params.append('start_date', filters.startDate.toISOString());
      }
      if (filters.endDate) {
        params.append('end_date', filters.endDate.toISOString());
      }
      if (filters.symbol && filters.symbol !== 'all') {
        params.append('symbol', filters.symbol);
      }
      if (filters.tradeType !== 'all') {
        params.append('type', filters.tradeType);
      }
      
      // Add pagination parameters
      params.append('page', currentPage.toString());
      params.append('limit', ITEMS_PER_PAGE.toString());

      const url = `/api/positions/history?${params}`;
      console.log('Fetching positions from:', url);

      const response = await fetch(url);
      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Received response:', result);
      
      // Handle both array and paginated response formats
      let positionsArray: Position[];
      let totalCount: number;
      let totalPagesCount: number;
      let statsData: Stats;
      
      if (Array.isArray(result)) {
        // Direct array response - implement client-side pagination
        positionsArray = result;
        totalCount = result.length;
        totalPagesCount = Math.ceil(totalCount / ITEMS_PER_PAGE);
        
        // Calculate stats from all positions
        let totalPnL = 0;
        let winningTrades = 0;
        
        result.forEach(pos => {
          const pnl = parseFloat(pos.pnl?.toString() || '0');
          totalPnL += pnl;
          if (pnl >= 0) winningTrades++;
        });
        
        statsData = {
          totalTrades: totalCount,
          winRate: Number((totalCount > 0 ? (winningTrades / totalCount) * 100 : 0).toFixed(2)),
          totalPnL: Number(totalPnL.toFixed(2)),
          avgPnLPerTrade: Number((totalCount > 0 ? totalPnL / totalCount : 0).toFixed(2))
        };
        
        // Apply client-side pagination
        const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
        const endIndex = startIndex + ITEMS_PER_PAGE;
        positionsArray = result.slice(startIndex, endIndex);
      } else if (result.data && Array.isArray(result.data)) {
        // Paginated response with stats
        positionsArray = result.data;
        totalCount = result.total;
        totalPagesCount = result.totalPages;
        statsData = result.stats;
      } else {
        throw new Error('Invalid response format: expected array or paginated response');
      }
      
      // Filter by outcome if needed
      let filteredPositions = positionsArray;
      if (filters.outcome !== 'all') {
        filteredPositions = positionsArray.filter(pos => 
          filters.outcome === 'profit' ? parseFloat(pos.pnl?.toString() || '0') >= 0 : parseFloat(pos.pnl?.toString() || '0') < 0
        );
      }

      setPositions(filteredPositions);
      setTotalPositions(totalCount);
      setTotalPages(totalPagesCount);
      setStats(statsData);
    } catch (err) {
      console.error('Error in fetchPositions:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
      setPositions([]);
      setTotalPositions(0);
      setTotalPages(0);
      setStats({
        totalTrades: 0,
        winRate: 0,
        totalPnL: 0,
        avgPnLPerTrade: 0
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('Effect triggered with:', { currentPage, filters });
    fetchPositions();
  }, [currentPage, filters]);

  const getStatusColor = (pnl: number) => {
    return pnl >= 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  const getClosedReasonColor = (reason: string) => {
    switch (reason.toLowerCase()) {
      case 'take_profit': return 'bg-green-100 text-green-800';
      case 'stop_loss': return 'bg-red-100 text-red-800';
      case 'manual': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">Position History</h1>

      {/* Filters */}
      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start text-left font-normal">
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {filters.startDate ? format(filters.startDate, "PPP") : "Start Date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={filters.startDate || undefined}
                    onSelect={(date) => setFilters(prev => ({ ...prev, startDate: date }))}
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start text-left font-normal">
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {filters.endDate ? format(filters.endDate, "PPP") : "End Date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={filters.endDate || undefined}
                    onSelect={(date) => setFilters(prev => ({ ...prev, endDate: date }))}
                  />
                </PopoverContent>
              </Popover>
            </div>

            <div>
              <Select
                value={filters.symbol}
                onValueChange={(value) => setFilters(prev => ({ ...prev, symbol: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select symbol" />
                </SelectTrigger>
                <SelectContent>
                  {symbols.map((symbol) => (
                    <SelectItem key={symbol} value={symbol}>
                      {symbol === 'all' ? 'All Symbols' : symbol}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Select
                value={filters.tradeType}
                onValueChange={(value) => setFilters(prev => ({ ...prev, tradeType: value as 'all' | 'LONG' | 'SHORT' }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="LONG">Long</SelectItem>
                  <SelectItem value="SHORT">Short</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Select
                value={filters.outcome}
                onValueChange={(value) => setFilters(prev => ({ ...prev, outcome: value as 'all' | 'profit' | 'loss' }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select outcome" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Outcomes</SelectItem>
                  <SelectItem value="profit">Profit</SelectItem>
                  <SelectItem value="loss">Loss</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Button 
                className="w-full"
                onClick={() => {
                  setCurrentPage(1); // Reset to first page when applying filters
                  fetchPositions();
                }}
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Apply Filters'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Summary */}
      {!loading && positions && (
        <Card>
          <CardContent className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Total Trades</h3>
                <p className="text-2xl font-bold">{stats.totalTrades}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Win Rate</h3>
                <p className="text-2xl font-bold">{stats.winRate.toFixed(2)}%</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Total P&L</h3>
                <p className={`text-2xl font-bold ${stats.totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${stats.totalPnL.toFixed(2)}
                </p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Avg. P&L per Trade</h3>
                <p className={`text-2xl font-bold ${stats.avgPnLPerTrade >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${stats.avgPnLPerTrade.toFixed(2)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Positions Table/Cards */}
      <Card>
        <CardContent className="p-0">
          {/* Table View - Hidden on Mobile */}
          <div className="hidden md:block">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Entry Price</TableHead>
                  <TableHead>Exit Price</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead>Close Reason</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Closed At</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-4">
                      Loading...
                    </TableCell>
                  </TableRow>
                ) : !positions || positions.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-4">
                      No positions found
                    </TableCell>
                  </TableRow>
                ) : (
                  positions.map((position, index) => (
                    <TableRow key={`${position.symbol}-${position.closed_at}-${index}`}>
                      <TableCell>{position.symbol}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className={position.type === 'LONG' ? 'text-green-600' : 'text-red-600'}>
                          {position.type}
                        </Badge>
                      </TableCell>
                      <TableCell>${position.entry_price.toFixed(2)}</TableCell>
                      <TableCell>${position.exit_price.toFixed(2)}</TableCell>
                      <TableCell>
                        <span className={position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                          ${Math.abs(position.pnl).toFixed(2)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className={
                          position.closed_reason === "TP" 
                            ? "bg-green-100 text-green-800" 
                            : position.closed_reason === "SL" 
                              ? "bg-red-100 text-red-800" 
                              : "bg-gray-100 text-gray-800"
                        }>
                          {position.closed_reason === "TP" 
                            ? "Profit" 
                            : position.closed_reason === "SL" 
                              ? "Loss" 
                              : position.closed_reason}
                        </Badge>
                      </TableCell>
                      <TableCell>{position.duration}</TableCell>
                      <TableCell>{new Date(position.closed_at).toLocaleString()}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Card View - Visible on Mobile */}
          <div className="block md:hidden">
            {loading ? (
              <div className="text-center py-4">Loading...</div>
            ) : !positions || positions.length === 0 ? (
              <div className="text-center py-4">No positions found</div>
            ) : (
              <div className="space-y-4 p-4">
                {positions.map((position, index) => (
                  <Card key={`${position.symbol}-${position.closed_at}-${index}`} className="p-4">
                    <div className="space-y-3">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-semibold">{position.symbol}</h3>
                          <Badge variant="outline" className={cn(
                            "mt-1",
                            position.type === 'LONG' ? 'text-green-600' : 'text-red-600'
                          )}>
                            {position.type}
                          </Badge>
                        </div>
                        <Badge variant="outline" className={
                          position.closed_reason === "TP" 
                            ? "bg-green-100 text-green-800" 
                            : position.closed_reason === "SL" 
                              ? "bg-red-100 text-red-800" 
                              : "bg-gray-100 text-gray-800"
                        }>
                          {position.closed_reason === "TP" 
                            ? "Profit" 
                            : position.closed_reason === "SL" 
                              ? "Loss" 
                              : position.closed_reason}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-sm text-muted-foreground">Entry Price</p>
                          <p className="font-medium">${position.entry_price.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Exit Price</p>
                          <p className="font-medium">${position.exit_price.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">P&L</p>
                          <p className={cn(
                            "font-medium",
                            position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                          )}>
                            ${Math.abs(position.pnl).toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Duration</p>
                          <p className="font-medium">{position.duration}</p>
                        </div>
                      </div>
                      
                      <div>
                        <p className="text-sm text-muted-foreground">Closed At</p>
                        <p className="font-medium">{new Date(position.closed_at).toLocaleString()}</p>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Pagination */}
      {!loading && positions && positions.length > 0 && totalPages > 0 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {positions.length} of {totalPositions} trades
          </p>
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <Button
                  variant="outline"
                  size="sm"
                  className={`${currentPage === 1 ? 'pointer-events-none opacity-50' : ''}`}
                  onClick={() => currentPage > 1 && setCurrentPage(prev => prev - 1)}
                >
                  Previous
                </Button>
              </PaginationItem>
              
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter(page => {
                  return page === 1 || 
                         page === totalPages || 
                         Math.abs(currentPage - page) <= 1;
                })
                .map((page, index, array) => {
                  if (index > 0 && page - array[index - 1] > 1) {
                    return (
                      <PaginationItem key={`ellipsis-${page}`}>
                        <span className="px-3 py-2">...</span>
                      </PaginationItem>
                    );
                  }
                  return (
                    <PaginationItem key={page}>
                      <Button
                        variant={currentPage === page ? "default" : "outline"}
                        size="sm"
                        onClick={() => setCurrentPage(page)}
                      >
                        {page}
                      </Button>
                    </PaginationItem>
                  );
                })}

              <PaginationItem>
                <Button
                  variant="outline"
                  size="sm"
                  className={`${currentPage >= totalPages ? 'pointer-events-none opacity-50' : ''}`}
                  onClick={() => currentPage < totalPages && setCurrentPage(prev => prev + 1)}
                >
                  Next
                </Button>
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        </div>
      )}
    </div>
  );
};

export default PositionHistory;
