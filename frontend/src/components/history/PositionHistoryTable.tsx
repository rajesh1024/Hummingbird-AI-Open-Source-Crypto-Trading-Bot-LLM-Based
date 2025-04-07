
import { useState, useMemo } from 'react';
import { 
  Card, CardContent, CardHeader, CardTitle, CardDescription 
} from "@/components/ui/card";
import { 
  Table, TableHeader, TableBody, TableRow, TableHead, TableCell 
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue 
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { TrendingDown, TrendingUp, Search, Calendar } from "lucide-react";

interface Position {
  id: string;
  pair: string;
  type: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercentage: number;
  openTime: string;
  closeTime: string;
  status: 'WIN' | 'LOSS';
}

interface PositionHistoryTableProps {
  positions: Position[];
}

const PositionHistoryTable = ({ positions }: PositionHistoryTableProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [pairFilter, setPairFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const filteredPositions = useMemo(() => {
    return positions.filter(position => {
      // Search filter
      const matchesSearch = 
        position.pair.toLowerCase().includes(searchTerm.toLowerCase()) || 
        position.id.toLowerCase().includes(searchTerm.toLowerCase());
      
      // Pair filter
      const matchesPair = pairFilter === 'all' || position.pair === pairFilter;
      
      // Type filter
      const matchesType = typeFilter === 'all' || position.type === typeFilter;
      
      // Status filter
      const matchesStatus = statusFilter === 'all' || position.status === statusFilter;
      
      return matchesSearch && matchesPair && matchesType && matchesStatus;
    });
  }, [positions, searchTerm, pairFilter, typeFilter, statusFilter]);

  const uniquePairs = useMemo(() => {
    return Array.from(new Set(positions.map(position => position.pair)));
  }, [positions]);
  
  const calculatePnLColor = (pnl: number) => {
    if (pnl > 0) return 'text-green-600';
    if (pnl < 0) return 'text-red-600';
    return 'text-gray-600';
  };
  
  const resetFilters = () => {
    setSearchTerm('');
    setPairFilter('all');
    setTypeFilter('all');
    setStatusFilter('all');
  };
  
  return (
    <Card className="w-full shadow-sm">
      <CardHeader>
        <CardTitle className="text-xl mb-2">Position History</CardTitle>
        <CardDescription>
          View and filter your past trading positions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-6 space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-grow">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by pair or ID..."
                className="pl-8"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 md:w-2/3">
              <Select value={pairFilter} onValueChange={setPairFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by pair" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Pairs</SelectItem>
                  {uniquePairs.map((pair) => (
                    <SelectItem key={pair} value={pair}>{pair}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="LONG">Long</SelectItem>
                  <SelectItem value="SHORT">Short</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="WIN">Win</SelectItem>
                  <SelectItem value="LOSS">Loss</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="flex justify-end">
            <Button variant="outline" size="sm" onClick={resetFilters}>
              Reset Filters
            </Button>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          {filteredPositions.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              No positions match your filters
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Pair</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Entry Price</TableHead>
                  <TableHead>Exit Price</TableHead>
                  <TableHead>Quantity</TableHead>
                  <TableHead>PnL</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Open Time</TableHead>
                  <TableHead>Close Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredPositions.map((position) => (
                  <TableRow key={position.id}>
                    <TableCell>{position.pair}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className={
                        position.type === 'LONG' 
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }>
                        {position.type}
                      </Badge>
                    </TableCell>
                    <TableCell>${position.entryPrice}</TableCell>
                    <TableCell>${position.exitPrice}</TableCell>
                    <TableCell>{position.quantity}</TableCell>
                    <TableCell className="whitespace-nowrap">
                      <div className={`flex items-center ${calculatePnLColor(position.pnl)}`}>
                        {position.pnl > 0 ? <TrendingUp size={14} className="mr-1" /> : <TrendingDown size={14} className="mr-1" />}
                        ${Math.abs(position.pnl).toLocaleString()} ({position.pnlPercentage.toFixed(2)}%)
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={position.status === 'WIN' ? 'default' : 'destructive'}>
                        {position.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{new Date(position.openTime).toLocaleString()}</TableCell>
                    <TableCell>{new Date(position.closeTime).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default PositionHistoryTable;
