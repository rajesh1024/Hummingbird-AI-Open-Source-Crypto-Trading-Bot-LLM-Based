import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { TrendingDown, TrendingUp, Clock } from "lucide-react";
import { Position } from "@/hooks/useWebSocketData";
import { cn } from "@/lib/utils";

interface ActivePositionsProps {
  positions: Position[];
  lastUpdated?: string;
}

const ActivePositions = ({ positions, lastUpdated }: ActivePositionsProps) => {
  const calculatePnLColor = (pnl: number) => {
    if (pnl > 0) return 'text-green-600 dark:text-green-400';
    if (pnl < 0) return 'text-red-600 dark:text-red-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  // Calculate PnL percentage
  const calculatePnlPercentage = (position: Position) => {
    const pnlPercentage = (position.pnl / (position.entry_price * position.size)) * 100;
    return pnlPercentage;
  };

  const getPositionTypeStyles = (type: string) => {
    return type === 'LONG'
      ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/50'
      : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/50';
  };

  return (
    <Card className="w-full shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex justify-between items-center">
          <span>Active Positions</span>
          {lastUpdated && (
            <div className="flex items-center gap-1 text-sm font-normal text-muted-foreground">
              <Clock size={14} /> 
              <span>Last update: {lastUpdated}</span>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {positions.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No active positions at the moment
          </div>
        ) : (
          <>
            {/* Table View - Hidden on Mobile */}
            <div className="hidden md:block overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Entry Price</TableHead>
                    <TableHead>Stop Loss</TableHead>
                    <TableHead>Take Profit</TableHead>
                    <TableHead>Current Price</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>PnL</TableHead>
                    
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {positions.map((position) => {
                    const pnlPercentage = calculatePnlPercentage(position);
                    
                    return (
                      <TableRow key={position.id}>
                        <TableCell>{position.symbol}</TableCell>
                        <TableCell>
                          <Badge variant="outline" className={getPositionTypeStyles(position.position_type)}>
                            {position.position_type}
                          </Badge>
                        </TableCell>
                        <TableCell>${position.entry_price}</TableCell>
                        <TableCell className="text-red-600 dark:text-red-400">${position.stop_loss}</TableCell>
                        <TableCell className="text-green-600 dark:text-green-400">${position.take_profit}</TableCell>
                        <TableCell>${position.current_price}</TableCell>
                        <TableCell>{position.size}</TableCell>
                        <TableCell className="whitespace-nowrap">
                          <div className={`flex items-center ${calculatePnLColor(position.pnl)}`}>
                            {position.pnl > 0 ? <TrendingUp size={14} className="mr-1" /> : <TrendingDown size={14} className="mr-1" />}
                            ${Math.abs(position.pnl).toLocaleString()} ({Math.abs(pnlPercentage).toFixed(2)}%)
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <Badge variant={position.status === 'OPEN' ? 'outline' : 'secondary'}>
                            {position.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>

            {/* Card View - Visible on Mobile */}
            <div className="block md:hidden space-y-4">
              {positions.map((position) => {
                const pnlPercentage = calculatePnlPercentage(position);
                
                return (
                  <Card key={position.id} className="p-4">
                    <div className="space-y-3">
                      {/* Header: Symbol, Type, and Status */}
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-semibold">{position.symbol}</h3>
                          <Badge variant="outline" className={getPositionTypeStyles(position.position_type)}>
                            {position.position_type}
                          </Badge>
                        </div>
                        <Badge variant={position.status === 'OPEN' ? 'outline' : 'secondary'}>
                          {position.status}
                        </Badge>
                      </div>

                      {/* PnL Section */}
                      <div className={cn("flex items-center text-lg font-semibold", calculatePnLColor(position.pnl))}>
                        {position.pnl > 0 ? <TrendingUp size={18} className="mr-2" /> : <TrendingDown size={18} className="mr-2" />}
                        ${Math.abs(position.pnl).toLocaleString()} ({Math.abs(pnlPercentage).toFixed(2)}%)
                      </div>

                      {/* Price Information */}
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-sm text-muted-foreground">Entry Price</p>
                          <p className="font-medium">${position.entry_price}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Current Price</p>
                          <p className="font-medium">${position.current_price}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Size</p>
                          <p className="font-medium">{position.size}</p>
                        </div>
                      </div>

                      {/* Stop Loss and Take Profit */}
                      <div className="grid grid-cols-2 gap-2 pt-2 border-t dark:border-gray-800">
                        <div>
                          <p className="text-sm text-muted-foreground">Stop Loss</p>
                          <p className="font-medium text-red-600 dark:text-red-400">${position.stop_loss}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Take Profit</p>
                          <p className="font-medium text-green-600 dark:text-green-400">${position.take_profit}</p>
                        </div>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ActivePositions;
