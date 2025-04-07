import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ChevronDown, Clock } from "lucide-react";
import { useState } from "react";
import { Signal, Position } from "@/hooks/useWebSocketData";
import ActivePositions from "./ActivePositions";

interface SignalAnalysisProps {
  symbol: string;
  currentSignal: Signal;
  historySignals: Signal[];
  lastUpdated?: string;
  activePosition?: Position | null;
}

const SignalAnalysis = ({ 
  symbol, 
  currentSignal, 
  historySignals, 
  lastUpdated,
  activePosition 
}: SignalAnalysisProps) => {
  const [showAllHistory, setShowAllHistory] = useState(false);
  
  const visibleHistory = showAllHistory ? historySignals : historySignals.slice(0, 3);
  
  const getSignalColor = (signal: string) => {
    switch(signal) {
      case 'BUY': return 'bg-green-100 text-green-800 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-400 dark:hover:bg-green-900/50';
      case 'SELL': return 'bg-red-100 text-red-800 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50';
      case 'HOLD': return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-400 dark:hover:bg-yellow-900/50';
      case 'LONG': return 'bg-green-100 text-green-800 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-400 dark:hover:bg-green-900/50';
      case 'SHORT': return 'bg-red-100 text-red-800 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50';
      default: return 'bg-gray-100 text-gray-800 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700';
    }
  };
  
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600 dark:text-green-400';
    if (confidence >= 50) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const formatDateTime = (timestamp: string) => {
    try {
      // Remove microseconds from the timestamp if present
      const cleanTimestamp = timestamp.split('.')[0];
      const date = new Date(cleanTimestamp);
      if (isNaN(date.getTime())) {
        return timestamp; // Return original if parsing fails
      }

      // Convert to IST by adding 5 hours and 30 minutes
      const istDate = new Date(date.getTime() + (5 * 60 + 30) * 60 * 1000);
      
      // Format date and time
      const dateStr = istDate.toLocaleDateString('en-IN');
      const timeStr = istDate.toLocaleTimeString('en-IN');
      
      return `${dateStr}, ${timeStr}`;
    } catch (e) {
      console.error('Error formatting timestamp:', e);
      return timestamp;
    }
  };
  
  return (
    <Card className="w-full shadow-sm dark:bg-slate-800 dark:border-slate-700">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center justify-between">
          <span>Signal & Analysis</span>
          {lastUpdated && (
            <div className="flex items-center gap-1 text-sm font-normal">
              <Clock size={14} /> 
              <span>Last update: {lastUpdated}</span>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <h3 className="font-medium mb-2">Current Signal</h3>
              <div className="flex flex-wrap items-center gap-2 mb-3">
                {activePosition && (
                  <div className="flex items-center gap-2">
                    <Badge className={`${getSignalColor(activePosition.position_type)} text-xs sm:text-sm`} variant="outline">
                      A: {activePosition.position_type}
                    </Badge>
                    <span className="text-sm text-muted-foreground hidden sm:inline">|</span>
                  </div>
                )}
                <Badge className={`${getSignalColor(currentSignal.signal)} text-xs sm:text-sm`} variant="outline">
                  S: {currentSignal.signal}
                </Badge>
                <span className={`text-xs sm:text-sm font-semibold ${getConfidenceColor(currentSignal.confidence)}`}>
                  {currentSignal.confidence}% Confidence
                </span>
              </div>
              
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="p-2 bg-slate-100 dark:bg-slate-700 rounded">
                  <p className="text-xs text-muted-foreground">Entry</p>
                  <p className="font-medium">${currentSignal.entry_price}</p>
                </div>
                <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded">
                  <p className="text-xs text-muted-foreground">Target</p>
                  <p className="font-medium text-green-700 dark:text-green-400">${currentSignal.take_profit}</p>
                </div>
                <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded">
                  <p className="text-xs text-muted-foreground">Stop Loss</p>
                  <p className="font-medium text-red-700 dark:text-red-400">${currentSignal.stop_loss}</p>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-medium mb-2">Position Management</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 bg-slate-100 dark:bg-slate-700 rounded text-center">
                  <p className="text-xs text-muted-foreground">Action</p>
                  <Badge variant="outline" className="mt-1">
                    {currentSignal.position_management?.action || 'N/A'}
                  </Badge>
                </div>
                <div className="p-2 bg-slate-100 dark:bg-slate-700 rounded text-center">
                  <p className="text-xs text-muted-foreground">Trailing SL</p>
                  <p className="font-medium">
                    {currentSignal.position_management?.stop_loss_adjustment 
                      ? `${currentSignal.position_management.stop_loss_adjustment}` 
                      : 'None'}
                  </p>
                </div>
                <div className="p-2 bg-slate-100 dark:bg-slate-700 rounded text-center">
                  <p className="text-xs text-muted-foreground">Trailing TP</p>
                  <p className="font-medium">
                    {currentSignal.position_management?.take_profit_adjustment 
                      ? `${currentSignal.position_management.take_profit_adjustment}` 
                      : 'None'}
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">Analysis Reasoning</h3>
              <Button 
                variant="ghost" 
                size="sm" 
                className="text-xs h-8"
                onClick={() => setShowAllHistory(!showAllHistory)}
              >
                {showAllHistory ? 'Show Less' : 'Show All'}
                <ChevronDown className={`ml-1 h-4 w-4 transition-transform ${showAllHistory ? 'transform rotate-180' : ''}`} />
              </Button>
            </div>
            
            <div className="space-y-3 max-h-[300px] overflow-y-auto">
              {visibleHistory.length > 0 ? (
                visibleHistory.map((signal, index) => (
                  <div key={index} className="p-3 border rounded space-y-2 dark:border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge className={getSignalColor(signal.signal)} variant="outline">
                          {signal.signal}
                        </Badge>
                        <span className={`text-xs ${getConfidenceColor(signal.confidence)}`}>
                          {signal.confidence * 100}%
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Clock size={14} />
                        <span className="text-sm whitespace-nowrap">
                          {formatDateTime(signal.timestamp)}
                        </span>
                      </div>
                    </div>
                    <div className="text-sm">
                      <p className="font-medium mb-1">Analysis:</p>
                      <p className="text-muted-foreground whitespace-pre-wrap">{signal.reasoning}</p>
                    </div>
                    
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-muted-foreground">
                  No analysis history available
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default SignalAnalysis;

