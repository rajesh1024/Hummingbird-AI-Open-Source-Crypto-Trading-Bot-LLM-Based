import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/components/ui/use-toast";
import MarketOverview from "@/components/dashboard/MarketOverview";
import SignalAnalysis from "@/components/dashboard/SignalAnalysis";
import ActivePositions from "@/components/dashboard/ActivePositions";
import useWebSocketData, { Signal } from "@/hooks/useWebSocketData";

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState("eth_usdt");
  const [wsUrl, setWsUrl] = useState("ws://localhost:8000/ws/dashboard?symbol=ETH/USDT");
  const { data, isConnected, error, signalHistory } = useWebSocketData(wsUrl);
  const { toast } = useToast();
  
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  
  const [filteredHistory, setFilteredHistory] = useState<{
    ethHistory: Signal[];
    btcHistory: Signal[];
  }>({
    ethHistory: [],
    btcHistory: [],
  });

  useEffect(() => {
    // Update WebSocket URL when tab changes
    const symbol = activeTab === "eth_usdt" ? "ETH/USDT" : "BTC/USDT";
    const newWsUrl = `ws://localhost:8000/ws/dashboard?symbol=${encodeURIComponent(symbol)}`;
    setWsUrl(newWsUrl);
  }, [activeTab]);

  useEffect(() => {
    if (data) {
      setLastUpdated(new Date().toLocaleTimeString());
    }
  }, [data]);

  useEffect(() => {
    if (isConnected) {
      toast({
        title: "Connected to data feed",
        description: "Real-time market data is now streaming",
      });
    } else if (error) {
      toast({
        title: "Connection Error",
        description: error,
        variant: "destructive",
      });
    }
  }, [isConnected, error, toast]);

  useEffect(() => {
    if (signalHistory.length > 0) {
      setFilteredHistory({
        ethHistory: signalHistory.filter(signal => signal.symbol === "ETH/USDT"),
        btcHistory: signalHistory.filter(signal => signal.symbol === "BTC/USDT"),
      });
    }
  }, [signalHistory]);

  if (!data) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Trading Dashboard</h1>
        <div className="p-12 text-center">
          <p className="text-lg text-muted-foreground">
            {error ? "Error connecting to data feed" : "Connecting to data feed..."}
          </p>
        </div>
      </div>
    );
  }

  const defaultMarketData = {
    symbol: "",
    current_price: 0,
    price_change_24h: 0,
    volume_24h: 0,
    technical_indicators: {
      RSI: 0,
      MACD: { MACD: 0, Signal: 0, Histogram: 0 },
      EMA: { EMA8: 0, EMA21: 0, EMA50: 0 },
      BB: { Upper: 0, Middle: 0, Lower: 0 },
      Volume: { "1h": 0, "4h": 0, "1d": 0 },
      ATR: 0
    },
    trading_mode: ""
  };

  const marketData = data?.data?.market_data || defaultMarketData;
  const positions = data?.data?.positions || [];
  const currentSignal = data?.data?.signal || {
    signal: "HOLD" as const,
    confidence: 0,
    symbol: "",
    timeframe: "",
    entry_price: 0,
    stop_loss: 0,
    take_profit: 0,
    reasoning: "No signal available",
    timestamp: new Date().toISOString(),
    position_management: {
      action: "HOLD",
      take_profit_adjustment: null,
      stop_loss_adjustment: null
    }
  };

  const ethData = marketData.symbol === "ETH/USDT" ? marketData : {
    ...defaultMarketData,
    symbol: "ETH/USDT"
  };
  
  const btcData = marketData.symbol === "BTC/USDT" ? marketData : {
    ...defaultMarketData,
    symbol: "BTC/USDT"
  };
  
  const ethPositions = positions.filter(p => p.symbol === "ETH/USDT");
  const btcPositions = positions.filter(p => p.symbol === "BTC/USDT");
  
  const emptySignal: Signal = {
    signal: "HOLD",
    confidence: 0,
    symbol: "",
    timeframe: "",
    entry_price: 0,
    stop_loss: 0,
    take_profit: 0,
    reasoning: "No signal available",
    timestamp: new Date().toISOString(),
    position_management: {
      action: "HOLD",
      take_profit_adjustment: null,
      stop_loss_adjustment: null
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Trading Dashboard</h1>
      
      <Tabs defaultValue="eth_usdt" value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="eth_usdt">ETH/USDT</TabsTrigger>
          <TabsTrigger value="btc_usdt">BTC/USDT</TabsTrigger>
        </TabsList>
        
        <TabsContent value="eth_usdt" className="space-y-6">
          <MarketOverview 
            symbol="ETHUSDT" 
            data={ethData}
            lastUpdated={lastUpdated || undefined}
          />
          
          <SignalAnalysis 
            symbol="ETH/USDT"
            currentSignal={currentSignal.symbol === "ETH/USDT" ? currentSignal : emptySignal}
            historySignals={filteredHistory.ethHistory}
            lastUpdated={lastUpdated || undefined}
            activePosition={ethPositions.find(p => p.status === "OPEN")}
          />
        </TabsContent>
        
        <TabsContent value="btc_usdt" className="space-y-6">
          <MarketOverview 
            symbol="BTCUSDT" 
            data={btcData}
            lastUpdated={lastUpdated || undefined}
          />
          
          <SignalAnalysis 
            symbol="BTC/USDT"
            currentSignal={currentSignal.symbol === "BTC/USDT" ? currentSignal : emptySignal}
            historySignals={filteredHistory.btcHistory}
            lastUpdated={lastUpdated || undefined}
            activePosition={btcPositions.find(p => p.status === "OPEN")}
          />
        </TabsContent>
      </Tabs>
      
      <ActivePositions 
        positions={activeTab === "eth_usdt" ? ethPositions : btcPositions} 
        lastUpdated={lastUpdated || undefined}
      />
    </div>
  );
};

export default Dashboard;
