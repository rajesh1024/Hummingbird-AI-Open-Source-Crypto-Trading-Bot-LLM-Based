import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ArrowDown, ArrowUp, TrendingUp, TrendingDown, ChartBar, Clock } from "lucide-react";
import { useEffect, useRef } from "react";
import { MarketData } from "@/hooks/useWebSocketData";
import { useTheme } from "next-themes";

interface MarketOverviewProps {
  symbol: string;
  data: MarketData;
  lastUpdated?: string;
}

const MarketOverview = ({ symbol, data, lastUpdated }: MarketOverviewProps) => {
  const tradingViewRef = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const container = document.getElementById('tradingview_widget');
    if (!container) return;

    container.innerHTML = '';
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.type = 'text/javascript';
    script.async = true;

    script.innerHTML = JSON.stringify({
      "autosize": true,
      "symbol": `BINANCE:${symbol.replace('/', '')}`,
      "interval": "15",
      "timezone": "Etc/UTC",
      "theme": theme === 'dark' ? 'dark' : 'light',
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "hide_top_toolbar": true,
      "hide_legend": true,
      "save_image": false,
      "calendar": false,
      "hide_volume": true,
      "support_host": "https://www.tradingview.com"
    });

    container.appendChild(script);

    return () => {
      if (container) {
        container.innerHTML = '';
      }
    };
  }, [symbol, theme]);

  const isPriceUp = data.price_change_24h >= 0;
  const { technical_indicators } = data;

  return (
    <Card className="w-full shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex justify-between items-center">
          <span>Market Overview</span>
          {lastUpdated && (
            <div className="flex items-center gap-1 text-sm font-normal">
              <Clock size={14} /> 
              <span>Last update: {lastUpdated}</span>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm text-muted-foreground">Current Price</p>
                <h3 className="text-2xl font-bold">${data.current_price.toLocaleString()}</h3>
                <div className={`flex items-center text-sm ${isPriceUp ? 'text-green-600' : 'text-red-600'}`}>
                  {isPriceUp ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                  <span>${Math.abs(data.price_change_24h).toLocaleString()} ({Math.abs(data.price_change_24h / (data.current_price - data.price_change_24h) * 100).toFixed(2)}%)</span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-muted-foreground">24h Volume</p>
                <p className="font-semibold">{(data.volume_24h / 1000000).toFixed(2)}M</p>
              </div>
            </div>

            <Separator />
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">15min Volume RSI</p>
                <div className="flex items-center">
                  <p className={`font-semibold ${technical_indicators.RSI > 70 ? 'text-green-600' : 
                                               technical_indicators.RSI < 30 ? 'text-red-600' : 'text-gray-700'}`}>
                    {technical_indicators.RSI.toFixed(2)}
                  </p>
                  {technical_indicators.RSI > 70 ? <TrendingUp className="ml-1 text-green-600" size={16} /> : 
                   technical_indicators.RSI < 30 ? <TrendingDown className="ml-1 text-red-600" size={16} /> : null}
                </div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">MACD</p>
                <div className="flex items-center">
                  <p className={`font-semibold ${technical_indicators.MACD.MACD > technical_indicators.MACD.Signal ? 'text-green-600' : 'text-red-600'}`}>
                    {technical_indicators.MACD.MACD.toFixed(2)}
                  </p>
                  {technical_indicators.MACD.MACD > technical_indicators.MACD.Signal ? 
                    <TrendingUp className="ml-1 text-green-600" size={16} /> : 
                    <TrendingDown className="ml-1 text-red-600" size={16} />}
                </div>
                <p className="text-xs text-muted-foreground">Signal: {technical_indicators.MACD.Signal.toFixed(2)}</p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">1h Volume</p>
                <p className="font-semibold">{(technical_indicators.Volume["1h"] / 1000).toFixed(2)}K</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">4h Volume</p>
                <p className="font-semibold">{(technical_indicators.Volume["4h"] / 1000).toFixed(2)}K</p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 relative min-h-[250px]">
            <div id="tradingview_widget" className="tradingview-widget-container w-full h-full">
              <div className="tradingview-widget-container__widget w-full h-full"></div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MarketOverview;
