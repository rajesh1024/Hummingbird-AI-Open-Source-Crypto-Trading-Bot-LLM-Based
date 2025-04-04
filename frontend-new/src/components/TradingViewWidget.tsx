import React, { useEffect, useRef, useState } from 'react';

const TradingViewWidget: React.FC = () => {
  const container = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const containerElement = container.current;
    
    if (!containerElement) return;

    const loadScript = () => {
      try {
        const script = document.createElement("script");
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
        script.type = "text/javascript";
        script.async = true;
        script.innerHTML = JSON.stringify({
          autosize: true,
          symbol: "BINANCE:BTCUSD",
          interval: "5",
          timezone: "Asia/Kolkata",
          theme: "dark",
          style: "1",
          locale: "en",
          hide_top_toolbar: true,
          hide_legend: true,
          allow_symbol_change: true,
          save_image: false,
          studies: [
            "STD;VWAP"
          ],
          hide_volume: true,
          support_host: "https://www.tradingview.com"
        });

        // Clean up existing widget if any
        const existingWidget = containerElement.querySelector('.tradingview-widget-container__widget');
        if (existingWidget) {
          existingWidget.innerHTML = '';
        }

        // Remove any existing script
        const existingScript = containerElement.querySelector('script');
        if (existingScript) {
          existingScript.remove();
        }

        script.onerror = () => {
          setError('Failed to load TradingView widget');
        };

        containerElement.appendChild(script);
      } catch (err) {
        setError('Error initializing TradingView widget');
        console.error('TradingView widget error:', err);
      }
    };

    // Add a small delay to ensure the DOM is ready
    const timer = setTimeout(loadScript, 100);

    return () => {
      clearTimeout(timer);
      if (containerElement) {
        const script = containerElement.querySelector('script');
        if (script) {
          script.remove();
        }
      }
    };
  }, []);

  if (error) {
    return (
      <div style={{ 
        height: "100%", 
        width: "100%", 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "center",
        backgroundColor: "rgba(0, 0, 0, 0.1)",
        borderRadius: "4px"
      }}>
        <div style={{ color: "red", textAlign: "center" }}>
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="tradingview-widget-container" ref={container} style={{ height: "100%", width: "100%" }}>
      <div className="tradingview-widget-container__widget" style={{ height: "100%", width: "100%" }}></div>
      <div className="tradingview-widget-copyright" style={{ display: 'none' }}>
        <a href="https://www.tradingview.com/?utm_source=localhost&amp;utm_medium=widget_new&amp;utm_campaign=advanced-chart" rel="noopener noreferrer nofollow" target="_blank">
          <span className="blue-text">Track all markets on TradingView</span>
        </a>
      </div>
    </div>
  );
};

export default TradingViewWidget; 