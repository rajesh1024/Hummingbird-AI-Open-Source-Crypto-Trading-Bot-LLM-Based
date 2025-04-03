import asyncio
import websockets
import json
from datetime import datetime

async def test_dashboard():
    uri = "ws://localhost:8000/ws/dashboard"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to dashboard WebSocket")
            print("Press Ctrl+C to stop the client")
            
            while True:
                try:
                    data = await websocket.recv()
                    json_data = json.loads(data)
                    
                    # Print formatted data
                    print("\n=== Dashboard Update ===")
                    print(f"Timestamp: {json_data['timestamp']}")
                    
                    if json_data['market_data']:
                        print("\nMarket Data:")
                        print(f"Symbol: {json_data['market_data']['symbol']}")
                        print(f"Current Price: {json_data['market_data']['current_price']}")
                        print(f"24h Change: {json_data['market_data']['price_change_24h']}%")
                        print(f"RSI: {json_data['market_data']['rsi']}")
                    
                    if json_data['positions']:
                        print("\nActive Positions:")
                        for pos in json_data['positions']:
                            print(f"- {pos['symbol']}: {pos['position_type']} @ {pos['entry_price']}")
                            print(f"  PnL: {pos['pnl']}%")
                    
                    if json_data['signals']:
                        print("\nLatest Signal:")
                        signal = json_data['signals']
                        print(f"Type: {signal['signal']}")
                        print(f"Confidence: {signal['confidence']}")
                        print(f"Entry: {signal['entry_price']}")
                        print(f"Stop Loss: {signal['stop_loss']}")
                        print(f"Take Profit: {signal['take_profit']}")
                        
                        # Handle nested risk_reward_ratio
                        if 'position_management' in signal:
                            print(f"R:R: {signal['position_management'].get('risk_reward_ratio', 'N/A')}")
                        else:
                            print("R:R: N/A")
                            
                        print(f"Reasoning: {signal['reasoning']}")
                    
                    print("\nSettings:")
                    print(json.dumps(json_data['settings'], indent=2))
                    
                except Exception as e:
                    print(f"Error processing data: {str(e)}")
                    break
                    
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Connection error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_dashboard()) 