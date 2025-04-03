import requests
import json
from datetime import datetime

def test_api_endpoints():
    base_url = "http://localhost:8000"
    symbol = "BTC/USDT"
    
    # Test settings endpoint
    print("\n=== Testing Settings Endpoint ===")
    settings_data = {
        "symbol": symbol,
        "trading_mode": "scalping"
    }
    try:
        response = requests.post(f"{base_url}/api/settings", json=settings_data)
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test market data endpoint
    print("\n=== Testing Market Data Endpoint ===")
    try:
        response = requests.get(f"{base_url}/api/market-data?symbol={symbol}")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test positions endpoint
    print("\n=== Testing Positions Endpoint ===")
    try:
        response = requests.get(f"{base_url}/api/positions")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test signals endpoint
    print("\n=== Testing Signals Endpoint ===")
    try:
        response = requests.get(f"{base_url}/api/signals?symbol={symbol}")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api_endpoints() 