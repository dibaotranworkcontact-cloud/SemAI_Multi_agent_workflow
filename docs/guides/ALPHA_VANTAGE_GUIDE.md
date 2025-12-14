# Alpha Vantage API Integration Guide

## Overview

The Alpha Vantage API integration module provides direct access to market data for derivative pricing. Your API key (`5E0NVW3VI9N9E6PC`) supports:

### Free Tier Endpoints ✓
- **Symbol Search**: Find stock tickers and information
- **Daily Stock Data**: Historical daily OHLCV data
- **Weekly Stock Data**: Historical weekly aggregates
- **Monthly Stock Data**: Historical monthly aggregates
- **Technical Indicators**: SMA, EMA, MACD, RSI, and 70+ more

### Premium Endpoints ⚠️
- Intraday data (1-min, 5-min, 15-min, 30-min, 60-min)
- Adjusted close prices
- Forex rates
- Cryptocurrency data

## Quick Start

### 1. Basic Usage

```python
from semai.alpha_vantage_handler import AlphaVantageHandler

# Initialize client with your API key
api_key = "5E0NVW3VI9N9E6PC"
client = AlphaVantageHandler(api_key)

# Get daily data for Apple
aapl_data = client.get_daily_data('AAPL')
print(aapl_data.head())
print(f"Downloaded {len(aapl_data)} rows")

# Export to CSV
client.export_to_csv(aapl_data, 'aapl_daily.csv')
```

### 2. Search for Symbols

```python
# Search for Tesla
tesla = client.get_symbol_search('Tesla')
print(tesla[['1_symbol', '2_name', '4_region']])

# Columns returned:
# - 1_symbol: Ticker symbol
# - 2_name: Company/security name
# - 3_type: Type (Equity, ETF, etc.)
# - 4_region: Region/exchange
# - 5_marketopen: Market open time
# - 6_marketclose: Market close time
# - 7_timezone: Time zone
# - 8_currency: Trading currency
```

### 3. Get Historical Data

```python
# Daily data (100 most recent trading days)
daily = client.get_daily_data('MSFT')

# Weekly data
weekly = client.get_weekly_data('SPY')

# Monthly data
monthly = client.get_monthly_data('QQQ')

# All return pandas DataFrame with columns:
# 1._open, 2._high, 3._low, 4._close, 5._volume
```

### 4. Technical Indicators

```python
# Simple Moving Average (20-day)
sma = client.get_technical_indicators(
    symbol='AAPL',
    function='SMA',
    interval='daily',
    time_period=20,
    series_type='close'
)

# Other available indicators:
# EMA, MACD, RSI, BBANDS, ATR, ADX, CCI, ROC, TRIX, WILLR
```

### 5. Create Derivative Dataset

```python
# Generate synthetic option dataset from market data
dataset_info = client.create_derivative_dataset('AAPL')
print(dataset_info)
# Returns:
# {
#     'symbol': 'AAPL',
#     'rows': 45000,
#     'filepath': '/path/to/AAPL_derivative_dataset.csv',
#     'features': ['S', 'K', 'T', 'q', 'r', 'sigma', 'C'],
#     'description': 'Synthetic derivative dataset from AAPL market data',
#     'date_range': '2025-07-21 to 2025-12-09'
# }
```

## Data Format

### Daily/Weekly/Monthly Data
Each row contains OHLCV data with datetime index:

```
                   1._open  2._high   3._low  4._close  5._volume
2025-12-09  212.10   215.78  211.6300    212.48   51377434
2025-12-08  213.14   214.95  212.2301    214.40   46404072
2025-12-07  215.00   215.15  212.4100    214.15   46989301
```

### Symbol Search Results
```
   1_symbol                 2_name  3_type  4_region  9_matchscore
0       AAPL              Apple Inc  Equity    United States  1.0000
1      AAPL34.SAO         Apple Inc  Equity         Brazil  0.8571
2     APC.DEX              Apple Inc  Equity          Germany  0.5833
```

## Rate Limiting

The free tier API has rate limits:
- **5 calls per minute** (enforced by the handler)
- Automatic throttling between requests
- Cached responses to reduce API calls

```python
# First call to API
aapl_1 = client.get_daily_data('AAPL')  # ~12 seconds

# Second call uses cache (instant)
aapl_2 = client.get_daily_data('AAPL')  # ~0.1 seconds
```

Cache directory: `src/semai/.cache/`

## Integration with SEMAI

### Using Alpha Vantage data with derivative pricing models

```python
from semai.alpha_vantage_handler import AlphaVantageHandler
from semai.builtin_models import ModelRegistry
import pandas as pd
import numpy as np

# 1. Download market data
client = AlphaVantageHandler("5E0NVW3VI9N9E6PC")
stock_data = client.get_daily_data('SPY')

# 2. Create derivative pricing dataset
dataset_info = client.create_derivative_dataset('SPY')
df = pd.read_csv(dataset_info['filepath'])

# 3. Train models
registry = ModelRegistry()
model = registry.get_model('ann_standard')

X_train = df[['S', 'K', 'T', 'r', 'sigma']].values
y_train = df['C'].values

model.train(X_train, y_train, epochs=50)

# 4. Price new options
X_test = np.array([[100, 100, 0.25, 0.02, 0.20]])
price, std_dev = model.predict(X_test)
print(f"Option price: ${price:.2f} +/- ${std_dev:.2f}")
```

## Performance Characteristics

### Download Times (Free Tier)
| Data Type | Symbols | Time |
|-----------|---------|------|
| Symbol search | 10 matches | 11-12 seconds |
| Daily data | 100 rows | 1 second |
| Weekly data | 50 rows | 1 second |
| Monthly data | 20 rows | 1 second |
| Intraday data | Premium only | N/A |

### Cache Behavior
- Responses cached in JSON format
- Subsequent requests return cached data (~0.1 seconds)
- Cache cleared manually: delete `.cache/` directory

## Common Symbols

### Major Indices & ETFs
```
SPY  - S&P 500 ETF
QQQ  - Nasdaq-100 ETF
DIA  - Dow Jones Industrial ETF
IWM  - Russell 2000 ETF
VTI  - Total US Stock Market
```

### Large Cap Tech
```
AAPL - Apple
MSFT - Microsoft
GOOGL - Alphabet/Google
NVDA - NVIDIA
TSLA - Tesla
```

### Options Underlyings
```
SPY  - Good for index options
QQQ  - Good for tech options
IWM  - Good for small cap options
EWZ  - Good for EM options
TLT  - Good for bond options
```

## Troubleshooting

### "Premium endpoint" Error
**Issue**: Intraday data requests fail with "premium endpoint" message
**Solution**: Use daily/weekly/monthly data instead. Intraday requires paid subscription.

### Rate Limit Exceeded
**Issue**: API returns error after 5 rapid calls
**Solution**: Built-in throttling handles this automatically. Cache responses are instant.

### API Key Issues
**Issue**: "Error Message" in response
**Solution**: Verify API key in call:
```python
client = AlphaVantageHandler("YOUR_API_KEY_HERE")
```

### No Data Returned
**Issue**: `get_daily_data()` returns None
**Solution**: 
1. Check symbol is correct (use `get_symbol_search()`)
2. Verify internet connection
3. Check API rate limits (5 per minute)

## API Key Management

Your API key: `5E0NVW3VI9N9E6PC`

**Security Note**: In production, store in environment variable:
```python
import os
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
client = AlphaVantageHandler(api_key)
```

## Upgrade Options

### Free Tier Limitations
- 5 API calls per minute
- 500 per day
- Compact data (100 rows max)
- No intraday data
- No adjusted prices

### Premium Plans
See: https://www.alphavantage.co/premium/

Benefits include:
- Unlimited API calls
- Full historical data (20+ years)
- Intraday data at all intervals
- Forex & cryptocurrency
- Direct email support

## Example: Complete Workflow

```python
from semai.alpha_vantage_handler import AlphaVantageHandler
from semai.builtin_models import ModelRegistry
import pandas as pd

# Initialize
api_key = "5E0NVW3VI9N9E6PC"
client = AlphaVantageHandler(api_key)
registry = ModelRegistry()

# Step 1: Search for symbol
symbols = client.get_symbol_search('Tesla')
symbol = symbols.iloc[0]['1_symbol']  # 'TSLA'

# Step 2: Download market data
print(f"Downloading {symbol} data...")
market_data = client.get_daily_data(symbol)
print(f"Got {len(market_data)} trading days")

# Step 3: Create derivative dataset
print("Creating derivative dataset...")
dataset = client.create_derivative_dataset(symbol)
df = pd.read_csv(dataset['filepath'])
print(f"Generated {len(df)} option samples")

# Step 4: Train multiple models
print("Training models...")
X = df[['S', 'K', 'T', 'r', 'sigma']].values
y = df['C'].values

models_to_train = ['deep_learning_small', 'nn_sde_standard', 'ann_standard']

trained_models = {}
for model_name in models_to_train:
    model = registry.get_model(model_name)
    model.train(X, y, epochs=20)
    trained_models[model_name] = model
    print(f"  {model_name}: trained")

# Step 5: Evaluate performance
print("\nModel Performance:")
test_idx = slice(-1000, None)  # Last 1000 samples
X_test = X[test_idx]
y_test = y[test_idx]

for name, model in trained_models.items():
    metrics = model.evaluate(X_test, y_test)
    print(f"  {name}:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

print("\nComplete!")
```

## Next Steps

1. **Real-time Trading**: Integrate with your favorite broker API
2. **Portfolio Risk**: Calculate Greeks (delta, gamma, vega) from model predictions
3. **Backtesting**: Use historical data to validate model performance
4. **Monitoring**: Set up daily data refresh and model retraining
5. **Alert System**: Monitor mispriced options across your data

---

**Created**: December 10, 2025
**API Key**: 5E0NVW3VI9N9E6PC
**Free Tier**: Daily, Weekly, Monthly, Symbol Search, 70+ Indicators
**Premium**: Intraday, Forex, Crypto (see alphavantage.co)
