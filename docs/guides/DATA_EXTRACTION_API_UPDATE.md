## Data Extraction Agent - API Integration Update

### Changes Summary

The **Data Extraction Agent** now offers users **4 data source options** instead of just URLs:

#### 1. **Local Datasets**
- Pre-loaded derivative pricing dataset (300k samples)
- Features: S, K, T, q, r, sigma, C

#### 2. **HTTP/HTTPS URLs**
- Kaggle datasets
- GitHub repositories
- Direct URL downloads

#### 3. **Local File Paths**
- Direct file system access
- CSV, JSON, Parquet support

#### 4. **Alpha Vantage API** (NEW)
- Live stock market data
- Daily, weekly, monthly timeframes
- Auto-generate synthetic derivative datasets
- Quick symbols: AAPL, MSFT, GOOGL, AMZN, NVDA, SPY, QQQ, VTI, TSLA, META
- 500 API calls/day free tier

---

### Updated Components

**1. `data_input_handler.py`**
- Added `AlphaVantageAPIHandler` class
- Updated `DataExtractionInputInterface` with 4-option menu
- Added `_fetch_from_api()` method
- Supports programmatic and interactive modes

**2. `config/agents.yaml`**
- Updated `data_extraction_agent` role and backstory
- Now mentions API data source capability
- Emphasizes support for derivatives pricing

**3. `config/tasks.yaml`**
- Updated `data_extraction_task` description
- Documents 3 data source options
- Mentions derivative dataset generation

**4. `data_extraction_demo.py` (NEW)**
- Interactive demo script
- Programmatic usage examples
- Shows all data source capabilities
- Run with: `python data_extraction_demo.py --mode interactive`

---

### User Experience

When users run the data extraction workflow, they see:

```
DATA EXTRACTION - SELECT DATA SOURCE

How would you like to provide your dataset?

  [1] Use available local dataset
  [2] Provide a dataset URL (HTTP/HTTPS)
  [3] Provide a local file path
  [4] Fetch from Alpha Vantage API (Stock Market Data)

Select option (1-4): _
```

If they choose **[4]**:

```
Alpha Vantage API - Stock Market Data:

Quick Symbols: AAPL, MSFT, GOOGL, AMZN, NVDA, SPY, QQQ, VTI, TSLA, META
- Download daily, weekly, or monthly data
- Auto-generate synthetic derivative pricing datasets
- ~100 recent trading days per request (free tier)

Select data type:
  [1] Daily stock data
  [2] Weekly stock data
  [3] Generate synthetic derivative dataset

Select option (1-3): _
Enter stock symbol (e.g., AAPL, SPY): _
```

---

### Code Example

```python
from semai.data_input_handler import DataExtractionInputInterface

# Initialize interface
interface = DataExtractionInputInterface(api_key='5E0NVW3VI9N9E6PC')

# Interactive mode
result = interface.interactive_submit()

# Or programmatic mode
msft_data = interface.api_handler.fetch_daily_data('MSFT')
print(f"Downloaded {len(msft_data)} rows")

# Generate derivative dataset
dataset_info = interface.api_handler.create_derivative_dataset('SPY')
print(f"Generated {dataset_info['rows']} option samples")
```

---

### Features

✅ Seamless integration with existing workflows  
✅ 4 data source options to choose from  
✅ API rate limiting (5 calls/minute)  
✅ Smart caching to avoid duplicate API calls  
✅ Auto-generates synthetic derivative datasets  
✅ Graceful fallback if API module unavailable  
✅ Both interactive and programmatic interfaces  

---

### Configuration

To use the API integration:

1. **Default API Key** (already configured):
   ```python
   interface = DataExtractionInputInterface(api_key='5E0NVW3VI9N9E6PC')
   ```

2. **Environment Variable** (optional):
   ```bash
   export ALPHA_VANTAGE_API_KEY=your_api_key
   ```

3. **No changes needed** - Agent automatically uses available sources

---

### Next Steps

1. Data extraction agent now supports API data sources
2. Users can choose their preferred data source at runtime
3. Seamless integration with rest of ML pipeline
4. Supports derivatives pricing analysis workflow

