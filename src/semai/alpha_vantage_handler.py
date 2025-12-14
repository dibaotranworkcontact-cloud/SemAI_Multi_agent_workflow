"""
Alpha Vantage API Integration Handler
Extracts market data (stock prices, forex, crypto) for derivative pricing
"""

import os
import sys
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlphaVantageHandler:
    """
    Alpha Vantage API handler for extracting market data
    
    Supports:
    - Intraday stock data (1min, 5min, 15min, 30min, 60min)
    - Daily stock data
    - Weekly stock data
    - Monthly stock data
    - Forex rates
    - Cryptocurrency data
    """
    
    # Alpha Vantage API endpoints
    BASE_URL = "https://www.alphavantage.co/query"
    
    # API function types
    INTRADAY_FUNCTIONS = {
        '1min': 'TIME_SERIES_INTRADAY',
        '5min': 'TIME_SERIES_INTRADAY',
        '15min': 'TIME_SERIES_INTRADAY',
        '30min': 'TIME_SERIES_INTRADAY',
        '60min': 'TIME_SERIES_INTRADAY'
    }
    
    DAILY_FUNCTIONS = {
        'daily': 'TIME_SERIES_DAILY',
        'daily_adjusted': 'TIME_SERIES_DAILY_ADJUSTED'
    }
    
    WEEKLY_FUNCTIONS = {
        'weekly': 'TIME_SERIES_WEEKLY',
        'weekly_adjusted': 'TIME_SERIES_WEEKLY_ADJUSTED'
    }
    
    MONTHLY_FUNCTIONS = {
        'monthly': 'TIME_SERIES_MONTHLY',
        'monthly_adjusted': 'TIME_SERIES_MONTHLY_ADJUSTED'
    }
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage handler
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.rate_limit = 5  # API calls per minute
        self.last_call_time = 0
        self.cache_dir = Path(__file__).parent / '.cache'
        self.cache_dir.mkdir(exist_ok=True)
    
    def _throttle_api_call(self):
        """Throttle API calls to respect rate limits"""
        elapsed = time.time() - self.last_call_time
        min_interval = 60.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_call_time = time.time()
    
    def _get_cache_path(self, symbol: str, function: str) -> Path:
        """Get cache file path for a specific request"""
        cache_file = f"{symbol}_{function}.json"
        return self.cache_dir / cache_file
    
    def _load_from_cache(self, symbol: str, function: str) -> Optional[Dict]:
        """Load data from cache if available"""
        cache_path = self._get_cache_path(symbol, function)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {symbol} {function} from cache")
                    return data
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None
    
    def _save_to_cache(self, symbol: str, function: str, data: Dict):
        """Save data to cache"""
        cache_path = self._get_cache_path(symbol, function)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Cached {symbol} {function}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _make_api_call(self, params: Dict[str, str]) -> Optional[Dict]:
        """
        Make API call to Alpha Vantage
        
        Args:
            params: Query parameters
            
        Returns:
            dict: API response or None
        """
        self._throttle_api_call()
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
                return None
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def get_intraday_data(self, symbol: str, interval: str = '1min', 
                         use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get intraday stock data
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            use_cache: Use cached data if available
            
        Returns:
            pandas.DataFrame: Time series data or None
        """
        if interval not in self.INTRADAY_FUNCTIONS:
            logger.error(f"Invalid interval: {interval}")
            return None
        
        function = self.INTRADAY_FUNCTIONS[interval]
        
        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(symbol, function)
            if cached_data:
                return self._parse_time_series(cached_data, function)
        
        # Make API call (use compact for free tier)
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        logger.info(f"Fetching intraday data for {symbol} ({interval})")
        data = self._make_api_call(params)
        
        if data:
            self._save_to_cache(symbol, function, data)
            return self._parse_time_series(data, function)
        
        return None
    
    def get_daily_data(self, symbol: str, adjusted: bool = False,
                      use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get daily stock data
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            adjusted: Return adjusted close prices (Note: requires premium API)
            use_cache: Use cached data if available
            
        Returns:
            pandas.DataFrame: Daily time series or None
        """
        # Note: adjusted prices require premium plan, use standard daily
        function = 'TIME_SERIES_DAILY'
        
        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(symbol, function)
            if cached_data:
                return self._parse_time_series(cached_data, function)
        
        # Make API call (use compact for free tier)
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        logger.info(f"Fetching daily data for {symbol}")
        data = self._make_api_call(params)
        
        if data:
            self._save_to_cache(symbol, function, data)
            return self._parse_time_series(data, function)
        
        return None
    
    def get_weekly_data(self, symbol: str, adjusted: bool = False,
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get weekly stock data
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            adjusted: Return adjusted close prices
            use_cache: Use cached data if available
            
        Returns:
            pandas.DataFrame: Weekly time series or None
        """
        function_key = 'weekly_adjusted' if adjusted else 'weekly'
        function = self.WEEKLY_FUNCTIONS[function_key]
        
        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(symbol, function)
            if cached_data:
                return self._parse_time_series(cached_data, function)
        
        # Make API call (use compact for free tier)
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        logger.info(f"Fetching weekly data for {symbol}")
        data = self._make_api_call(params)
        
        if data:
            self._save_to_cache(symbol, function, data)
            return self._parse_time_series(data, function)
        
        return None
    
    def get_monthly_data(self, symbol: str, adjusted: bool = False,
                        use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get monthly stock data
        
        Args:
            symbol: Stock symbol (e.g., 'QQQ')
            adjusted: Return adjusted close prices
            use_cache: Use cached data if available
            
        Returns:
            pandas.DataFrame: Monthly time series or None
        """
        function_key = 'monthly_adjusted' if adjusted else 'monthly'
        function = self.MONTHLY_FUNCTIONS[function_key]
        
        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(symbol, function)
            if cached_data:
                return self._parse_time_series(cached_data, function)
        
        # Make API call (use compact for free tier)
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        logger.info(f"Fetching monthly data for {symbol}")
        data = self._make_api_call(params)
        
        if data:
            self._save_to_cache(symbol, function, data)
            return self._parse_time_series(data, function)
        
        return None
    
    def _parse_time_series(self, data: Dict, function: str) -> Optional[pd.DataFrame]:
        """
        Parse time series data from API response
        
        Args:
            data: API response dictionary
            function: Function name (to identify correct key)
            
        Returns:
            pandas.DataFrame: Parsed time series or None
        """
        # Find the time series key (varies by function)
        time_series_key = None
        for key in data.keys():
            if key.startswith('Time Series'):
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error("No time series data found in response")
            return None
        
        try:
            time_series = data[time_series_key]
            df_dict = {}
            
            for timestamp, ohlcv in time_series.items():
                df_dict[timestamp] = ohlcv
            
            df = pd.DataFrame.from_dict(df_dict, orient='index')
            
            # Convert columns to lowercase
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Parse index as datetime
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert numeric columns
            numeric_cols = df.select_dtypes(include=['object']).columns
            for col in numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            
            logger.info(f"Parsed {len(df)} rows of time series data")
            return df
        
        except Exception as e:
            logger.error(f"Error parsing time series: {e}")
            return None
    
    def get_symbol_search(self, keywords: str) -> Optional[pd.DataFrame]:
        """
        Search for symbols by keywords
        
        Args:
            keywords: Search keywords (e.g., 'Apple')
            
        Returns:
            pandas.DataFrame: Search results or None
        """
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': keywords,
            'apikey': self.api_key
        }
        
        logger.info(f"Searching for symbols: {keywords}")
        data = self._make_api_call(params)
        
        if data and 'bestMatches' in data:
            try:
                results = data['bestMatches']
                df = pd.DataFrame(results)
                # Keep original column names but make them accessible
                df.columns = [col.lower().replace(' ', '_').replace('.', '') for col in df.columns]
                logger.info(f"Found {len(df)} symbols matching '{keywords}'")
                return df
            except Exception as e:
                logger.error(f"Error parsing search results: {e}")
                return None
        
        return None
    
    def get_technical_indicators(self, symbol: str, function: str = 'SMA',
                                interval: str = 'daily', time_period: int = 20,
                                series_type: str = 'close') -> Optional[pd.DataFrame]:
        """
        Get technical indicator data
        
        Args:
            symbol: Stock symbol
            function: Indicator function (SMA, EMA, MACD, RSI, etc.)
            interval: Time interval
            time_period: Period for the indicator
            series_type: Series to use (open, close, high, low)
            
        Returns:
            pandas.DataFrame: Indicator data or None
        """
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type,
            'apikey': self.api_key
        }
        
        logger.info(f"Fetching {function} for {symbol}")
        data = self._make_api_call(params)
        
        if data:
            return self._parse_time_series(data, function)
        
        return None
    
    def export_to_csv(self, df: pd.DataFrame, filename: str, 
                      output_dir: Optional[str] = None) -> Optional[str]:
        """
        Export data to CSV file
        
        Args:
            df: DataFrame to export
            filename: Output filename
            output_dir: Output directory (default: data directory)
            
        Returns:
            str: Path to saved file or None
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / 'data'
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            filepath = output_dir / filename
            df.to_csv(filepath)
            logger.info(f"Exported data to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return None
    
    def create_derivative_dataset(self, symbol: str, start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create derivative pricing dataset from market data
        
        Generates synthetic option prices based on downloaded stock data
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            dict: Dataset info and path
        """
        # Get daily stock data
        df = self.get_daily_data(symbol, adjusted=True)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return None
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if df.empty:
            logger.error(f"No data found for {symbol} in date range")
            return None
        
        # Generate synthetic option prices (Black-Scholes)
        try:
            from scipy.stats import norm
            
            dataset_rows = []
            S = df['4._close'].values
            
            r = 0.02  # Risk-free rate
            q = 0.01  # Dividend yield
            
            for idx in range(len(S) - 252):
                # Current spot price
                spot = float(S[idx])
                
                # Compute historical volatility (1-year rolling)
                window = min(252, idx + 1)
                returns = np.log(S[max(0, idx-window+1):idx+1] / S[max(0, idx-window):idx])
                sigma = np.std(returns) * np.sqrt(252)
                
                # Generate option parameters
                for K_mult in [0.90, 0.95, 1.0, 1.05, 1.10]:
                    K = spot * K_mult
                    for days_to_exp in [30, 60, 90, 180]:
                        T = days_to_exp / 365.0
                        
                        # Black-Scholes call price
                        d1 = (np.log(spot/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                        d2 = d1 - sigma*np.sqrt(T)
                        C = spot * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                        
                        dataset_rows.append({
                            'S': spot,
                            'K': K,
                            'T': T,
                            'q': q,
                            'r': r,
                            'sigma': sigma,
                            'C': max(C, 0.01)  # Ensure positive price
                        })
            
            dataset_df = pd.DataFrame(dataset_rows)
            
            # Export
            import numpy as np
            output_file = f"{symbol}_derivative_dataset.csv"
            filepath = self.export_to_csv(dataset_df, output_file)
            
            return {
                'symbol': symbol,
                'rows': len(dataset_df),
                'filepath': filepath,
                'features': ['S', 'K', 'T', 'q', 'r', 'sigma', 'C'],
                'description': f'Synthetic derivative dataset from {symbol} market data',
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}"
            }
        
        except ImportError:
            logger.error("Scipy not available for Black-Scholes calculation")
            return None
        except Exception as e:
            logger.error(f"Error creating derivative dataset: {e}")
            return None


def create_alpha_vantage_client(api_key: str) -> AlphaVantageHandler:
    """
    Factory function to create Alpha Vantage client
    
    Args:
        api_key: Alpha Vantage API key
        
    Returns:
        AlphaVantageHandler: Initialized client
    """
    return AlphaVantageHandler(api_key)


if __name__ == "__main__":
    # Example usage
    API_KEY = "5E0NVW3VI9N9E6PC"
    
    client = AlphaVantageHandler(API_KEY)
    
    # Get daily data for Apple
    print("Fetching AAPL daily data...")
    aapl_data = client.get_daily_data('AAPL', adjusted=True)
    if aapl_data is not None:
        print(f"Downloaded {len(aapl_data)} rows")
        print(aapl_data.head())
        
        # Export to CSV
        client.export_to_csv(aapl_data, 'aapl_daily.csv')
        
        # Create derivative dataset
        print("\nCreating derivative dataset...")
        dataset_info = client.create_derivative_dataset('AAPL')
        if dataset_info:
            print(f"Dataset created: {dataset_info}")
    
    # Search for symbols
    print("\nSearching for Apple symbols...")
    search_results = client.get_symbol_search('Apple')
    if search_results is not None:
        print(search_results[['1._symbol', '2._name']].head())
