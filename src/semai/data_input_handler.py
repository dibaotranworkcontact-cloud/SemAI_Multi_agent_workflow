"""
Data Input Handler Module
Allows users to submit data links or use available local datasets
Supports flexible interaction for derivative pricing datasets
"""

import os
import sys
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd


class AlphaVantageAPIHandler:
    """Manages Alpha Vantage API connections for market data extraction"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage handler
        
        Args:
            api_key: Alpha Vantage API key (if None, will check environment variables and defaults)
        """
        # Try to get API key from parameter, environment, or default
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY') or "5E0NVW3VI9N9E6PC"
        self.available = self.api_key is not None
        self.handler = None
        
        if self.available:
            try:
                from .alpha_vantage_handler import AlphaVantageHandler
                self.handler = AlphaVantageHandler(self.api_key)
            except (ImportError, ModuleNotFoundError):
                # Keep API key available but mark handler as not loaded
                # This allows graceful degradation if module is added later
                self.handler = None
    
    def get_available_symbols(self) -> List[str]:
        """Return list of commonly used symbols for quick selection"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'SPY', 'QQQ', 'VTI', 'TSLA', 'META']
    
    def search_symbol(self, keyword: str) -> Optional[pd.DataFrame]:
        """Search for symbols matching keyword"""
        if not self.handler:
            return None
        return self.handler.get_symbol_search(keyword)
    
    def fetch_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch daily stock data for a symbol"""
        if not self.handler:
            return None
        return self.handler.get_daily_data(symbol)
    
    def fetch_weekly_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch weekly stock data for a symbol"""
        if not self.handler:
            return None
        return self.handler.get_weekly_data(symbol)
    
    def create_derivative_dataset(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Create synthetic derivative pricing dataset from market data"""
        if not self.handler:
            return None
        return self.handler.create_derivative_dataset(symbol)


class LocalDatasetManager:
    """Manages available local datasets"""
    
    def __init__(self):
        """Initialize the local dataset manager"""
        self.project_root = Path(__file__).parent.parent.parent
        self.datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> Dict[str, Dict[str, str]]:
        """
        Discover available local datasets
        
        Returns:
            dict: Available datasets with their paths
        """
        datasets = {}
        
        # Check for derivative pricing datasets
        semai_dir = self.project_root / 'src' / 'semai'
        data_dir = semai_dir / 'data'
        
        # Option 1: Check for train/validation split
        train_path = semai_dir / 'derivative_pricing_train.csv'
        val_path = semai_dir / 'derivative_pricing_validation.csv'
        
        if train_path.exists():
            datasets['derivative_pricing'] = {
                'name': 'Derivative Pricing Dataset',
                'train': str(train_path),
                'validation': str(val_path) if val_path.exists() else None,
                'type': 'local',
                'description': 'Black-Scholes simulated call option prices',
                'features': ['S', 'K', 'T', 'q', 'r', 'sigma', 'C']
            }
        
        # Option 2: Check for SPY daily data
        spy_path = data_dir / 'spy_daily_export.csv'
        if spy_path.exists():
            datasets['spy_daily'] = {
                'name': 'SPY Daily Stock Data',
                'train': str(spy_path),
                'type': 'local',
                'description': 'SPY ETF daily price data from Alpha Vantage',
                'features': ['open', 'high', 'low', 'close', 'volume']
            }
        
        # Option 3: Check for any CSV files in data directory
        if data_dir.exists():
            for csv_file in data_dir.glob('*.csv'):
                name = csv_file.stem
                if name not in ['spy_daily_export']:  # Already added above
                    datasets[name] = {
                        'name': name.replace('_', ' ').title(),
                        'train': str(csv_file),
                        'type': 'local',
                        'description': f'Dataset from {csv_file.name}'
                    }
        
        return datasets
    
    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available local datasets
        
        Returns:
            dict: Available datasets
        """
        return self.datasets
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            dict: Dataset information or None
        """
        return self.datasets.get(dataset_name)
    
    def load_dataset(self, dataset_name: str, dataset_type: str = 'train') -> Optional[pd.DataFrame]:
        """
        Load a dataset into memory
        
        Args:
            dataset_name: Name of the dataset
            dataset_type: 'train' or 'test'
            
        Returns:
            pandas.DataFrame or None
        """
        dataset_info = self.get_dataset_info(dataset_name)
        
        if not dataset_info:
            return None
        
        file_key = 'train' if dataset_type.lower() == 'train' else 'test'
        file_path = dataset_info.get(file_key)
        
        if file_path and Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return None
        
        return None
    
    def get_dataset_stats(self, dataset_name: str, dataset_type: str = 'train') -> Optional[Dict[str, Any]]:
        """
        Get statistics about a dataset
        
        Args:
            dataset_name: Name of the dataset
            dataset_type: 'train' or 'test'
            
        Returns:
            dict: Dataset statistics
        """
        df = self.load_dataset(dataset_name, dataset_type)
        
        if df is None:
            return None
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes),
            'missing_values': dict(df.isnull().sum()),
            'description': df.describe().to_dict()
        }


class DataLinkHandler:
    """Handles user-submitted data links for the extraction agent"""
    
    def __init__(self):
        self.data_link: Optional[str] = None
        self.link_type: Optional[str] = None
        self.validation_status: bool = False
        self.error_message: Optional[str] = None
        self.dataset_manager = LocalDatasetManager()
    
    def validate_link(self, link: str) -> bool:
        """
        Validate the provided data link
        
        Args:
            link: User-provided data link/URL
            
        Returns:
            bool: True if link is valid, False otherwise
        """
        if not link or not isinstance(link, str):
            self.error_message = "Link must be a non-empty string"
            return False
        
        link = link.strip()
        
        # Check for common data sources
        if 'kaggle' in link.lower():
            self.link_type = 'kaggle'
            self.validation_status = True
            self.data_link = link
            return True
        elif 'github' in link.lower():
            self.link_type = 'github'
            self.validation_status = True
            self.data_link = link
            return True
        elif link.startswith('http://') or link.startswith('https://'):
            self.link_type = 'url'
            self.validation_status = True
            self.data_link = link
            return True
        elif os.path.isfile(link) or os.path.isdir(link):
            self.link_type = 'local'
            self.validation_status = True
            self.data_link = link
            return True
        else:
            self.error_message = f"Unsupported link format: {link}. Please provide a valid URL, Kaggle link, GitHub link, or local file path."
            self.validation_status = False
            return False
    
    def submit_link(self, link: str) -> Dict[str, Any]:
        """
        Submit a data link for processing
        
        Args:
            link: Data link/URL to be processed
            
        Returns:
            dict: Status and details of the submission
        """
        if self.validate_link(link):
            return {
                "status": "success",
                "message": f"Data link accepted for processing",
                "link": self.data_link,
                "link_type": self.link_type,
                "details": f"The Data Extraction Agent will now fetch data from the {self.link_type} source."
            }
        else:
            return {
                "status": "error",
                "message": "Invalid data link",
                "error": self.error_message,
                "link": link
            }
    
    def get_extraction_context(self) -> Dict[str, str]:
        """
        Get context for the data extraction agent
        
        Returns:
            dict: Context containing the validated data link
        """
        if self.validation_status and self.data_link:
            return {
                "data_link": self.data_link,
                "link_type": self.link_type,
                "instruction": f"Fetch the dataset from this {self.link_type} link: {self.data_link}"
            }
        else:
            return {"error": "No valid data link has been submitted"}
    
    def reset(self):
        """Reset the handler for a new submission"""
        self.data_link = None
        self.link_type = None
        self.validation_status = False
        self.error_message = None


class DataExtractionInputInterface:
    """Interactive interface for users to submit data links or select local datasets"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.handler = DataLinkHandler()
        self.dataset_manager = LocalDatasetManager()
        self.api_handler = AlphaVantageAPIHandler(api_key)
        self.selected_source: Optional[Dict[str, Any]] = None
    
    def _display_local_datasets(self) -> None:
        """Display available local datasets"""
        datasets = self.dataset_manager.list_datasets()
        
        if not datasets:
            print("  [No local datasets found]\n")
            return
        
        print("  Local Datasets:\n")
        for idx, (dataset_id, dataset_info) in enumerate(datasets.items(), 1):
            print(f"    [{idx}] {dataset_info['name']}")
            print(f"        Type: {dataset_info['type']}")
            print(f"        Description: {dataset_info['description']}")
            print(f"        Features: {', '.join(dataset_info['features'])}")
            print()
    
    def _display_api_options(self) -> None:
        """Display Alpha Vantage API data source options"""
        if not self.api_handler.available:
            print("  [Alpha Vantage API not available - API key not configured]\n")
            return
        
        print("  Alpha Vantage API - Stock Market Data:\n")
        print("    Quick Symbols: " + ", ".join(self.api_handler.get_available_symbols()))
        print("    - Download daily, weekly, or monthly data")
        print("    - Auto-generate synthetic derivative pricing datasets")
        print("    - ~100 recent trading days per request (free tier)")
        print()
    
    def interactive_submit(self) -> Optional[Dict[str, Any]]:
        """
        Interactive mode: ask user to choose data source
        
        Returns:
            dict: Submission result with data source info
        """
        print("\n" + "="*80)
        print("DATA EXTRACTION - SELECT DATA SOURCE")
        print("="*80 + "\n")
        
        print("How would you like to provide your dataset?\n")
        print("  [1] Use available local dataset")
        print("  [2] Provide a dataset URL (HTTP/HTTPS)")
        print("  [3] Provide a local file path")
        print("  [4] Fetch from Alpha Vantage API (Stock Market Data)")
        print()
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            return self._select_local_dataset()
        elif choice == '2':
            return self._input_url_source()
        elif choice == '3':
            return self._input_local_file()
        elif choice == '4':
            return self._fetch_from_api()
        else:
            print("\n[Error] Invalid choice. Please select 1, 2, 3, or 4.\n")
            return None
    
    def _fetch_from_api(self) -> Optional[Dict[str, Any]]:
        """
        Fetch data from Alpha Vantage API
        
        Returns:
            dict: API fetch result
        """
        print("\n" + "-"*80)
        
        if not self.api_handler.handler:
            print("[Error] Alpha Vantage API not available.")
            if not self.api_handler.api_key:
                print("Please set ALPHA_VANTAGE_API_KEY environment variable or provide API key.")
            else:
                print("API module not found. Please ensure alpha_vantage_handler.py is in the correct location.")
            print()
            return None
        
        self._display_api_options()
        
        print("Select data type:")
        print("  [1] Daily stock data")
        print("  [2] Weekly stock data")
        print("  [3] Generate synthetic derivative dataset")
        print()
        
        data_type = input("Select option (1-3): ").strip()
        symbol = input("Enter stock symbol (e.g., AAPL, SPY): ").strip().upper()
        
        if not symbol:
            print("\n[Error] Symbol cannot be empty.\n")
            return None
        
        try:
            if data_type == '1':
                print(f"\nFetching daily data for {symbol}...")
                data = self.api_handler.fetch_daily_data(symbol)
                if data is not None:
                    return {
                        "status": "success",
                        "source_type": "alpha_vantage_api",
                        "data_type": "daily",
                        "symbol": symbol,
                        "rows": len(data),
                        "data": data,
                        "message": f"Successfully fetched daily data for {symbol}"
                    }
            elif data_type == '2':
                print(f"\nFetching weekly data for {symbol}...")
                data = self.api_handler.fetch_weekly_data(symbol)
                if data is not None:
                    return {
                        "status": "success",
                        "source_type": "alpha_vantage_api",
                        "data_type": "weekly",
                        "symbol": symbol,
                        "rows": len(data),
                        "data": data,
                        "message": f"Successfully fetched weekly data for {symbol}"
                    }
            elif data_type == '3':
                print(f"\nGenerating synthetic derivative dataset from {symbol} data...")
                dataset_info = self.api_handler.create_derivative_dataset(symbol)
                if dataset_info:
                    return {
                        "status": "success",
                        "source_type": "alpha_vantage_api",
                        "data_type": "derivative_dataset",
                        "symbol": symbol,
                        "rows": dataset_info.get('rows', 0),
                        "filepath": dataset_info.get('filepath'),
                        "features": dataset_info.get('features', []),
                        "message": f"Generated {dataset_info.get('rows')} derivative pricing samples"
                    }
            
            print(f"\n[Error] Failed to fetch data for {symbol}. Check symbol and retry.\n")
            return None
        
        except Exception as e:
            print(f"\n[Error] API error: {str(e)}\n")
            return None
    
    def _select_local_dataset(self) -> Optional[Dict[str, Any]]:
        """
        Let user select from available local datasets
        
        Returns:
            dict: Selected dataset information
        """
        print("\n" + "-"*80)
        datasets = self.dataset_manager.list_datasets()
        
        if not datasets:
            print("❌ No local datasets available.\n")
            return None
        
        self._display_local_datasets()
        
        dataset_list = list(datasets.keys())
        
        if len(dataset_list) == 1:
            # Auto-select if only one dataset
            selected_id = dataset_list[0]
            print(f"✅ Auto-selected: {datasets[selected_id]['name']}\n")
        else:
            choice = input("Select dataset (enter number): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(dataset_list):
                    selected_id = dataset_list[idx]
                else:
                    print("❌ Invalid selection.\n")
                    return None
            except ValueError:
                print("❌ Invalid input.\n")
                return None
        
        dataset_info = datasets[selected_id]
        
        # Ask for train or validation
        print("\nSelect dataset subset:")
        print("  [1] Training set (240,000 samples - 80%)")
        print("  [2] Validation set (60,000 samples - 20%)")
        print()
        
        subset_choice = input("Select subset (1-2): ").strip()
        subset_type = 'train' if subset_choice == '1' else 'test'
        
        self.selected_source = {
            'status': 'success',
            'source_type': 'local_dataset',
            'dataset_name': selected_id,
            'dataset_display_name': dataset_info['name'],
            'subset': subset_type,
            'file_path': dataset_info[subset_type],
            'features': dataset_info['features'],
            'description': dataset_info['description'],
            'message': f"Using {dataset_info['name']} ({subset_type} set)"
        }
        
        print(f"\n✅ {self.selected_source['message']}")
        print(f"   File: {self.selected_source['file_path']}")
        print(f"   Features: {', '.join(self.selected_source['features'])}\n")
        print("="*80 + "\n")
        
        return self.selected_source
    
    def _input_url_source(self) -> Optional[Dict[str, Any]]:
        """
        Get URL input from user
        
        Returns:
            dict: Submission result
        """
        print("\n" + "-"*80)
        print("Enter your dataset URL. Supported formats:")
        print("  • Kaggle: https://www.kaggle.com/datasets/...")
        print("  • GitHub: https://github.com/.../raw/main/...")
        print("  • Direct URL: https://example.com/data.csv")
        print()
        
        data_link = input("Enter dataset URL: ").strip()
        
        if not data_link:
            print("\n❌ No URL provided.\n")
            return None
        
        result = self.handler.submit_link(data_link)
        
        if result["status"] == "success":
            self.selected_source = {
                'status': 'success',
                'source_type': 'url',
                'link': result['link'],
                'link_type': result['link_type'],
                'message': f"URL accepted: {result['link_type'].upper()} source"
            }
            print(f"\n✅ Success: {self.selected_source['message']}")
            print(f"   URL: {result['link']}\n")
            print("="*80 + "\n")
            return self.selected_source
        else:
            print(f"\n❌ Error: {result['error']}\n")
            return None
    
    def _input_local_file(self) -> Optional[Dict[str, Any]]:
        """
        Get local file path from user
        
        Returns:
            dict: Submission result
        """
        print("\n" + "-"*80)
        print("Enter the path to your local data file:")
        print()
        
        file_path = input("Enter file path: ").strip()
        
        if not file_path or not os.path.isfile(file_path):
            print(f"\n❌ File not found: {file_path}\n")
            return None
        
        result = self.handler.submit_link(file_path)
        
        if result["status"] == "success":
            self.selected_source = {
                'status': 'success',
                'source_type': 'local_file',
                'file_path': result['link'],
                'message': f"Local file accepted"
            }
            print(f"\n✅ Success: {self.selected_source['message']}")
            print(f"   File: {result['link']}\n")
            print("="*80 + "\n")
            return self.selected_source
        else:
            print(f"\n❌ Error: {result['error']}\n")
            return None
    
    def programmatic_submit(self, data_link: str) -> Dict[str, Any]:
        """
        Programmatic mode: submit link via code
        
        Args:
            data_link: The data link to submit
            
        Returns:
            dict: Submission result
        """
        return self.handler.submit_link(data_link)
    
    def get_crew_context(self) -> Dict[str, Any]:
        """
        Get the context to pass to the crew
        
        Returns:
            dict: Context with validated data source
        """
        if self.selected_source:
            return self.selected_source
        return self.handler.get_extraction_context()
    
    def reset(self):
        """Reset for next submission"""
        self.handler.reset()
        self.selected_source = None


def get_data_link_from_user() -> Optional[Dict[str, Any]]:
    """
    Convenience function to get data source from user
    
    Returns:
        dict: Data source information or None
    """
    interface = DataExtractionInputInterface()
    result = interface.interactive_submit()
    
    if result and result["status"] == "success":
        return result
    return None


if __name__ == "__main__":
    # Example usage
    interface = DataExtractionInputInterface()
    
    # Interactive mode
    result = interface.interactive_submit()
    
    if result and result["status"] == "success":
        context = interface.get_crew_context()
        print("Context for Crew:")
        print(context)

