"""
Data Extraction Demo - Shows how to use the updated data input handler
with support for local datasets, URLs, and Alpha Vantage API
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_input_handler import DataExtractionInputInterface


def demo_interactive_mode():
    """Run interactive data extraction demo"""
    
    # Initialize with API key (can be passed here or set via environment)
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or '5E0NVW3VI9N9E6PC'
    
    interface = DataExtractionInputInterface(api_key=api_key)
    
    print("\n" + "="*80)
    print("DATA EXTRACTION INTERFACE - INTERACTIVE DEMO")
    print("="*80)
    print("\nThis demo shows how to fetch data from multiple sources:")
    print("  1. Local datasets (pre-loaded)")
    print("  2. HTTP/HTTPS URLs (Kaggle, GitHub, etc.)")
    print("  3. Local file paths")
    print("  4. Alpha Vantage API (Live Stock Market Data)")
    print("="*80 + "\n")
    
    # Run interactive submission
    result = interface.interactive_submit()
    
    if result:
        print("\n" + "="*80)
        print("DATA EXTRACTION RESULT")
        print("="*80)
        
        if result.get('status') == 'success':
            print(f"\n[SUCCESS] {result.get('message')}")
            print(f"\nSource Type: {result.get('source_type')}")
            
            if result.get('source_type') == 'alpha_vantage_api':
                print(f"Data Type: {result.get('data_type')}")
                print(f"Symbol: {result.get('symbol')}")
                print(f"Rows: {result.get('rows')}")
                
                if result.get('filepath'):
                    print(f"Saved to: {result.get('filepath')}")
                
                if result.get('features'):
                    print(f"Features: {', '.join(result.get('features'))}")
                
                if result.get('data') is not None:
                    print(f"\nData Preview:")
                    print(result.get('data').head())
            else:
                print(f"Details: {result}")
        else:
            print(f"\n[ERROR] {result.get('message')}")
            print(f"Error: {result.get('error')}")
    else:
        print("\n[ERROR] Data extraction cancelled.\n")


def demo_programmatic_mode():
    """Demonstrate programmatic usage without interaction"""
    
    print("\n" + "="*80)
    print("DATA EXTRACTION - PROGRAMMATIC MODE")
    print("="*80 + "\n")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or '5E0NVW3VI9N9E6PC'
    interface = DataExtractionInputInterface(api_key=api_key)
    
    # Example 1: Get available local datasets
    print("[1] Available Local Datasets:")
    print("-" * 80)
    datasets = interface.dataset_manager.list_datasets()
    for name, info in datasets.items():
        print(f"  - {info['name']}: {info['description']}")
    print()
    
    # Example 2: Check API availability
    print("[2] Alpha Vantage API Status:")
    print("-" * 80)
    print(f"  Available: {interface.api_handler.available}")
    if interface.api_handler.available:
        print(f"  Quick Symbols: {', '.join(interface.api_handler.get_available_symbols())}")
    print()
    
    # Example 3: Fetch stock data (if API available)
    if interface.api_handler.available:
        print("[3] Sample API Call - Fetch AAPL Daily Data:")
        print("-" * 80)
        data = interface.api_handler.fetch_daily_data('AAPL')
        if data is not None:
            print(f"  Rows: {len(data)}")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Date range: {data.index[-1].date()} to {data.index[0].date()}")
            print(f"\n  Latest Data:")
            print(f"    Close: ${float(data.iloc[0]['4._close']):.2f}")
            print(f"    Volume: {int(float(data.iloc[0]['5._volume'])):,}")
        else:
            print("  [Failed to fetch data]")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Extraction Demo')
    parser.add_argument('--mode', choices=['interactive', 'programmatic', 'both'], 
                        default='interactive',
                        help='Demo mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        demo_interactive_mode()
    elif args.mode == 'programmatic':
        demo_programmatic_mode()
    elif args.mode == 'both':
        demo_programmatic_mode()
        print("\n" + "="*80 + "\n")
        demo_interactive_mode()
