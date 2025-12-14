# Data Extraction - Flexible Input System

## Overview
The data extraction system has been enhanced to flexibly interact with users, supporting both URL-based datasets and local derivative pricing datasets.

## Features

### 1. **Multiple Data Source Options**
Users can now choose between:
- **Local Derivative Pricing Dataset** - Pre-generated 300k call option prices
- **URL-based Dataset** - Any dataset from Kaggle, GitHub, or direct URLs
- **Local File Path** - Any CSV file on the local system

### 2. **Interactive User Interface**
The system presents users with a clear menu:

```
📊 DATA EXTRACTION - SELECT DATA SOURCE
================================================

How would you like to provide your dataset?

  [1] Use available local dataset
  [2] Provide a dataset URL
  [3] Provide a local file path

Select option (1-3):
```

### 3. **Automatic Dataset Discovery**
The `LocalDatasetManager` automatically discovers available datasets:
- Scans for `derivative_pricing_train.csv` and `derivative_pricing_test.csv`
- Displays dataset metadata and statistics
- Shows features, sample counts, and descriptions

### 4. **Flexible Data Input Handler**

#### Classes:
- **`LocalDatasetManager`**: Manages and discovers local datasets
- **`DataLinkHandler`**: Validates and processes data links
- **`DataExtractionInputInterface`**: Interactive interface for user input

#### Methods:
- `list_datasets()` - Shows all available local datasets
- `get_dataset_info()` - Get details about a specific dataset
- `load_dataset()` - Load dataset into pandas DataFrame
- `get_dataset_stats()` - Get statistical summary of datasets
- `interactive_submit()` - Interactive user interface for data selection
- `_select_local_dataset()` - Let user choose from local datasets
- `_input_url_source()` - Get URL from user
- `_input_local_file()` - Get local file path from user

## Usage Flow

### For Local Derivative Pricing Dataset:
1. User selects option [1]
2. System displays available datasets
3. User selects desired dataset (auto-selected if only one)
4. User chooses between training (240k) or validation (60k) subset
5. Dataset path is passed to the data extraction agent

### For URL-based Dataset:
1. User selects option [2]
2. User enters URL (Kaggle, GitHub, or direct)
3. System validates the URL format
4. URL is passed to the data extraction agent

### For Local File:
1. User selects option [3]
2. User enters file path
3. System verifies file exists
4. File path is passed to the data extraction agent

## Data Return Format

The system returns standardized information:

```python
{
    'status': 'success',
    'source_type': 'local_dataset',  # or 'url', 'local_file'
    'dataset_name': 'derivative_pricing',
    'dataset_display_name': 'Derivative Pricing Dataset',
    'subset': 'train',  # or 'test'
    'file_path': '/path/to/derivative_pricing_train.csv',
    'features': ['S', 'K', 'T', 'q', 'r', 'sigma', 'C'],
    'description': 'Black-Scholes simulated call option prices...',
    'message': 'Using Derivative Pricing Dataset (train set)'
}
```

## Integration with Main Crew

The enhanced `run()` function now:
1. Calls `get_dataset_path()` which uses the new interface
2. Displays dataset information based on source type
3. Prepares appropriate inputs for crew execution
4. Shows whether it's a local dataset with subset info or external URL

## Available Local Datasets

### Derivative Pricing Dataset
- **Name**: Derivative Pricing Dataset
- **Type**: Local (simulated)
- **Total Samples**: 300,000
- **Training Samples**: 240,000 (80%)
- **Validation Samples**: 60,000 (20%)
- **Features**: S, K, T, q, r, sigma, C (7 columns)
- **Description**: Black-Scholes simulated call option prices with realistic parameter ranges

## Example Usage

```python
from semai.data_input_handler import DataExtractionInputInterface

# Create interface
interface = DataExtractionInputInterface()

# Get user input
result = interface.interactive_submit()

# Get context for crew
if result and result['status'] == 'success':
    crew_context = interface.get_crew_context()
    print(crew_context)
```

## Benefits

✅ **User-Friendly**: Clear, intuitive menu system
✅ **Flexible**: Supports multiple data sources
✅ **Extensible**: Easy to add new datasets
✅ **Automatic Discovery**: Finds available datasets automatically
✅ **Validation**: Validates all inputs before processing
✅ **Informative**: Shows dataset details and statistics
✅ **Integration Ready**: Works seamlessly with CrewAI agents

## Future Enhancements

- Add support for database connections (SQL, MongoDB, etc.)
- Support for streaming large datasets
- Caching mechanism for frequently used datasets
- Dataset preview functionality
- Support for multiple data formats (Parquet, HDF5, etc.)

---
**Updated**: December 9, 2025
**Version**: 2.0
