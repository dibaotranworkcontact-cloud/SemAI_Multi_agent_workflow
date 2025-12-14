"""
Derivative Pricing Dataset Generator
Generates simulated call option prices for training and test sets.
Based on parameters from Table 1 specifications.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import os


class DerivativePricingDatasetGenerator:
    """
    Generates simulated call option pricing data using Black-Scholes model.
    Creates 300,000 call option prices with realistic parameter ranges.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the dataset generator.
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Parameter ranges (from Table 1)
        self.param_ranges = {
            'S': (10, 500),           # Stock price: $10 – $500
            'K': (7, 650),            # Strike price: $7 – $650
            'T': (1/252, 3),          # Maturity: 1 day to 3 years (in years)
            'q': (0.0, 0.03),         # Dividend rate: 0% – 3%
            'r': (0.01, 0.03),        # Risk-free rate: 1% – 3%
            'sigma': (0.05, 0.90)     # Volatility: 5% – 90%
        }
        
        self.dataset_size = 300000
        self.train_size = 240000
        self.test_size = 60000
    
    def black_scholes_call(self, S, K, T, r, q, sigma):
        """
        Calculate call option price using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            q: Dividend yield rate
            sigma: Volatility (annualized)
            
        Returns:
            Call option price
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_price
    
    def generate_dataset(self, num_samples=None):
        """
        Generate simulated call option prices.
        
        Args:
            num_samples: Number of samples to generate (default: 300,000)
            
        Returns:
            DataFrame with option parameters and prices
        """
        if num_samples is None:
            num_samples = self.dataset_size
        
        print(f"📊 Generating {num_samples:,} call option prices...")
        print("="*80)
        
        # Initialize parameter arrays
        data = {
            'S': np.random.uniform(self.param_ranges['S'][0], self.param_ranges['S'][1], num_samples),
            'K': np.random.uniform(self.param_ranges['K'][0], self.param_ranges['K'][1], num_samples),
            'T': np.random.uniform(self.param_ranges['T'][0], self.param_ranges['T'][1], num_samples),
            'q': np.random.uniform(self.param_ranges['q'][0], self.param_ranges['q'][1], num_samples),
            'r': np.random.uniform(self.param_ranges['r'][0], self.param_ranges['r'][1], num_samples),
            'sigma': np.random.uniform(self.param_ranges['sigma'][0], self.param_ranges['sigma'][1], num_samples)
        }
        
        # Calculate call option prices using Black-Scholes
        call_prices = []
        for i in range(num_samples):
            call_price = self.black_scholes_call(
                data['S'][i],
                data['K'][i],
                data['T'][i],
                data['r'][i],
                data['q'][i],
                data['sigma'][i]
            )
            call_prices.append(call_price)
            
            # Progress indicator
            if (i + 1) % 50000 == 0:
                print(f"  ✓ Generated {i + 1:,} samples...")
        
        data['C'] = np.array(call_prices)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Validate ranges
        print("\n📈 Dataset Statistics:")
        print("="*80)
        for col in ['S', 'K', 'T', 'q', 'r', 'sigma', 'C']:
            print(f"  {col:6s}: min={df[col].min():10.4f}, max={df[col].max():10.4f}, mean={df[col].mean():10.4f}")
        
        print("="*80)
        
        return df
    
    def split_train_test(self, df):
        """
        Split dataset into training and test sets.
        
        Args:
            df: Full dataset DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Split: 240,000 training, 60,000 test
        train_df = df_shuffled.iloc[:self.train_size].reset_index(drop=True)
        test_df = df_shuffled.iloc[self.train_size:self.train_size + self.test_size].reset_index(drop=True)
        
        print(f"\n✅ Dataset Split:")
        print("="*80)
        print(f"  Training set:    {len(train_df):,} samples (80%)")
        print(f"  Test set:        {len(test_df):,} samples (20%)")
        print("="*80)
        
        return train_df, test_df
    
    def save_datasets(self, train_df, test_df, output_dir=None):
        """
        Save training and test datasets to CSV files.
        
        Args:
            train_df: Training dataset DataFrame
            test_df: Test dataset DataFrame
            output_dir: Output directory (default: current working directory)
            
        Returns:
            Tuple of (train_path, test_path)
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save files
        train_path = os.path.join(output_dir, 'derivative_pricing_train.csv')
        test_path = os.path.join(output_dir, 'derivative_pricing_test.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n💾 Files Saved:")
        print("="*80)
        print(f"  Training:    {train_path}")
        print(f"  Test:        {test_path}")
        print("="*80)
        
        return train_path, test_path


def create_derivative_pricing_dataset(output_dir=None, num_samples=300000):
    """
    Main function to create the derivative pricing dataset.
    
    Args:
        output_dir: Output directory for saving datasets
        num_samples: Total number of samples to generate
        
    Returns:
        Dictionary with dataset information
    """
    print("\n" + "="*80)
    print("🎯 DERIVATIVE PRICING DATASET GENERATION")
    print("="*80)
    print(f"Creating {num_samples:,} simulated call option prices...\n")
    
    # Initialize generator
    generator = DerivativePricingDatasetGenerator()
    
    # Generate full dataset
    full_dataset = generator.generate_dataset(num_samples)
    
    # Split into train and validation
    train_df, validation_df = generator.split_train_validation(full_dataset)
    
    # Save datasets
    train_path, validation_path = generator.save_datasets(train_df, validation_df, output_dir)
    
    result = {
        'train_data': train_df,
        'validation_data': validation_df,
        'train_path': train_path,
        'validation_path': validation_path,
        'full_dataset': full_dataset,
        'metadata': {
            'total_samples': num_samples,
            'train_samples': len(train_df),
            'validation_samples': len(validation_df),
            'parameters': ['S', 'K', 'T', 'q', 'r', 'sigma', 'C'],
            'description': 'Black-Scholes simulated call option prices'
        }
    }
    
    print("\n✅ Dataset generation completed successfully!\n")
    
    return result


if __name__ == "__main__":
    # Generate the dataset
    result = create_derivative_pricing_dataset()
    
    # Display summary
    print("\n📊 Dataset Summary:")
    print("="*80)
    print(f"Total samples: {result['metadata']['total_samples']:,}")
    print(f"Training samples: {result['metadata']['train_samples']:,}")
    print(f"Validation samples: {result['metadata']['validation_samples']:,}")
    print(f"Features: {', '.join(result['metadata']['parameters'])}")
    print("="*80)
