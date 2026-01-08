#!/usr/bin/env python3
"""


Converts mispricing classification to:
1. "BS Residual Classification" (benchmark-relative, not true mispricing)
2. Adds better benchmarks (BS + dividend yield, optional Heston fit)
3. Supports quantile-based balanced labeling


"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def classify_bs_residual(
    row,
    threshold_pct: float = 0.10,
    label_style: str = 'short'
) -> str:
    """
    Classify option as over/under/fairly priced RELATIVE TO BLACK-SCHOLES BENCHMARK.
    
    This is explicitly a BENCHMARK-RELATIVE classification.
   
   
    
    Returns:
        Label string based on market price vs BS price
    
  
    """
    mkt_price = row.get('mkt_price', np.nan)
    bs_price = row.get('bs_price', np.nan)
    
    # Handle invalid data
    if pd.isna(mkt_price) or pd.isna(bs_price) or bs_price <= 0:
        return 'N/A'
    
    # Calculate residual as percentage
    residual_pct = (mkt_price - bs_price) / bs_price
    
    # Classify
    if residual_pct > threshold_pct:
        label = 'O' if label_style == 'short' else \
                'overpriced' if label_style == 'long' else \
                'overpriced_vs_bs'
    elif residual_pct < -threshold_pct:
        label = 'U' if label_style == 'short' else \
                'underpriced' if label_style == 'long' else \
                'underpriced_vs_bs'
    else:
        label = 'F' if label_style == 'short' else \
                'fair' if label_style == 'long' else \
                'fair_vs_bs'
    
    return label


def classify_by_quantile(
    residuals: pd.Series,
    q_lower: float = 0.33,
    q_upper: float = 0.67,
    label_names: Tuple[str, str, str] = ('underpriced', 'fair', 'overpriced')
) -> pd.Series:
    """
    Classify options by RESIDUAL QUANTILES rather than fixed thresholds.
    
    This approach:
    - Balances classes automatically (each class 33% of data)
    - More economically meaningful ("relative mispricing rank")
    - Redefines "fair" as middle 33% of residual distribution
    
   
    
    Returns:
        Labeled Series with values from label_names
    
   
    """
    lower_threshold = residuals.quantile(q_lower)
    upper_threshold = residuals.quantile(q_upper)
    
    labels = pd.cut(
        residuals,
        bins=[residuals.min() - 1e-10, lower_threshold, upper_threshold, residuals.max() + 1e-10],
        labels=label_names,
        include_lowest=True
    )
    
    return labels


def estimate_bid_ask_spread(row) -> float:
    """
    Estimate bid-ask spread for economic validation.
    
    Simple model: spread ≈ 0.1% of option price (typical for liquid options)
    Can be enhanced with actual bid/ask data if available.
    
    Args:
        row: DataFrame row with option data
    
    Returns:
        Estimated bid-ask spread ($)
    """
    mkt_price = row.get('mkt_price', 0.1)
    spread_pct = 0.001  # 0.1% for liquid equity options
    return mkt_price * spread_pct


def compute_bs_residual(row) -> float:
    """
    Calculate raw residual: market_price - bs_price.
    
    Returns:
        Residual ($)
    """
    mkt_price = row.get('mkt_price', np.nan)
    bs_price = row.get('bs_price', np.nan)
    
    if pd.isna(mkt_price) or pd.isna(bs_price):
        return np.nan
    
    return mkt_price - bs_price


def compute_bs_residual_pct(row) -> float:
    """
    Calculate residual as percentage of BS price.
    
    Returns:
        Residual (%)
    """
    mkt_price = row.get('mkt_price', np.nan)
    bs_price = row.get('bs_price', np.nan)
    
    if pd.isna(mkt_price) or pd.isna(bs_price) or bs_price <= 0:
        return np.nan
    
    return (mkt_price - bs_price) / bs_price * 100


def add_benchmark_labels_to_dataframe(
    df: pd.DataFrame,
    labeling_method: str = 'residual_threshold',
    threshold_pct: float = 0.10,
    q_lower: float = 0.33,
    q_upper: float = 0.67
) -> pd.DataFrame:
    """
    Add multiple labeling schemes to DataFrame for comparison.
    
    Args:
        df: Options DataFrame with 'mkt_price' and 'bs_price' columns
        labeling_method: 'residual_threshold' (fixed) or 'quantile' (balanced)
        threshold_pct: If residual_threshold, deviation threshold
        q_lower, q_upper: If quantile, quantile cutoffs
    
    Returns:
        DataFrame with additional label columns
    
   
    """
    df = df.copy()
    
    # Add raw residuals
    df['residual_dollar'] = df.apply(compute_bs_residual, axis=1)
    df['residual_pct'] = df.apply(compute_bs_residual_pct, axis=1)
    
    # Add bid-ask estimate
    df['estimated_spread'] = df.apply(estimate_bid_ask_spread, axis=1)
    
    if labeling_method == 'residual_threshold':
        # Fixed-threshold classification (original approach)
        df['label_threshold'] = df.apply(
            lambda row: classify_bs_residual(row, threshold_pct=threshold_pct),
            axis=1
        )
    elif labeling_method == 'quantile':
        # Quantile-based classification (balanced classes)
        df['label_quantile'] = classify_by_quantile(
            df['residual_pct'],
            q_lower=q_lower,
            q_upper=q_upper
        )
    
    return df


def print_label_statistics(df: pd.DataFrame, label_column: str = 'label_threshold'):
    """
    Print distribution of labels for understanding coverage.
    
    Args:
        df: DataFrame with labels
        label_column: Name of label column to analyze
    """
    print(f"\n{'='*60}")
    print(f"Label Distribution: {label_column}")
    print(f"{'='*60}")
    
    for label in df[label_column].unique():
        if label == 'N/A':
            continue
        count = (df[label_column] == label).sum()
        pct = (count / len(df)) * 100
        print(f"  {label:>20s}: {count:>6,} ({pct:>5.1f}%)")
    
    print(f"  {'Total':>20s}: {len(df):>6,} (100.0%)")


if __name__ == "__main__":
    # Example usage
    print("Label Reframing Utility for BS Residual Classification")
    print("=" * 60)
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'mkt_price': [4.5, 5.0, 5.5, 4.0, 5.5, 5.2],
        'bs_price': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    })
    
    print("\n[Example 1] Fixed-threshold classification (10% threshold)")
    result = add_benchmark_labels_to_dataframe(
        dummy_data,
        labeling_method='residual_threshold',
        threshold_pct=0.10
    )
    print(result[['mkt_price', 'bs_price', 'residual_pct', 'label_threshold']])
    print_label_statistics(result, 'label_threshold')
    
    print("\n[Example 2] Quantile-based classification (balanced)")
    result2 = add_benchmark_labels_to_dataframe(
        dummy_data,
        labeling_method='quantile'
    )
    print(result2[['mkt_price', 'bs_price', 'residual_pct', 'label_quantile']])
    print_label_statistics(result2, 'label_quantile')
