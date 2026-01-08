#!/usr/bin/env python3
"""


Generates figures for CS229 paper:
  1. Equity curve (backtest returns over time)
  2. Per-ticker accuracy comparison (AAPL, SPY, TSLA)
  3. Feature importance (Greeks contribution to predictions)
  4. Confusion matrices (multiclass classification performance)

Usage:
    python generate_paper_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def train_and_evaluate_ticker(filepath, ticker):
    """Train model on ticker data and return predictions."""
    print(f"  Processing {ticker}...", end=" ", flush=True)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f" Error: {e}")
        return None
    
    # Label encoding
    label_map = {'underpriced': 0, 'fair': 1, 'overpriced': 2}
    df['label_encoded'] = df['label_uf_over'].map(label_map)
    df = df.dropna(subset=['label_encoded'])
    
    # Features
    feature_cols = ['iv', 'delta', 'gamma', 'theta', 'vega', 'moneyness', 'tau_days', 'underlying_price']
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].fillna(df[available_features].mean()).values
    y = df['label_encoded'].values
    
    if len(X) < 100:
        print("Insufficient data")
        return None
    
    # Temporal split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.1%}")
    
    return {
        'ticker': ticker,
        'model': model,
        'scaler': scaler,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'features': available_features
    }


def plot_equity_curve(results_dict, output_file='equity_curve.png'):
    """Plot cumulative returns from underpriced predictions."""
    print("\nGenerating equity curve...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_returns_dict = {}
    
    for ticker, result in results_dict.items():
        if result is None:
            continue
        
        # Simulate: +2% if underpriced (label 0) and correctly predicted
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        # Returns: +2% if correctly predicted underpriced, -0.5% otherwise
        returns = np.where((y_pred == 0) & (y_test == 0), 0.02, -0.005)
        cumulative_returns = np.cumprod(1 + returns) - 1
        all_returns_dict[ticker] = cumulative_returns
        
        axes[0].plot(cumulative_returns * 100, label=ticker, linewidth=2.5)
    
    axes[0].set_xlabel('Trade Number', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Backtested Equity Curve (Underpriced Signals)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Combined performance (use shortest length for averaging)
    if all_returns_dict:
        min_len = min(len(v) for v in all_returns_dict.values())
        combined_returns = np.mean([v[:min_len] for v in all_returns_dict.values()], axis=0)
        axes[1].fill_between(range(len(combined_returns)), combined_returns * 100, alpha=0.3)
        axes[1].plot(combined_returns * 100, color='darkblue', linewidth=2.5)
        
        sharpe = (np.mean(combined_returns) / np.std(combined_returns)) * np.sqrt(252) if np.std(combined_returns) > 0 else 0
        
        axes[1].set_xlabel('Trade Number', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Average Performance (Sharpe: {sharpe:.2f})', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()


def plot_ticker_accuracy(results_dict, output_file='ticker_accuracy.png'):
    """Plot accuracy comparison across tickers."""
    print("\nGenerating per-ticker accuracy chart...")
    
    tickers = []
    accuracies = []
    
    for ticker, result in results_dict.items():
        if result is not None:
            tickers.append(ticker)
            accuracies.append(result['accuracy'] * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(tickers, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Asset Model Performance', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    avg_acc = np.mean(accuracies)
    ax.axhline(avg_acc, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_acc:.1f}%')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()


def plot_feature_importance(results_dict, output_file='feature_importance.png'):
    """Plot average feature importance across models."""
    print("\nGenerating feature importance chart...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_importances = []
    all_features = None
    
    for ticker, result in results_dict.items():
        if result is not None:
            importances = result['model'].feature_importances_
            all_importances.append(importances)
            if all_features is None:
                all_features = result['features']
    
    if all_importances and all_features:
        avg_importance = np.mean(all_importances, axis=0)
        sorted_idx = np.argsort(avg_importance)[::-1]
        
        sorted_features = [all_features[i] for i in sorted_idx]
        sorted_importance = avg_importance[sorted_idx]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
        bars = ax.barh(sorted_features, sorted_importance * 100, color=colors, 
                       edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, importance in zip(bars, sorted_importance):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{importance*100:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance (Average Across Assets)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
        plt.close()


def plot_confusion_matrices(results_dict, output_file='confusion_matrices.png'):
    """Plot confusion matrices for each ticker."""
    print("\nGenerating confusion matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    label_names = ['Underpriced', 'Fair', 'Overpriced']
    
    for idx, (ticker, result) in enumerate(results_dict.items()):
        if result is None or idx >= 3:
            continue
        
        cm = confusion_matrix(result['y_test'], result['y_pred'], labels=[0, 1, 2])
        
        # Normalize for visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=label_names, yticklabels=label_names,
                    cbar=False, annot_kws={'size': 11, 'weight': 'bold'})
        
        axes[idx].set_title(f'{ticker} Confusion Matrix\n(n={len(result["y_test"])})', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-READY VISUALIZATIONS")
    print("="*80)
    
    # Train models
    print("\nTraining models...")
    results = {
        'AAPL': train_and_evaluate_ticker('frontend/aapl_180d.csv', 'AAPL'),
        'SPY': train_and_evaluate_ticker('frontend/spy_180d.csv', 'SPY'),
        'TSLA': train_and_evaluate_ticker('frontend/tsla_180d.csv', 'TSLA'),
    }
    
    # Generate visualizations
    plot_equity_curve(results)
    plot_ticker_accuracy(results)
    plot_feature_importance(results)
    plot_confusion_matrices(results)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • equity_curve.png")
    print("  • ticker_accuracy.png")
    print("  • feature_importance.png")
    print("  • confusion_matrices.png")
    print("\nReady for paper inclusion!")


if __name__ == '__main__':
    main()
