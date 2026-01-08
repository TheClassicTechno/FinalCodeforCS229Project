#!/usr/bin/env python3
"""
Train models on REAL MARKET DATA (AAPL options) with deep learning baseline.
This script reproduces all experiments from the paper using actual market data.
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                            confusion_matrix, roc_auc_score)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

def load_real_market_data(ticker='AAPL', use_enhanced_features=False):
    """
    Load real market data from frontend folder.
    
    Args:
        ticker: One of 'AAPL', 'TSLA', 'MU'
        use_enhanced_features: Whether to create enhanced features
    
    Returns:
        X: Feature matrix
        y: Labels (encoded to 0, 1, 2)
        feature_names: List of feature names
    """
    print(f"\n{'='*80}")
    print(f"Loading {ticker} Real Market Data")
    print(f"{'='*80}")
    
    # Load data
    file_map = {
        'AAPL': 'frontend/aapl_180d.csv',
        'SPY': 'frontend/spy_180d.csv',
        'TSLA': 'frontend/tsla_180d.csv',
        'MU': 'frontend/mu_options.csv'
    }
    
    df = pd.read_csv(file_map[ticker])
    print(f"[LOADED] {len(df):,} contracts")
    
    # Base features (same as paper)
    base_features = [
        'moneyness', 'tau_days', 'iv',
        'delta', 'gamma', 'theta', 'vega', 'vix'
    ]
    
    X = df[base_features].copy()
    feature_names = base_features.copy()
    
    print(f"   Base features: {len(base_features)}")
    
    # Enhanced features (if requested)
    if use_enhanced_features:
        # Greek interactions
        X['delta_gamma'] = X['delta'] * X['gamma']
        X['delta_vega'] = X['delta'] * X['vega']
        X['gamma_vega_ratio'] = X['gamma'] / (X['vega'] + 1e-10)
        
        # Moneyness features
        X['moneyness_abs_atm'] = np.abs(X['moneyness'] - 1.0)
        X['moneyness_squared'] = X['moneyness'] ** 2
        X['log_moneyness'] = np.log(X['moneyness'] + 1e-10)
        
        # Time features
        X['theta_tau'] = X['theta'] / (X['tau_days'] + 1)
        X['tau_iv'] = X['tau_days'] * X['iv']
        X['tau_buckets'] = pd.cut(X['tau_days'], bins=5, labels=False)
        
        # Volatility features
        X['iv_vix_ratio'] = X['iv'] / (X['vix'] + 1e-10)
        X['iv_zscore'] = (X['iv'] - X['iv'].mean()) / X['iv'].std()
        
        # Vega/tau (from paper's misclassification example)
        X['vega_tau'] = X['vega'] / (X['tau_days'] + 1)
        
        enhanced_features = [col for col in X.columns if col not in base_features]
        feature_names.extend(enhanced_features)
        print(f"   Enhanced features: {len(enhanced_features)}")
        print(f"   Total features: {len(feature_names)}")
    
    # Extract labels and encode
    y = df['label_uf_over'].copy()
    
    # Map labels to numeric (same as synthetic data)
    label_map = {'underpriced': -1, 'fair': 0, 'overpriced': 1}
    y = y.map(label_map)
    
    # Encode to 0, 1, 2 for sklearn
    y = y + 1
    
    print(f"\n   Label distribution:")
    for label, name in [(0, 'underpriced'), (1, 'fair'), (2, 'overpriced')]:
        count = (y == label).sum()
        pct = (count / len(y)) * 100
        print(f"      {name:>12s} ({label}): {count:>6,} ({pct:>5.1f}%)")
    
    return X.values, y.values, feature_names, df

def temporal_train_test_split(df, X, y, test_size=0.2):
    """
    Split data chronologically (most recent dates for testing).
    """
    print(f"\n{'='*80}")
    print(f"Temporal Train/Test Split")
    print(f"{'='*80}")
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    sorted_idx = df['date'].argsort()
    
    # Split chronologically
    split_idx = int(len(X) * (1 - test_size))
    
    train_idx = sorted_idx[:split_idx]
    test_idx = sorted_idx[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    train_dates = df.iloc[train_idx]['date']
    test_dates = df.iloc[test_idx]['date']
    
    print(f"   Training: {len(X_train):>6,} contracts ({train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')})")
    print(f"   Testing:  {len(X_test):>6,} contracts ({test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')})")
    
    return X_train, X_test, y_train, y_test

def calculate_f1_with_cv(model, X_train, y_train, X_test, y_test, model_name, scaler):
    """Calculate F1 scores with cross-validation."""
    print(f"\n{'-'*80}")
    print(f"Training: {model_name}")
    print(f"{'-'*80}")
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test_scaled)
    
    # Test F1-macro
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_acc = accuracy_score(y_test, y_pred)
    
    # Cross-validation F1-macro
    cv = TimeSeriesSplit(n_splits=5)
    cv_f1_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_f1_mean = cv_f1_scores.mean()
    cv_f1_std = cv_f1_scores.std()
    
    # Cross-validation accuracy
    cv_acc_scores = cross_val_score(model, X_train_scaled, y_train,
                                    cv=cv, scoring='accuracy', n_jobs=-1)
    cv_acc_mean = cv_acc_scores.mean()
    cv_acc_std = cv_acc_scores.std()
    
    print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"   CV Accuracy:    {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print(f"   Test F1-macro:  {test_f1:.4f} ({test_f1*100:.1f}%)")
    print(f"   CV F1-macro:    {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    
    # Classification report
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, 
                                  target_names=['underpriced', 'fair', 'overpriced'])
    for line in report.split('\n'):
        if line.strip():
            print(f"      {line}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"      Predicted:  U    F    O")
    for i, label in enumerate(['U', 'F', 'O']):
        print(f"      Actual {label}:  {cm[i, 0]:>4} {cm[i, 1]:>4} {cm[i, 2]:>4}")
    
    return {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'cv_accuracy_mean': float(cv_acc_mean),
        'cv_accuracy_std': float(cv_acc_std),
        'cv_f1_mean': float(cv_f1_mean),
        'cv_f1_std': float(cv_f1_std),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def train_all_models(X_train, X_test, y_train, y_test, feature_type='enhanced'):
    """Train all models from the paper."""
    print(f"\n{'='*80}")
    print(f"Training All Models ({feature_type} features)")
    print(f"{'='*80}")
    
    results = {}
    scaler = StandardScaler()
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    results['Random Forest'] = calculate_f1_with_cv(
        rf, X_train, y_train, X_test, y_test, 'Random Forest', StandardScaler()
    )
    
    # 2. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    results['Gradient Boosting'] = calculate_f1_with_cv(
        gb, X_train, y_train, X_test, y_test, 'Gradient Boosting', StandardScaler()
    )
    
    # 3. Logistic Regression
    lr = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=1000, random_state=42)
    results['Logistic Regression'] = calculate_f1_with_cv(
        lr, X_train, y_train, X_test, y_test, 'Logistic Regression', StandardScaler()
    )
    
    # 4. Voting Ensemble
    voting = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(C=1.0, multi_class='multinomial', max_iter=1000, random_state=42))
        ],
        voting='soft'
    )
    results['Voting Ensemble'] = calculate_f1_with_cv(
        voting, X_train, y_train, X_test, y_test, 'Voting Ensemble', StandardScaler()
    )
    
    # 5. **NEW** Deep Learning Baseline (Multi-Layer Perceptron)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    results['MLP (Deep Learning)'] = calculate_f1_with_cv(
        mlp, X_train, y_train, X_test, y_test, 'MLP (Deep Learning)', StandardScaler()
    )
    
    return results

def main():
    """Main execution."""
    print("\n" + "="*80)
    print("REAL MARKET DATA EXPERIMENTS")
    print("Training on AAPL options with Deep Learning Baseline")
    print("="*80 + "\n")
    
    # Load AAPL data with base features
    X_base, y, feature_names_base, df = load_real_market_data('AAPL', use_enhanced_features=False)
    
    # Load AAPL data with enhanced features
    X_enhanced, _, feature_names_enhanced, _ = load_real_market_data('AAPL', use_enhanced_features=True)
    
    # Temporal split
    X_train_base, X_test_base, y_train, y_test = temporal_train_test_split(df, X_base, y)
    X_train_enhanced, X_test_enhanced, _, _ = temporal_train_test_split(df, X_enhanced, y)
    
    # Train models with BASE features
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT 1: BASE FEATURES (8 features)")
    print(f"{'#'*80}")
    results_base = train_all_models(X_train_base, X_test_base, y_train, y_test, 'base')
    
    # Train models with ENHANCED features
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT 2: ENHANCED FEATURES ({len(feature_names_enhanced)} features)")
    print(f"{'#'*80}")
    results_enhanced = train_all_models(X_train_enhanced, X_test_enhanced, y_train, y_test, 'enhanced')
    
    # Save results
    all_results = {
        'dataset': 'AAPL_real_market',
        'n_samples': len(X_base),
        'n_train': len(X_train_base),
        'n_test': len(X_test_base),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        },
        'base_features': {
            'feature_names': feature_names_base,
            'n_features': len(feature_names_base),
            'results': results_base
        },
        'enhanced_features': {
            'feature_names': feature_names_enhanced,
            'n_features': len(feature_names_enhanced),
            'results': results_enhanced
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = 'real_market_results_aapl.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[COMPLETE] All experiments finished")
    print(f"{'='*80}")
    print(f"\nResults Summary:")
    print(f"\n   BASE FEATURES (8 features):")
    for model, metrics in results_base.items():
        print(f"      {model:>25s}: {metrics['test_accuracy']*100:>5.1f}% acc, {metrics['test_f1_macro']*100:>5.1f}% F1")
    
    print(f"\n   ENHANCED FEATURES ({len(feature_names_enhanced)} features):")
    for model, metrics in results_enhanced.items():
        print(f"      {model:>25s}: {metrics['test_accuracy']*100:>5.1f}% acc, {metrics['test_f1_macro']*100:>5.1f}% F1")
    
    print(f"\n[SAVED] Results saved to: {output_file}")


if __name__ == '__main__':
    main()
