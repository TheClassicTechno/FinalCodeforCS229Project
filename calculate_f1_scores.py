#!/usr/bin/env python3
"""
Calculate F1 scores for all models to verify paper claims
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the enhanced options data"""
    data_path = Path(__file__).parent / 'svm' / 'data' / 'enhanced_options_data.csv'
    df = pd.read_csv(data_path)
    
    # Separate features and target
    target_col = 'label_uf_over'
    
    # Drop non-feature columns
    drop_cols = ['date', 'ticker', 'option_type', 'S', 'K', 'bs_price', 'mkt_price', 
                 'residual', 'label_uf_over', 'fwd_option_return']
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    y = df[target_col].values
    X = df.drop(columns=drop_cols).values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print(f"Class counts: {[(val, np.sum(y==val)) for val in np.unique(y)]}")
    
    # Convert labels from {-1, 0, 1} to {0, 1, 2} for sklearn
    y = y + 1
    
    return X, y

def calculate_f1_with_cv(model, X_train, y_train, X_test, y_test, model_name):
    """Calculate F1 scores with cross-validation"""
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred_test = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    # Cross-validation F1 scores
    cv_f1_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    cv_f1_mean = cv_f1_scores.mean()
    cv_f1_std = cv_f1_scores.std()
    
    print(f"Test F1-macro:  {test_f1:.1%}")
    print(f"CV F1-macro:    {cv_f1_mean:.1%} ± {cv_f1_std:.1%}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, 
                               target_names=['Underpriced', 'Fair', 'Overpriced']))
    
    return {
        'test_f1_macro': test_f1,
        'cv_f1_mean': cv_f1_mean,
        'cv_f1_std': cv_f1_std
    }

def main():
    print("Loading enhanced feature data...")
    X, y = load_data()
    
    # Use the same split as in your paper (80/20)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define models (same hyperparameters as in paper)
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            random_state=42
        )
    }
    
    # Calculate F1 scores for each model
    results = {}
    for name, model in models.items():
        results[name] = calculate_f1_with_cv(
            model, X_train, y_train, X_test, y_test, name
        )
    
    # Calculate Voting Ensemble
    print(f"\n{'='*60}")
    print(f"Model: Voting Ensemble")
    print(f"{'='*60}")
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    lr = LogisticRegression(multi_class='multinomial', C=1.0, max_iter=1000, random_state=42)
    
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    
    results['Voting Ensemble'] = calculate_f1_with_cv(
        voting, X_train, y_train, X_test, y_test, 'Voting Ensemble'
    )
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY - F1-MACRO SCORES")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Test F1':<12} {'CV F1':<15}")
    print(f"{'-'*60}")
    for name, scores in results.items():
        test_f1 = scores['test_f1_macro']
        cv_f1 = scores['cv_f1_mean']
        cv_std = scores['cv_f1_std']
        print(f"{name:<25} {test_f1:>10.1%}  {cv_f1:>10.1%} ± {cv_std:.1%}")
    
    # Save results to JSON
    output_file = Path(__file__).parent / 'svm' / 'data' / 'f1_scores_verified.json'
    
    results_serializable = {
        name: {
            'test_f1_macro': float(scores['test_f1_macro']),
            'cv_f1_mean': float(scores['cv_f1_mean']),
            'cv_f1_std': float(scores['cv_f1_std'])
        }
        for name, scores in results.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Compare with paper claims
    print(f"\n{'='*60}")
    print("COMPARISON WITH PAPER CLAIMS")
    print(f"{'='*60}")
    
    paper_claims = {
        'Random Forest': 0.791,
        'Voting Ensemble': 0.789,
        'Gradient Boosting': 0.786,
        'Logistic Regression': 0.710
    }
    
    for name in results.keys():
        if name in paper_claims:
            paper_f1 = paper_claims[name]
            actual_f1 = results[name]['test_f1_macro']
            difference = actual_f1 - paper_f1
            status = "[MATCH]" if abs(difference) < 0.02 else "[DIFFERS]"
            print(f"{name:<25} Paper: {paper_f1:.1%}  Actual: {actual_f1:.1%}  Diff: {difference:+.1%}  {status}")

if __name__ == '__main__':
    main()
