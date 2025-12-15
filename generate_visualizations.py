#!/usr/bin/env python3
"""
Quick Visualization Generator
Uses existing JSON results to create performance graphs
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load existing results from JSON files"""
    results_path = Path("/Users/julih/Documents/CS229_QuantML_Project/outputs/week3_4")
    
    # Load fast SVM results (most recent/accurate)
    fast_results_path = results_path / "fast_svm_results.json"
    with open(fast_results_path, 'r') as f:
        fast_results = json.load(f)
    
    # Load comparison results 
    comparison_path = results_path / "kpca_svm_comparison.json"
    with open(comparison_path, 'r') as f:
        comparison_results = json.load(f)
    
    return fast_results, comparison_results

def create_performance_comparison_plot(fast_results, comparison_results):
    """Create accuracy vs F1 performance comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Kernel PCA + SVM Performance Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data from fast_results (most accurate)
    models_data = []
    for result in fast_results:
        models_data.append({
            'kernel': result['kernel'].title(),
            'accuracy': result['metrics']['accuracy'] * 100,  # Convert to percentage
            'f1_macro': result['metrics']['f1_macro'] * 100,
            'auc_macro': result['metrics']['auc_macro_ovr'] * 100,
            'C': result['best_params']['svc__C'],
            'gamma': result['best_params']['svc__gamma']
        })
    
    df = pd.DataFrame(models_data)
    
    # Plot 1: Accuracy vs F1 Score
    ax1 = axes[0, 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    scatter = ax1.scatter(df['accuracy'], df['f1_macro'], 
                         c=colors[:len(df)], s=150, alpha=0.7, edgecolors='black')
    
    # Add labels for each point
    for i, row in df.iterrows():
        ax1.annotate(row['kernel'], 
                    (row['accuracy'], row['f1_macro']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_ylabel('F1-Macro Score (%)', fontsize=12)
    ax1.set_title('Accuracy vs F1-Macro Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(69, 73)
    ax1.set_ylim(66, 70)
    
    # Plot 2: Performance Ranking Bar Chart
    ax2 = axes[0, 1]
    df_sorted = df.sort_values('f1_macro', ascending=True)
    bars = ax2.barh(range(len(df_sorted)), df_sorted['f1_macro'], 
                    color=colors[:len(df_sorted)], alpha=0.8)
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['kernel'], fontsize=12)
    ax2.set_xlabel('F1-Macro Score (%)', fontsize=12)
    ax2.set_title('Kernel PCA Performance Ranking', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted['f1_macro'])):
        ax2.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    # Plot 3: AUC Performance
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['kernel'], df['auc_macro'], 
                    color=colors[:len(df)], alpha=0.8, edgecolor='black')
    ax3.set_ylabel('AUC-Macro (%)', fontsize=12)
    ax3.set_title('AUC-Macro Performance by Kernel', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars3, df['auc_macro']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Hyperparameter Analysis
    ax4 = axes[1, 1]
    # Create a table of best hyperparameters
    table_data = []
    for _, row in df.iterrows():
        table_data.append([row['kernel'], f"C={row['C']}", f"γ={row['gamma']}", f"{row['f1_macro']:.1f}%"])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Kernel', 'C', 'γ', 'F1-Score'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    ax4.axis('off')
    ax4.set_title('Optimal Hyperparameters', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/Users/julih/Documents/CS229_QuantML_Project/evaluation/figures/performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] Performance comparison plot: {output_path}")
    
    return output_path

def create_confusion_matrix_analysis(fast_results):
    """Create confusion matrix visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(fast_results):
        ax = axes[i//2, i%2]
        
        # Get confusion matrix
        cm = np.array(result['confusion_matrix'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{result["kernel"].title()} KPCA + RBF SVM', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_xticklabels(['Underpriced (-1)', 'Fair (0)', 'Overpriced (+1)'])
        ax.set_yticklabels(['Underpriced (-1)', 'Fair (0)', 'Overpriced (+1)'])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/Users/julih/Documents/CS229_QuantML_Project/evaluation/figures/confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] Confusion matrix analysis: {output_path}")
    
    return output_path

def create_summary_table():
    """Create a summary table of results"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Summary data from our accurate results (including actual baselines)
    summary_data = [
        ['Sigmoid KPCA + RBF SVM', '71.9%', '68.8%', '84.0%', 'C=10, γ=scale'],
        ['Linear KPCA + RBF SVM', '71.6%', '68.4%', '84.2%', 'C=10, γ=0.1'],
        ['RBF KPCA + RBF SVM', '70.9%', '67.7%', '83.3%', 'C=10, γ=scale'],
        ['Random Forest (baseline)', '70.8%', '68.4%', '85.2%', 'Original features'],
        ['Logistic Regression (baseline)', '62.8%', '61.7%', '75.0%', 'Original features'],
        ['OLS Residual (baseline)', '58.0%', '58.3%', '70.5%', 'Original features']
    ]
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Model Configuration', 'Accuracy', 'F1-Macro', 'AUC-Macro', 'Best Parameters'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Style the table
    table[(0, 0)].set_facecolor('#E8F4FD')
    table[(0, 1)].set_facecolor('#E8F4FD')
    table[(0, 2)].set_facecolor('#E8F4FD')
    table[(0, 3)].set_facecolor('#E8F4FD')
    table[(0, 4)].set_facecolor('#E8F4FD')
    
    plt.title('Kernel PCA + SVM: Final Performance Summary', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save the plot
    output_path = "/Users/julih/Documents/CS229_QuantML_Project/evaluation/figures/performance_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] Performance summary table: {output_path}")
    
    return output_path

def main():
    """Generate all visualizations"""
    print("[GENERATING] Visualizations from existing results...")
    
    # Create figures directory
    figures_dir = Path("/Users/julih/Documents/CS229_QuantML_Project/evaluation/figures")
    figures_dir.mkdir(exist_ok=True)
    
    # Load results
    fast_results, comparison_results = load_results()
    print(f"[LOADED] {len(fast_results)} model results")
    
    # Generate plots
    plots_created = []
    
    # 1. Performance comparison
    plot1 = create_performance_comparison_plot(fast_results, comparison_results)
    plots_created.append(plot1)
    
    # 2. Confusion matrices
    plot2 = create_confusion_matrix_analysis(fast_results)
    plots_created.append(plot2)
    
    # 3. Summary table
    plot3 = create_summary_table()
    plots_created.append(plot3)
    
    print(f"\n[SUCCESS] Generated {len(plots_created)} visualizations")
    print("Files created:")
    for plot in plots_created:
        print(f"   - {plot}")
    
    print("\nKey Insights from Generated Plots:")
    print("   - Sigmoid KPCA achieved highest accuracy (71.9%)")
    print("   - Linear KPCA surprisingly competitive (71.6%)")
    print("   - All models exceed 83% AUC-macro (strong discrimination)")
    print("   - C=10 consistently optimal across kernels")
    
    return plots_created

if __name__ == "__main__":
    main()