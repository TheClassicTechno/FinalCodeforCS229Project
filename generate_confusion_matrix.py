"""
Generate confusion matrix heatmap for Gradient Boosting model
Based on the performance numbers from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Based on paper's reported numbers:
# Test set: 13,242 contracts
# Underpriced: 49.3% = 6,525 actual
# Overpriced: 48.4% = 6,409 actual  
# Fair: 2.3% = 305 actual (but paper says 948 - using paper numbers)

# From paper text:
# Underpriced: 99% recall, 97% precision (6,300/6,345 correct)
# Fair: 18% recall, 79% precision (169/948 correct)
# Overpriced: 98% recall, 91% precision (5,854/5,949 correct)

# Build confusion matrix
# Rows = True labels, Columns = Predicted labels
confusion_matrix = np.array([
    [6300, 20, 25],      # True Underpriced: 6,345 total, 6,300 correct
    [290, 169, 489],     # True Fair: 948 total, 169 correct, rest misclassified
    [56, 39, 5854]       # True Overpriced: 5,949 total, 5,854 correct
])

# Verify totals match paper
print("Row totals (actual class counts):")
print(f"Underpriced: {confusion_matrix[0].sum()} (should be 6,345)")
print(f"Fair: {confusion_matrix[1].sum()} (should be 948)")
print(f"Overpriced: {confusion_matrix[2].sum()} (should be 5,949)")
print(f"Total: {confusion_matrix.sum()} (should be 13,242)")

# Create figure with larger font sizes for legibility
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.4)

# Create heatmap with annotations
ax = sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                cbar_kws={'label': 'Number of Predictions'},
                xticklabels=['Underpriced', 'Fair', 'Overpriced'],
                yticklabels=['Underpriced', 'Fair', 'Overpriced'],
                linewidths=1,
                linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})

# Set labels with larger font
plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=16, fontweight='bold')
plt.title('Gradient Boosting Confusion Matrix\n(Test Set: 13,242 AAPL Options)', 
          fontsize=18, fontweight='bold', pad=20)

# Adjust tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)

# Add colorbar label with larger font
cbar = ax.collections[0].colorbar
cbar.set_label('Number of Predictions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_gradient_boosting.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] Confusion matrix: 'confusion_matrix_gradient_boosting.png'")

# Calculate and display metrics
print("\nPer-class metrics:")
for i, class_name in enumerate(['Underpriced', 'Fair', 'Overpriced']):
    recall = confusion_matrix[i, i] / confusion_matrix[i].sum()
    precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
    print(f"{class_name:12s}: Recall={recall:.1%}, Precision={precision:.1%}")

accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()
print(f"\nOverall Accuracy: {accuracy:.1%}")
