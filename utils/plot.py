import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import glob

# Find and read all relevant CSV files
dfs = []
for m in ['vit', 'performer']:
    for v in ['mixed', 'axial']:
        files = glob.glob(f'training_metrics_{m}_{v}*.csv')
        for file in files:
            df = pd.read_csv(file)
            df['model'] = m
            df['variant'] = v
            dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

plt.figure(figsize=(12, 8))

styles = {
    'vit': {'linestyle': '-', 'color': ['#1f77b4', '#ff7f0e']},
    'performer': {'linestyle': '--', 'color': ['#2ca02c', '#d62728']}
}

# Plot test accuracy for all combinations
for i, model in enumerate(['vit', 'performer']):
    for j, variant in enumerate(['mixed', 'axial']):
        data = combined_df[(combined_df['model'] == model) & 
                          (combined_df['variant'] == variant)]
        if not data.empty:
            plt.plot(data['epoch'], 
                    data['test_accuracy'],
                    label=f'{model.upper()} - {variant}',
                    linestyle=styles[model]['linestyle'],
                    color=styles[model]['color'][j],
                    linewidth=2)

plt.title('Test Accuracy Comparison Across Models and Rope Variants', 
          fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12, labelpad=10)
plt.ylabel('Accuracy (%)', fontsize=12, labelpad=10)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.xlim(-5, 105)
plt.gca().xaxis.set_major_locator(MultipleLocator(10))

plt.tight_layout()
plt.show()

# save as svg
plt.savefig('test_accuracy_comparison.svg', format='svg')

print("\nPerformance Summary:")
print("=" * 50)
for model in ['vit', 'performer']:
    for variant in ['mixed', 'axial']:
        data = combined_df[(combined_df['model'] == model) & 
                          (combined_df['variant'] == variant)]
        if not data.empty:
            print(f"\n{model.upper()} - {variant}:")
            print(f"Best Test Accuracy: {data['test_accuracy'].max():.2f}%")
            print(f"Final Test Accuracy: {data['test_accuracy'].iloc[-1]:.2f}%")