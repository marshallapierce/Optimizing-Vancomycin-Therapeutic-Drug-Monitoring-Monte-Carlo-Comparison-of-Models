import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# File suffixes (adjust if needed)
suffix = "CltotalfourpointfivePKTR"  # or "_test" etc., based on what you used

# Read data from two compartment file
auc_data_two = []
if os.path.exists(f'auc_data_two_compartment{suffix}.csv'):
    with open(f'auc_data_two_compartment{suffix}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            auc_data_two.append([row[0], float(row[1]), float(row[2])])

# Read data from one compartment file
auc_data_one = []
if os.path.exists(f'auc_data_one_compartment{suffix}.csv'):
    with open(f'auc_data_one_compartment{suffix}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            auc_data_one.append([row[0], float(row[1]), float(row[2])])

# Combine all data
all_data = auc_data_two + auc_data_one

# Collect all AUC_true for SD calculation
all_auc_true = [row[1] for row in all_data]
sd_auc_true = np.std(all_auc_true)

# Prepare data for plotting
methods = {}
for method, true, pred in all_data:
    if method not in methods:
        methods[method] = {'x': [], 'y': []}
    diff = pred - true
    methods[method]['x'].append(diff / sd_auc_true)
    methods[method]['y'].append(diff)

# Calculate bias and RMSE for each method
method_stats = {}
for method, data in methods.items():
    diffs = data['y']
    bias = np.mean(diffs)
    rmse = np.sqrt(np.mean([d**2 for d in diffs]))
    method_stats[method] = {'bias': bias, 'rmse': rmse}

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid for 4 methods
axes = axes.flatten()
colors = ['blue', 'red', 'green', 'orange']
method_names = list(methods.keys())

# Mapping for new titles
title_map = {
    'Two-Compartment': 'Two-Compartment Bayesian',
    'Fixed Vd': 'One-Compartment Fixed-VD',
    'Peak Trough': 'One-Compartment Peak Trough',
    'Bayesian': 'One-Compartment Bayesian'
}

for i, (method, data) in enumerate(methods.items()):
    ax = axes[i]
    ax.scatter(data['x'], data['y'], alpha=0.5, s=5, color=colors[i])
    ax.axhline(y=0, color='black', linestyle='--')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel('(AUC Predicted - AUC True) / SD(AUC True)')
    ax.set_ylabel('AUC Predicted - AUC True')
    ax.set_title(title_map.get(method, method))  # Use mapped title or original if not found
    ax.grid(True)
    
    # Add bias and RMSE text
    bias = method_stats[method]['bias']
    rmse = method_stats[method]['rmse']
    ax.text(0.05, 0.95, f'Bias: {bias:.2f}\nRMSE: {rmse:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'combined_accuracy_plot{suffix}.pdf')
plt.show()

print(f"Plot saved to combined_accuracy_plot{suffix}.pdf")