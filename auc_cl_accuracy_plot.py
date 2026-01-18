import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy import stats

# File suffixes (adjust if needed)
suffix = "CltotalfourpointfivePKTR"  # or "_test" etc., based on what you used

# Read data from two compartment file
auc_cl_data_two = []
if os.path.exists(f'auc_cl_data_two_compartment{suffix}.csv'):
    with open(f'auc_cl_data_two_compartment{suffix}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            auc_cl_data_two.append([row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])])

# Read data from one compartment file
auc_cl_data_one = []
if os.path.exists(f'auc_cl_data_one_compartment{suffix}.csv'):
    with open(f'auc_cl_data_one_compartment{suffix}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            auc_cl_data_one.append([row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])])

# Combine all data
all_data = auc_cl_data_two + auc_cl_data_one

# Collect all Cl_true for SD calculation
all_cl_true = [row[3] for row in all_data]
sd_cl_true = np.std(all_cl_true)

# Prepare data for plotting
methods = {}
for method, auc_true, auc_pred, cl_true, cl_pred in all_data:
    if method not in methods:
        methods[method] = {'x': [], 'y': []}
    auc_diff = auc_pred - auc_true
    cl_diff = cl_pred - cl_true
    methods[method]['x'].append(cl_diff / sd_cl_true)
    methods[method]['y'].append(auc_diff)

# Calculate bias and RMSE for each method (for AUC)
method_stats = {}
for method, data in methods.items():
    auc_diffs = data['y']
    bias = np.mean(auc_diffs)
    rmse = np.sqrt(np.mean([d**2 for d in auc_diffs]))
    method_stats[method] = {'bias': bias, 'rmse': rmse}

# Calculate overall x and y limits
all_x = []
all_y = []
for method in methods.values():
    all_x.extend(method['x'])
    all_y.extend(method['y'])
x_min = min(all_x)
x_max = max(all_x)
y_min = min(all_y)
y_max = max(all_y)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid for 4 methods
axes = axes.flatten()
colors = ['black', 'black', 'black', 'black']#['blue', 'red', 'green', 'orange']
method_names = list(methods.keys())

# Mapping for new titles
title_map = {
    'Two-Compartment Bayesian': 'Two-Compartment Bayesian',
    'One-Compartment Fixed-VD': 'One-Compartment Fixed-VD',
    'One-Compartment Peak Trough': 'One-Compartment Peak Trough',
    'One-Compartment Bayesian': 'One-Compartment Bayesian'
}

for i, method in enumerate(method_names):
    ax = axes[i]
    data = methods[method]
    ax.scatter(data['x'], data['y'], alpha=0.5, s=1, color=colors[i])#s=5)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.axvline(x=0, color='black', linestyle='--')
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['x'], data['y'])
    x_line = np.array([x_min, x_max])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression (r={r_value:.2f})')
    
    ax.set_xlabel('(Cl Predicted - Cl True) / SD(Cl True)')
    ax.set_ylabel('AUC Predicted - AUC True')
    ax.set_title(title_map.get(method, method))  # Use mapped title or original if not found
    ax.set_xlim(x_min, x_max)  # Set same x limits for all
    ax.set_ylim(y_min, y_max)  # Set same y limits for all
    ax.grid(True)
    
    # Add bias and RMSE text
    bias = method_stats[method]['bias']
    rmse = method_stats[method]['rmse']
    ax.text(0.05, 0.95, f'Bias: {bias:.2f}\nRMSE: {rmse:.2f}\nCorr: {r_value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'auc_cl_accuracy_plot{suffix}.pdf')
plt.show()

print(f"Plot saved to auc_cl_accuracy_plot{suffix}.pdf")