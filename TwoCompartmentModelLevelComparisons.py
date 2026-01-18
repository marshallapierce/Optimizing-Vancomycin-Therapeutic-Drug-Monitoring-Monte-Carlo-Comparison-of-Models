import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt  # optional for plotting
import csv
import os

file_suffix = input("Enter a suffix for the CSV files (e.g., 'test1'): ")

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)  # For repeatable random values

# Population parameters for two-compartment model
pop_params = {'Vc': 58.4, 'Vp': 38.4, 'Cl_total': 4.5, 'Cl_dist': 6.5}#mean values
cvs = {'Vc': 0.3, 'Vp': 0.3, 'Cl_total': 0.3, 'Cl_dist': 0.4}#CVs for parameters
limits = {'Vc': (23, 93), 'Vp': (15, 61), 'Cl_total': (1.8, 7.2), 'Cl_dist': (1.3, 11.7)}# +/-2 SD Limits for parameter generation
fit_bounds = {'Vc': (5.8, 110), 'Vp': (3.8, 72), 'Cl_total': (0.45, 8.55), 'Cl_dist': (1.3, 11.7)}# +/-3 SD Bounds for parameter fitting

# original code
# times_true = np.array([0, 1, 2, 3, 4, 8, 9, 10])# User entered, levels in grids.cvs
# level_scenarios = [                             #this will need to match the level scenarios in grids.csv
#     ([0], "0"),
#     ([0, 1], "0,1"),
#     ([0, 1, 2, 3], "0,1,2,3"),
#     ([0, 1, 2, 3, 4, 8], "0,1,2,3,4,8"),
#     ([0, 1, 2, 3, 4, 8, 9, 10], "0,1,2,3,4,8,9,10"),
#     ([10], "10"),
#     ([9, 10], "9,10"),
#     ([4, 8, 9, 10], "4,8,9,10"),
#     ([2, 3, 4, 8, 9, 10], "2,3,4,8,9,10")
# ]  # Different level scenarios compared in cl_diff grid


N = 30000 # Number of simulated groups
# Time points for true and fitting concentrations post dose 
times_true = np.array([0, 1, 2, 3, 4, 8, 9, 10])# User entered, levels in grids.cvs change line 144 to match
# Original level_scenarios (commented out for reference):
# level_scenarios = [                             #this will need to match the level scenarios in grids.csv
#     ([0], "0"),
#     ([0, 1], "0,1"),
#     ([0, 1, 2, 3], "0,1,2,3"),
#     ([0, 1, 2, 3, 4, 8], "0,1,2,3,4,8"),
#     ([0, 1, 2, 3, 4, 8, 9, 10], "0,1,2,3,4,8,9,10"),
#     ([0, 1], "2,10"),
#     ([10], "10"),
#     ([9, 10], "9,10"),
#     ([4, 8, 9, 10], "4,8,9,10"),
#     ([2, 3, 4, 8, 9, 10], "2,3,4,8,9,10")
# ]  # Different level scenarios compared in cl_diff grid
level_scenarios = [   #this will need to match the level scenarios in grids.csv note these are indices for times_true 
    ([2], "2"),
    ([0, 1, 2, 3, 4, 5, 6, 7], "0,1,2,3,4,8,9,10"),
    ([2, 4, 5, 7], "2,4,8,10"),
    ([2, 7], "2,10"),
    ([7], "10")
]  # Different level scenarios compared in cl_diff grid

# Generate random parameters for true values
params = {}
for key in pop_params:
    mean = pop_params[key]
    sd = mean * cvs[key]
    raw = np.random.normal(mean, sd, N)
    params[key] = np.clip(raw, limits[key][0], limits[key][1])

# Generate randomized starting parameters for fitting
start_params = {}
for key in pop_params:
    mean = pop_params[key]
    sd = mean * cvs[key]
    raw = np.random.normal(mean, sd, N)
    start_params[key] = np.clip(raw, limits[key][0], limits[key][1])

# Function to calculate Cp
def calculate_cp(Vc, Vp, Cl_total, Cl_dist, times):
    K10 = Cl_total / Vc
    K12 = Cl_dist / Vc
    K21 = Cl_dist / Vp
    sum_k = K10 + K12 + K21
    disc = np.sqrt(sum_k**2 - 4 * K21 * K10)
    alpha = 0.5 * (sum_k + disc)
    beta = 0.5 * (sum_k - disc)
    #1012.6 is dose administered need to change dose to keep AUC consistent
    #12 is the dosing interval in hours
    #2 is the infusion time in hours
    A = 1125 * (K21 - alpha) * (1 - np.exp(-alpha * 2)) / (2 * Vc * alpha * (beta - alpha) * (1 - np.exp(-alpha * 12)))
    B = 1125 * (beta - K21) * (1 - np.exp(-beta * 2)) / (2 * Vc * beta * (beta - alpha) * (1 - np.exp(-beta * 12)))
    Cp = A * np.exp(-alpha * times) + B * np.exp(-beta * times)
    return Cp

# Calculate true levels
true_levels = np.zeros((N, len(times_true)))
for i in range(N):
    true_levels[i] = calculate_cp(params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i], times_true)

# Calculate AUC_true Adjust dose to keep AUC consistent
AUC_true = 1125 * (24 / 12) / params['Cl_total']

# Add noise to get randomized levels
randomized_levels = true_levels * (1 + stats.norm.ppf(np.random.random((N, len(times_true))), 0, 0.1))

# Function for residuals in fitting
def residuals(p, Vc_pop, Vp_pop, Cl_total_pop, Cl_dist_pop, times, levels):
    Vc, Vp, Cl_total, Cl_dist = p
    cp_fit = calculate_cp(Vc, Vp, Cl_total, Cl_dist, times)
    res_levels = (levels - cp_fit) / (0.1 * levels)
    res_vc = (Vc - Vc_pop) / (0.3 * Vc_pop)
    res_vp = (Vp - Vp_pop) / (0.3 * Vp_pop)
    res_cl_total = (Cl_total - Cl_total_pop) / (0.3 * Cl_total_pop)
    res_cl_dist = (Cl_dist - Cl_dist_pop) / (0.4 * Cl_dist_pop)
    return np.concatenate([res_levels, [res_vc, res_vp, res_cl_total, res_cl_dist]])

# Initialize results for grid
grid_results = []
fitted_AUC = np.zeros((len(level_scenarios), N))

# Loop over different level scenarios
for idx, (times_fit_list, label) in enumerate(level_scenarios):
    times_fit = times_true[times_fit_list]
    indices = [np.where(times_true == t)[0][0] for t in times_fit]
    levels_for_fit = randomized_levels[:, indices]
    
    # Fit for each group
    fitted_params = {'Vc': np.zeros(N), 'Vp': np.zeros(N), 'Cl_total': np.zeros(N), 'Cl_dist': np.zeros(N)}
    fitted_sse = np.zeros(N)
    for i in range(N):
        x0 = [start_params['Vc'][i], start_params['Vp'][i], start_params['Cl_total'][i], start_params['Cl_dist'][i]]
        bounds = ([fit_bounds['Vc'][0], fit_bounds['Vp'][0], fit_bounds['Cl_total'][0], fit_bounds['Cl_dist'][0]],
                  [fit_bounds['Vc'][1], fit_bounds['Vp'][1], fit_bounds['Cl_total'][1], fit_bounds['Cl_dist'][1]])
        res = optimize.least_squares(residuals, x0, bounds=bounds,
                                     args=(pop_params['Vc'], pop_params['Vp'], pop_params['Cl_total'], pop_params['Cl_dist'],
                                           times_fit, levels_for_fit[i]))
        fitted_params['Vc'][i] = res.x[0]
        fitted_params['Vp'][i] = res.x[1]
        fitted_params['Cl_total'][i] = res.x[2]
        fitted_params['Cl_dist'][i] = res.x[3]
        fitted_sse[i] = 2 * res.cost  # SSE = sum of squared residuals
    
    # Calculate Cl_diff and Cl_diff_sq
    cl_diff = params['Cl_total'] - fitted_params['Cl_total']
    cl_diff_sq = cl_diff ** 2
    
    # Calculate averages
    mean_cl_diff = np.mean(cl_diff)
    rmse = np.sqrt(np.mean(cl_diff_sq))
    
    # Calculate AUC_fit and AUC_diff
    AUC_fit = 1125 * (24 / 12) / fitted_params['Cl_total']
    fitted_AUC[idx] = AUC_fit
    auc_diff = AUC_fit - AUC_true
    auc_diff_sq = auc_diff ** 2
    
    # Calculate AUC metrics
    mean_auc_diff = np.mean(auc_diff)
    auc_std = np.std(auc_diff)
    auc_rmse = np.sqrt(np.mean(auc_diff_sq))
    
    # Calculate 95% CI for calculated AUC (Bland-Altman style)
    mean_auc = np.mean(AUC_fit)
    std_auc = np.std(AUC_fit)
    auc_lower = mean_auc_diff - 1.96 * auc_std
    auc_upper = mean_auc_diff + 1.96 * auc_std
    
    # Calculate percent error metrics
    percent_error = ((AUC_fit - AUC_true) / AUC_true) * 100
    mean_percent_error = np.mean(percent_error)
    mape = np.mean(np.abs(percent_error))
    rmspe = np.sqrt(np.mean(percent_error**2))
    std_percent_error = np.std(percent_error)
    
    # Calculate Pearson's r for AUC_fit vs AUC_true
    r, _ = stats.pearsonr(AUC_true, AUC_fit)
    
    # Append to grid results
    grid_results.append([label, mean_cl_diff, rmse, mean_auc_diff, auc_std, auc_rmse, auc_lower, auc_upper, 
                         mean_percent_error, mape, rmspe, std_percent_error, r])
    
    # For the full early levels scenario, save the detailed CSV and grids, Input time determines what prints in grids_two_vrs_two csv file
    #no spaces between commas in label
    if label == "10": #"10": #"2,10": #"0,1,2,3,4,8,9,10": #"2,4,8,10":Cl # needs to match the scenario for the times wanted in grid (trough, Pk/tr, or level times 2, 4, 8, 10 ect)
        # Calculate fitted levels
        fitted_levels = np.zeros((N, len(times_fit)))
        for i in range(N):
            fitted_levels[i] = calculate_cp(fitted_params['Vc'][i], fitted_params['Vp'][i], fitted_params['Cl_total'][i], fitted_params['Cl_dist'][i], times_fit)
        
        # Calculate AUC_fit Adjust dose to keep AUC consistent
        AUC_fit = 1125 * (24 / 12) / fitted_params['Cl_total']
        
        # Save AUC data for combined plot
        with open(f'{output_dir}/auc_data_two_compartment{file_suffix}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Method', 'AUC_true', 'AUC_pred'])
            for i in range(N):
                writer.writerow(['Two-Compartment', AUC_true[i], AUC_fit[i]])
        
        print(f"AUC data saved to auc_data_two_compartment{file_suffix}.csv")
        
        # Save AUC and Cl data for combined plot
        with open(f'{output_dir}/auc_cl_data_two_compartment{file_suffix}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Method', 'AUC_true', 'AUC_pred', 'Cl_true', 'Cl_pred'])
            for i in range(N):
                writer.writerow(['Two-Compartment Bayesian', AUC_true[i], AUC_fit[i], params['Cl_total'][i], fitted_params['Cl_total'][i]])
        
        print(f"AUC and Cl data saved to auc_cl_data_two_compartment{file_suffix}.csv")
        percent_error = ((AUC_fit - AUC_true) / AUC_true) * 100
        
        # Create the comparison grid
        bins = [0,  400,  601, np.inf]
        custom_labels = ["< 400", "400 through 600", "> 600"]
        hist, xedges, yedges = np.histogram2d(AUC_true, AUC_fit, bins=[bins, bins])
        
        # Calculate fraction of correct predictions
        bin_indices_true = np.digitize(AUC_true, bins) - 1
        bin_indices_fit = np.digitize(AUC_fit, bins) - 1
        correct_count = np.sum(bin_indices_true == bin_indices_fit)
        fraction_correct = correct_count / N
        
        print("Actual AUC Ranges\tNumber Fit in AUC Ranges Below")
        print("Number In AUC ranges\tTotal\t< 400\t400 through 600\t> 600")
        for i in range(len(bins)-1):
            total = int(np.sum(hist[i]))
            row_label = custom_labels[i]
            print(f"{row_label}\t{total}\t{'\t'.join(map(str, hist[i].astype(int)))}")
        
        # Percentage grid
        print("\nPercentage of Fit AUCs in Each Range")
        print("Actual AUC Ranges\tPercentage Fit in AUC Ranges Below")
        print("Number In AUC ranges\tTotal\t< 400\t400 through 600\t> 600")
        for i in range(len(bins)-1):
            total = np.sum(hist[i])
            if total > 0:
                percs = hist[i] / total * 100
            else:
                percs = np.zeros_like(hist[i])
            row_label = custom_labels[i]
            print(f"{row_label}\t{total:.0f}\t{'\t'.join(map(lambda x: f'{x:.1f}%', percs))}")
        
        print(f"\nFraction of Correct Predictions: {fraction_correct:.4f}")
        
        # Write results to CSV
        with open(f'{output_dir}/monte_carlo_results_two_vrs_two{file_suffix}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Group', 'Vc_true', 'Vp_true', 'Cl_total_true', 'Cl_dist_true'] + [f'Cp_{t}_true' for t in times_true] + ['AUC_true'] + [f'Cp_{t}_rand' for t in times_true] + ['Vc_start', 'Vp_start', 'Cl_total_start', 'Cl_dist_start'] + [f'Cp_{t}_fit' for t in times_fit] + ['Vc_fit', 'Vp_fit', 'Cl_total_fit', 'Cl_dist_fit', 'AUC_fit', 'SSE', 'Cl_diff', 'Cl_diff_sq', 'Percent_Error']
            writer.writerow(header)
            for i in range(N):
                row = [i+1, params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i]] + list(true_levels[i]) + [AUC_true[i]] + list(randomized_levels[i]) + [start_params['Vc'][i], start_params['Vp'][i], start_params['Cl_total'][i], start_params['Cl_dist'][i]] + list(fitted_levels[i]) + [fitted_params['Vc'][i], fitted_params['Vp'][i], fitted_params['Cl_total'][i], fitted_params['Cl_dist'][i], AUC_fit[i], fitted_sse[i], cl_diff[i], cl_diff_sq[i], percent_error[i]]
                writer.writerow(row)
        
        print(f"Results saved to monte_carlo_results_two_vrs_two{file_suffix}.csv")
        
        # Write grids to separate CSV
        with open(f'{output_dir}/grids_two_vrs_two{file_suffix}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Comparison Grid'])
            writer.writerow(['Actual AUC Ranges', 'Number Fit in AUC Ranges Below'])
            writer.writerow(['Number In AUC ranges', 'Total', '< 400', '400 through 600', '> 600'])
            for i in range(len(bins)-1):
                total = int(np.sum(hist[i]))
                row_label = custom_labels[i]
                writer.writerow([row_label, total] + list(hist[i].astype(int)))
            
            writer.writerow([])
            writer.writerow(['Percentage Grid'])
            writer.writerow(['Actual AUC Ranges', 'Percentage Fit in AUC Ranges Below'])
            writer.writerow(['Number In AUC ranges', 'Total', '< 400', '400 through 600', '> 600'])
            for i in range(len(bins)-1):
                total = float(np.sum(hist[i]))
                if total > 0:
                    percs = hist[i] / total * 100
                else:
                    percs = np.zeros_like(hist[i])
                row_label = custom_labels[i]
                writer.writerow([row_label, f"{total:.0f}"] + [f"{x:.1f}%" for x in percs])
        
            writer.writerow([])
            writer.writerow(['Fraction of Correct Predictions', '', '', '', '', f"{fraction_correct:.4f}"])
        
        # Calculate Pearson r
        r, _ = stats.pearsonr(AUC_true, AUC_fit)
        
        # Plot residuals
        plt.figure(figsize=(8,6))
        plt.scatter(AUC_true, auc_diff, alpha=0.5, s=5)# s= size of points
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Actual AUC')
        plt.ylabel('Predicted AUC - Actual AUC')
        plt.title('Two Compt. Bayesian Model Residual vs Actual AUC')
        plt.text(0.05, 0.95, f'RMSE: {auc_rmse:.2f}\nPearson r: {r:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.savefig(f'2cAUC_residuals_{file_suffix}.pdf')
        plt.close()
        
        print(f"Grids saved to grids_two_vrs_two{file_suffix}.csv")
        
# Save the Cl_diff grid to separate CSV
with open(f'{output_dir}/cl_diff_grid_two_vrs_two{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Levels', 'Average Cl_diff', 'RMSE', 'Average AUC_diff', 'AUC_diff SD', 'AUC_diff RMSE', 'AUC Lower 95% CI', 'AUC Upper 95% CI', 
                     'Mean Percent Error(average bias)', 'MAPE(average absolute percenterror)', 'RMSPE(more weight to larger errors)', 'Percent Error SD(spread around mean)', "Pearson's r"])
    for row in grid_results:
        writer.writerow(row)

print(f"Cl_diff grid saved to cl_diff_grid_two_vrs_two{file_suffix}.csv")

# Bland-Altman analysis
scenario_names = [scenario[1] for scenario in level_scenarios]
auc_fits = fitted_AUC  # shape (len(level_scenarios), N)
auc_true = AUC_true  # shape (N,)

for i, scenario_name in enumerate(scenario_names):
    auc_fit = auc_fits[i]
    auc_differences = auc_true - auc_fit
    auc_means = (auc_true + auc_fit) / 2
    with open(f'{output_dir}/bland_altman_data_{scenario_name}{file_suffix}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['AUC_mean', 'AUC_difference'])
        for mean, diff in zip(auc_means, auc_differences):
            writer.writerow([mean, diff])

# Plot Bland-Altman for each scenario
for scenario_name in scenario_names:
    with open(f'{output_dir}/bland_altman_data_{scenario_name}{file_suffix}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        data = list(reader)
        auc_means = [float(row[0]) for row in data]
        auc_diffs = [float(row[1]) for row in data]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(auc_means, auc_diffs, alpha=0.5)
    plt.axhline(np.mean(auc_diffs), color='red', label='Mean Difference')
    plt.axhline(np.mean(auc_diffs) + 1.96 * np.std(auc_diffs), color='blue', linestyle='--', label='+1.96 SD')
    plt.axhline(np.mean(auc_diffs) - 1.96 * np.std(auc_diffs), color='blue', linestyle='--', label='-1.96 SD')
    plt.xlabel('Mean AUC')
    plt.ylabel('AUC Difference')
    plt.title(f'Bland-Altman Plot for {scenario_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/bland_altman_plot_{scenario_name}{file_suffix}.png', dpi=300)
    plt.close()