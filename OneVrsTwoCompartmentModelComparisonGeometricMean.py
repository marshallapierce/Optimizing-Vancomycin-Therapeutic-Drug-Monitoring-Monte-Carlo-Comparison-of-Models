import numpy as np
from scipy import stats, optimize
import csv
import matplotlib.pyplot as plt
import os
# two compartment vs one compartment model comparisons 
np.random.seed(42)  # For repeatable random values

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Get the Python file name for output files
py_file_name = os.path.splitext(os.path.basename(__file__))

# Population parameters for two-compartment limits are used in parameter randomization
# adjusted Cl_total to reflect population mean from real data
# adjust all doses (9 occurrences) for Avg AUC target 450
weight = 110 # input a weight as desired in kg
Crcl = 30 # input a creatinine clearance as desired in mL/min
Vc_mean = 58.4 * (weight / 70)
Cl_mean = 4.5 * (Crcl / 120) ** 0.8
pop_params = {'Vc': Vc_mean, 'Vp': 38.4, 'Cl_total': Cl_mean, 'Cl_dist': 6.5}#Cl_total 4.5
dose = 500 * Cl_mean * (12 / 24)
cvs = {'Vc': 0.3, 'Vp': 0.3, 'Cl_total': 0.3, 'Cl_dist': 0.4}# coefficient of variation for parameters
lower_vc = Vc_mean - 2 * cvs['Vc'] * Vc_mean
upper_vc = Vc_mean + 2 * cvs['Vc'] * Vc_mean
lower_cl = Cl_mean - 2 * cvs['Cl_total'] * Cl_mean
upper_cl = Cl_mean + 2 * cvs['Cl_total'] * Cl_mean
limits = {'Vc': (lower_vc, upper_vc), 'Vp': (15, 61), 'Cl_total': (lower_cl, upper_cl), 'Cl_dist': (1.3, 11.7)}# +/- 2SD for random parameters'Cl_total':(1.8,7.2)

N = 30000 # Number of simulated subjects
# the times below are the acceptable times for levels to be entered in line 17.
times_true = np.array([0, 1, 2, 3, 4, 8, 9, 10])# User entered time levels for two compartment fiting, levels in grids.cvs

# User input for fitting times (up to 8)
try:
    times_fit_input = input("Bayesian Analysis One compartment model:Enter up to 8 times separated by space or comma: ").replace(',', ' ').split()
    times_fit = np.array([float(t) for t in times_fit_input[:8]])
    # Ensure times_fit are subset of times_true for simplicity
    times_fit = np.array([t for t in times_fit if t in times_true])
    if len(times_fit) == 0:
        times_fit = times_true
except:
    print("Using default times: all")
    times_fit = times_true

# User input for file suffix
try:
    file_suffix = input("Enter file suffix for CSV files (e.g., _v1, leave blank for none): ")
except:
    file_suffix = ""
    print("Using empty suffix")

# Time points for true and fitting concentrations post dose 
#times_true = np.array([10]) # 0, 1, 2, 3, 4, 8, 9, 10
#times_fit = times_true  # User can modify this list for fitting times

# Generate random parameters for true values (two-compartment)
params = {}
clipping_stats = {}
for key in pop_params:
    mean = pop_params[key]
    sd = mean * cvs[key]
    raw = np.random.normal(mean, sd, N)
    params[key] = np.clip(raw, limits[key][0], limits[key][1])
    clipped_lower = np.sum(raw < limits[key][0])
    clipped_upper = np.sum(raw > limits[key][1])
    total_clipped = clipped_lower + clipped_upper
    percent_clipped = total_clipped / N * 100
    clipping_stats[key] = percent_clipped

# Save step 2 to CSV
# with open('step2_params.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Group', 'Vc', 'Vp', 'Cl_total', 'Cl_dist'])
#     for i in range(N):
#         writer.writerow([i+1, params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i]])

# Function to calculate Cp for two-compartment Step 3
def calculate_cp_2comp(Vc, Vp, Cl_total, Cl_dist, times):
    K10 = Cl_total / Vc
    K12 = Cl_dist / Vc
    K21 = Cl_dist / Vp
    sum_k = K10 + K12 + K21
    disc = np.sqrt(sum_k**2 - 4 * K21 * K10)
    alpha = 0.5 * (sum_k + disc)
    beta = 0.5 * (sum_k - disc)
    A = dose * (K21 - alpha) * (1 - np.exp(-alpha * 2)) / (2 * Vc * alpha * (beta - alpha) * (1 - np.exp(-alpha * 12)))
    B = dose * (beta - K21) * (1 - np.exp(-beta * 2)) / (2 * Vc * beta * (beta - alpha) * (1 - np.exp(-beta * 12)))
    Cp = A * np.exp(-alpha * times) + B * np.exp(-beta * times)
    return Cp

# Calculate true levels (step 4)
true_levels = np.zeros((N, len(times_true)))
for i in range(N):
    true_levels[i] = calculate_cp_2comp(params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i], times_true)

# Save step 4 to CSV
# with open('step4_true_levels.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     header = ['Group'] + [f'Cp_{t}_true' for t in times_true]
#     writer.writerow(header)
#     for i in range(N):
#         writer.writerow([i+1] + list(true_levels[i]))

# Calculate AUC_true (step 5)
AUC_true = dose * (24 / 12) / params['Cl_total'] # Calculates AUC based on true Cl_total randomized values

# Save step 5 to CSV
# with open('step5_auc_true.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Group', 'AUC_true'])
#     for i in range(N):
#         writer.writerow([i+1, AUC_true[i]])

# Add noise to get randomized levels (step 6)
randomized_levels = true_levels * (1 + stats.norm.ppf(np.random.random((N, len(times_true))), 0, 0.1))

# Select levels at times_fit
mask = np.isin(times_true, times_fit)
selected_levels = randomized_levels[:, mask]
selected_times = times_true[mask]

# Save step 6 to CSV
# with open('step6_randomized_levels.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     header = ['Group'] + [f'Cp_{t}_rand' for t in times_true]
#     writer.writerow(header)
#     for i in range(N):
#         writer.writerow([i+1] + list(randomized_levels[i]))

# One Compartment Non Bayesian Peak Trough Model (step 8)
Vdcalc = np.zeros(N)
Clcalc = np.zeros(N)
AUCcalc = np.zeros(N)
for i in range(N):
    idx_2h = np.where(times_true == 2)[0][0]
    idx_10h = np.where(times_true == 10)[0][0]
    Kcalc = np.log(randomized_levels[i, idx_2h] / randomized_levels[i, idx_10h]) / 8
    Vdcalc[i] = dose * (1 - np.exp(-Kcalc * 2)) * np.exp(-Kcalc * 2) / (randomized_levels[i, idx_2h] * Kcalc * 2 * (1 - np.exp(-Kcalc * 12)))
    Clcalc[i] = Kcalc * Vdcalc[i]
    AUCcalc[i] = dose * (24 / 12) / Clcalc[i]

Vdpop_mean = np.mean(Vdcalc[Vdcalc > 0]) #values with only Vdcalc > 0 as negative values are invalid
Clpop = np.mean(Clcalc[Vdcalc > 0])
Vdpop_std_mean = np.std(Vdcalc[Vdcalc > 0])
Clpop_std = np.std(Clcalc[Vdcalc > 0])
Vdpop_median = np.median(Vdcalc[Vdcalc > 0])
Vdpop_geometric_mean = np.exp(np.mean(np.log(Vdcalc[Vdcalc > 0])))
Vdpop_geometric_std = np.exp(np.std(np.log(Vdcalc[Vdcalc > 0])))

Vdpop = Vdpop_geometric_mean


# Save step 8 to CSV
# with open('step8_one_comp_non_bayes.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Group', 'Vdcalc', 'Clcalc', 'AUCcalc'])
#     for i in range(N):
#         writer.writerow([i+1, Vdcalc[i], Clcalc[i], AUCcalc[i]])

# Starting values for one comp fitting (step 9)
start_Vd = np.random.normal(Vdpop, Vdpop * 0.3, N)
start_Cl = np.random.normal(Clpop, Clpop * 0.3, N)
Vd_limits = (Vdpop - Vdpop * 0.3 * 2, Vdpop + Vdpop * 0.3 * 2)
Cl_limits = (Clpop - Clpop * 0.3 * 2, Clpop + Clpop * 0.3 * 2)
start_Vd = np.clip(start_Vd, Vd_limits[0], Vd_limits[1])
start_Cl = np.clip(start_Cl, Cl_limits[0], Cl_limits[1])

# Function for one comp Cp (fixed Vd)
def calculate_cp_1comp_fixedVd(Cl, times, Vd=Vdpop):
    K = Cl / Vd
    Cp = dose * (1 - np.exp(-K * 2)) * np.exp(-K * times) / (Cl * 2 * (1 - np.exp(-K * 12)))
    return Cp

# Fit for fixed Vd (step 10), This is fitting all entered times entered just like the bayesian model
fitted_Cl_fixedVd = np.zeros(N)
fitted_levels_fixedVd = np.zeros((N, len(times_fit)))
sse_fixedVd = np.zeros(N)
for i in range(N):
    try:
        idx_10h = np.where(times_true == 10)[0][0]
        target = randomized_levels[i, idx_10h]
        def obj(Cl):
            return (calculate_cp_1comp_fixedVd(Cl, 10) - target)**2
        bounds = (Clpop - Clpop * 0.3 * 3, Clpop + Clpop * 0.3 * 3)
        res = optimize.minimize_scalar(obj, bounds=bounds, method='bounded')
        fitted_Cl_fixedVd[i] = res.x
        fitted_levels_fixedVd[i] = calculate_cp_1comp_fixedVd(fitted_Cl_fixedVd[i], times_fit)
        sse_fixedVd[i] = res.fun
    except:
        fitted_Cl_fixedVd[i] = np.nan
        fitted_levels_fixedVd[i] = np.full(len(times_fit), np.nan)
        sse_fixedVd[i] = np.nan

AUC_fit_fixedVd = dose * (24 / 12) / fitted_Cl_fixedVd

# Save step 10 to CSV
# with open('step10_fit_fixedVd.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     header = ['Group'] + [f'Cp_{t}_fit' for t in times_fit] + ['Cl_fit', 'AUC_fit', 'SSE']
#     writer.writerow(header)
#     for i in range(N):
#         writer.writerow([i+1] + list(fitted_levels_fixedVd[i]) + [fitted_Cl_fixedVd[i], AUC_fit_fixedVd[i], sse_fixedVd[i]])

# Function for one comp Cp (Bayesian)
def calculate_cp_1comp(Cl, Vd, times):
    K = Cl / Vd
    Cp = dose * (1 - np.exp(-K * 2)) * np.exp(-K * times) / (Cl * 2 * (1 - np.exp(-K * 12)))
    return Cp

# Residuals for Bayesian fit
def residuals_1comp(p, times, levels):
    Vd, Cl = p
    cp_fit = calculate_cp_1comp(Cl, Vd, times)
    res_levels = np.where(np.abs(levels) < 1e-6, 0, (levels - cp_fit) / (0.1 * np.abs(levels)))
    res_vd = (Vd - Vdpop) / (0.3 * Vdpop)
    res_cl = (Cl - Clpop) / (0.3 * Clpop)
    return np.concatenate([res_levels, [res_vd, res_cl]])

# Fit Bayesian (step 14) for multiple scenarios
scenarios = ['full', 'last']
time_subsets = [times_fit, times_fit[-1:] if len(times_fit) > 0 else []]
bayes_results = {}
for scenario, subset in zip(scenarios, time_subsets):
    if len(subset) == 0:
        continue
    mask = np.isin(times_true, subset)
    selected_levels_sub = randomized_levels[:, mask]
    selected_times_sub = times_true[mask]
    fitted_Vd = np.zeros(N)
    fitted_Cl = np.zeros(N)
    fitted_levels = np.zeros((N, len(subset)))
    sse = np.zeros(N)
    for i in range(N):
        try:
            x0 = [start_Vd[i], start_Cl[i]]
            bounds = ([Vdpop - Vdpop * 0.3 * 3, Clpop - Clpop * 0.3 * 3], [Vdpop + Vdpop * 0.3 * 3, Clpop + Clpop * 0.3 * 3])
            res = optimize.least_squares(residuals_1comp, x0, bounds=bounds, args=(selected_times_sub, selected_levels_sub[i]))
            fitted_Vd[i] = res.x[0]
            fitted_Cl[i] = res.x[1]
            fitted_levels[i] = calculate_cp_1comp(fitted_Cl[i], fitted_Vd[i], subset)
            sse[i] = 2 * res.cost
        except:
            fitted_Vd[i] = np.nan
            fitted_Cl[i] = np.nan
            fitted_levels[i] = np.full(len(subset), np.nan)
            sse[i] = np.nan
    AUC_fit = dose * (24 / 12) / fitted_Cl
    bayes_results[scenario] = {
        'Vd': fitted_Vd,
        'Cl': fitted_Cl,
        'levels': fitted_levels,
        'AUC': AUC_fit,
        'SSE': sse,
        'times': subset
    }

# For backward compatibility, set the original variables to 'full' if exists
if 'full' in bayes_results:
    fitted_Vd_bayes = bayes_results['full']['Vd']
    fitted_Cl_bayes = bayes_results['full']['Cl']
    fitted_levels_bayes = bayes_results['full']['levels']
    sse_bayes = bayes_results['full']['SSE']
    AUC_fit_bayes = bayes_results['full']['AUC']
else:
    fitted_Vd_bayes = np.full(N, np.nan)
    fitted_Cl_bayes = np.full(N, np.nan)
    fitted_levels_bayes = np.full((N, len(times_fit)), np.nan)
    sse_bayes = np.full(N, np.nan)
    AUC_fit_bayes = np.full(N, np.nan)

# Save all data to one CSV
with open(f'{output_dir}/monte_carlo_all_data_onevrstwocompartment_geometric_mean{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = (['Group'] + 
              ['Vc_true', 'Vp_true', 'Cl_total_true', 'Cl_dist_true'] + 
              [f'Cp_{t}_true' for t in times_true] + 
              ['AUC_true'] + 
              [f'Cp_{t}_rand' for t in times_true] + 
              ['Vdcalc', 'Clcalc', 'AUCcalc'] + 
              [f'Cp_{t}_fit_fixed' for t in times_fit] + 
              ['Cl_fit_fixed', 'AUC_fit_fixed', 'SSE_fixed'])
    for scenario in scenarios:
        if scenario in bayes_results:
            subset = bayes_results[scenario]['times']
            header += [f'Cp_{t}_fit_bayes_{scenario}' for t in subset] + [f'Vd_fit_bayes_{scenario}', f'Cl_fit_bayes_{scenario}', f'AUC_fit_bayes_{scenario}', f'SSE_bayes_{scenario}']
    header += ['Cl_diff_calc', 'Cl_diff_calc_sq', 'Cl_diff_fixed', 'Cl_diff_fixed_sq']
    for scenario in scenarios:
        if scenario in bayes_results:
            header += [f'Cl_diff_bayes_{scenario}', f'Cl_diff_bayes_{scenario}_sq']
    writer.writerow(header)
    for i in range(N):
        row = ([i+1] + 
               [params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i]] + 
               list(true_levels[i]) + 
               [AUC_true[i]] + 
               list(randomized_levels[i]) + 
               [Vdcalc[i], Clcalc[i], AUCcalc[i]] + 
               list(fitted_levels_fixedVd[i]) + 
               [fitted_Cl_fixedVd[i], AUC_fit_fixedVd[i], sse_fixedVd[i]])
        for scenario in scenarios:
            if scenario in bayes_results:
                res = bayes_results[scenario]
                row += list(res['levels'][i]) + [res['Vd'][i], res['Cl'][i], res['AUC'][i], res['SSE'][i]]
        cl_diff_calc = params['Cl_total'][i] - Clcalc[i]
        cl_diff_fixed = params['Cl_total'][i] - fitted_Cl_fixedVd[i]
        row += [cl_diff_calc, cl_diff_calc ** 2, cl_diff_fixed, cl_diff_fixed ** 2]
        for scenario in scenarios:
            if scenario in bayes_results:
                cl_diff_bayes = params['Cl_total'][i] - bayes_results[scenario]['Cl'][i]
                row += [cl_diff_bayes, cl_diff_bayes ** 2]
        writer.writerow(row)

print(f"Monte Carlo data saved to {output_dir}/monte_carlo_all_data_onevrstwocompartment_geometric_mean{file_suffix}.csv")

# Grids for each model
bins = [0, 400, 601, np.inf]

# Non Bayesian Peak Trough
hist_nb = np.histogram2d(AUC_true, AUCcalc, bins=[bins, bins])[0]
hist_nb_filtered = np.histogram2d(AUC_true[Vdcalc > 0], AUCcalc[Vdcalc > 0], bins=[bins, bins])[0]

# Fixed Vd
hist_fixed = np.histogram2d(AUC_true, AUC_fit_fixedVd, bins=[bins, bins])[0]

# Bayesian
hist_bayes_full = np.histogram2d(AUC_true, bayes_results.get('full', {'AUC': np.full(N, np.nan)})['AUC'], bins=[bins, bins])[0] if 'full' in bayes_results else np.zeros((3,3))
hist_bayes_last = np.histogram2d(AUC_true, bayes_results.get('last', {'AUC': np.full(N, np.nan)})['AUC'], bins=[bins, bins])[0] if 'last' in bayes_results else np.zeros((3,3))

models = ['Non Bayesian Peak Trough', 'Fixed Vd']
hists = [hist_nb_filtered, hist_fixed]
auc_fits = [AUCcalc, AUC_fit_fixedVd]
if 'full' in bayes_results:
    models.append('Bayesian_full')
    hists.append(hist_bayes_full)
    auc_fits.append(bayes_results['full']['AUC'])
if 'last' in bayes_results:
    models.append('Bayesian_last')
    hists.append(hist_bayes_last)
    auc_fits.append(bayes_results['last']['AUC'])

# Save grids to CSV
with open(f'{output_dir}/grid_onevrstwocompart_Geometric_Mean{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for model, hist in zip(models, hists):
        writer.writerow([model])
        writer.writerow(['Raw Numbers'])
        writer.writerow(['AUC_true \\ AUC_fit', 'Total', '< 400', '400 to 600', '> 600'])
        for i, row in enumerate(hist):
            total = int(np.sum(row))
            writer.writerow([f'{bins[i]}-{bins[i+1]}' if i < len(bins)-2 else f'>{bins[i]}', total] + list(row.astype(int)))
        writer.writerow([])
        writer.writerow(['Percentages'])
        writer.writerow(['AUC_true \\ AUC_fit', 'Total', '< 400', '400 to 600', '> 600'])
        total = np.sum(hist)
        percentages = (hist / total * 100) if total > 0 else hist
        diagonal_sum = np.sum(np.diag(hist))
        total_correct_pct = (diagonal_sum / total * 100) if total > 0 else 0
        for i, row in enumerate(percentages):
            row_total = np.sum(row)
            writer.writerow([f'{bins[i]}-{bins[i+1]}' if i < len(bins)-2 else f'>{bins[i]}', f'{row_total:.2f}'] + [f'{val:.2f}' for val in row])
        writer.writerow([f'Total Percentage Correct: {total_correct_pct:.2f}%'])
        writer.writerow([])
print(f"Monte Carlo data saved to {output_dir}/monte_carlo_all_data_onevrstwocompartment_geometric_mean{file_suffix}.csv")
print(f"Grids saved to {output_dir}/grid_onevrstwocompart_Geometric_Mean{file_suffix}.csv")

Vdpop_geometric_std = np.exp(np.std(np.log(Vdcalc[Vdcalc > 0])))

Vdpop = Vdpop_geometric_mean


# Save step 8 to CSV
# with open('step8_one_comp_non_bayes.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Group', 'Vdcalc', 'Clcalc', 'AUCcalc'])
#     for i in range(N):
Vdpop_geometric_std = np.exp(np.std(np.log(Vdcalc[Vdcalc > 0])))

Vdpop = Vdpop_geometric_mean

# AUC data for one compartment
with open(f'{output_dir}/auc_data_one_compartment{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['AUC_true', 'AUC_fit', 'Model'])
    for auc_fit, model in zip(auc_fits, models):
        for true, fit in zip(AUC_true, auc_fit):
            writer.writerow([true, fit, model])

with open(f'{output_dir}/auc_cl_data_one_compartment{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Cl_true', 'Cl_fit', 'Model'])
    cl_fits = [Clcalc, fitted_Cl_fixedVd]
    if 'full' in bayes_results:
        cl_fits.append(bayes_results['full']['Cl'])
    if 'last' in bayes_results:
        cl_fits.append(bayes_results['last']['Cl'])
    for cl_fit, model in zip(cl_fits, models):
        for true, fit in zip(params['Cl_total'], cl_fit):
            writer.writerow([true, fit, model])

# Bland-Altman analysis
model_names = ['Non Bayesian Peak Trough', 'Fixed Vd']
auc_fits_stats = [AUCcalc, AUC_fit_fixedVd]
cl_fits_stats = [Clcalc, fitted_Cl_fixedVd]
if 'full' in bayes_results:
    model_names.append('Bayesian_full')
    auc_fits_stats.append(bayes_results['full']['AUC'])
    cl_fits_stats.append(bayes_results['full']['Cl'])
if 'last' in bayes_results:
    model_names.append('Bayesian_last')
    auc_fits_stats.append(bayes_results['last']['AUC'])
    cl_fits_stats.append(bayes_results['last']['Cl'])

auc_biases = []
auc_rmses = []
rmse_ratios = []
auc_sds = []
auc_uppers = []
auc_lowers = []
cl_biases = []
cl_rmses = []
pearson_rs = []

for model_name, auc_fit, cl_fit in zip(model_names, auc_fits, cl_fits):
    auc_differences = AUC_true - auc_fit
    auc_means = (AUC_true + auc_fit) / 2
    auc_bias = np.mean(auc_differences)
    auc_sd = np.std(auc_differences)
    auc_rmse = np.sqrt(np.mean(auc_differences**2))
    auc_rmse_ratio = auc_rmse / auc_sd
    auc_upper = auc_bias + 1.96 * auc_sd
    auc_lower = auc_bias - 1.96 * auc_sd
    auc_biases.append(auc_bias)
    auc_rmses.append(auc_rmse)
    rmse_ratios.append(auc_rmse_ratio)
    auc_sds.append(auc_sd)
    auc_uppers.append(auc_upper)
    auc_lowers.append(auc_lower)
    cl_differences = params['Cl_total'] - cl_fit
    cl_bias = np.mean(cl_differences)
    cl_rmse = np.sqrt(np.mean(cl_differences**2))
    pearson_r, _ = stats.pearsonr(AUC_true, auc_fit)
    cl_biases.append(cl_bias)
    cl_rmses.append(cl_rmse)
    pearson_rs.append(pearson_r)
    with open(f'{output_dir}/bland_altman_data_{model_name.lower().replace(" ", "_")}{file_suffix}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['AUC_mean', 'AUC_difference'])
        for mean, diff in zip(auc_means, auc_differences):
            writer.writerow([mean, diff])

# Save statistics
with open(f'{output_dir}/cl_differences_statistics_Geometric_Mean{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Statistic', 'Value'])
    for i, model in enumerate(model_names):
        writer.writerow([])
        writer.writerow([f'Average AUC Bias {model}', auc_biases[i]])
        writer.writerow([f'AUC RMSE {model}', auc_rmses[i]])
        writer.writerow([f'AUC RMSE/SD Ratio {model}', rmse_ratios[i]])
        writer.writerow([f'AUC SD {model}', auc_sds[i]])
        writer.writerow([f'AUC Upper 95% Limit Bland-Altman {model}', auc_uppers[i]])
        writer.writerow([f'AUC Lower 95% Limit Bland-Altman {model}', auc_lowers[i]])
        writer.writerow([f'Average Cl Bias {model}', cl_biases[i]])
        writer.writerow([f'Cl RMSE {model}', cl_rmses[i]])
        writer.writerow([f'Pearson r AUC predicted vs True {model}', pearson_rs[i]])
    for key in pop_params:
        writer.writerow([f'{key} Percent Clipped', clipping_stats[key]])
    writer.writerow([])
    writer.writerow(['Peak Trough Clpop', Clpop])
    writer.writerow(['Peak Trough Clpop_std', Clpop_std])
    writer.writerow(['Peak Trough Vdpop_mean', Vdpop_mean])
    writer.writerow(['Peak Trough Vdpop_std_mean', Vdpop_std_mean])
    writer.writerow(['Peak Trough Vdpop_median', Vdpop_median])
    writer.writerow(['Peak Trough Vdpop_geometric_mean', Vdpop_geometric_mean])
    writer.writerow(['Peak Trough Vdpop_geometric_std', Vdpop_geometric_std])

print(f"Statistics saved to {output_dir}/cl_differences_statistics_Geometric_Mean{file_suffix}.csv")

# Plot Bland-Altman for each model
for model_name in model_names:
    with open(f'{output_dir}/bland_altman_data_{model_name.lower().replace(" ", "_")}{file_suffix}.csv', 'r') as csvfile:
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
    plt.title(f'Bland-Altman Plot for {model_name}')
    plt.legend()
    plt.grid(True)
    title_for_filename = f'Bland-Altman Plot for {model_name}'.replace(' ', '_').replace(':', '_')
    plt.savefig(f'{output_dir}/{title_for_filename}_{py_file_name}{file_suffix}.png', dpi=300)
    plt.show()
