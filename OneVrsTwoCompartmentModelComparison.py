import numpy as np
from scipy import stats, optimize
import csv
# two compartment vs one compartment model comparisons 
np.random.seed(42)  # For repeatable random values

# Population parameters for two-compartment limits are used in parameter randomization
# adjusted Cl_total to reflect population mean from real data
pop_params = {'Vc': 58.4, 'Vp': 38.4, 'Cl_total': 1.485, 'Cl_dist': 6.5}#Cl_total 4.5
cvs = {'Vc': 0.3, 'Vp': 0.3, 'Cl_total': 0.3, 'Cl_dist': 0.4}
limits = {'Vc': (23, 93), 'Vp': (15, 61), 'Cl_total': (0.594, 2.375), 'Cl_dist': (1.3, 11.7)}#2SD 'Cl_total':(1.8,7.2)

N = 30000
# the times below are the acceptable times for levels to be entered in line 17.
times_true = np.array([0, 1, 2, 3, 4, 8, 9, 10])# User entered time levels for two compartment fiting, levels in grids.cvs

# User input for fitting times (up to 8)
times_fit_input = input("Bayesian Analysis One compartment model:Enter up to 8 times separated by space or comma: ").replace(',', ' ').split()
try:
    times_fit = np.array([float(t) for t in times_fit_input[:8]])
    # Ensure times_fit are subset of times_true for simplicity
    times_fit = np.array([t for t in times_fit if t in times_true])
    if len(times_fit) == 0:
        times_fit = times_true
except ValueError:
    print("Invalid input. Using default times: 0 1 2 3 4 8 9 10")
    times_fit = times_true

# User input for file suffix
file_suffix = input("Enter file suffix for CSV files (e.g., _v1, leave blank for none): ")

# Time points for true and fitting concentrations post dose 
#times_true = np.array([10]) # 0, 1, 2, 3, 4, 8, 9, 10
#times_fit = times_true  # User can modify this list for fitting times

# Generate random parameters for true values (two-compartment)
params = {}
for key in pop_params:
    mean = pop_params[key]
    sd = mean * cvs[key]
    raw = np.random.normal(mean, sd, N)
    params[key] = np.clip(raw, limits[key][0], limits[key][1])

# Save step 2 to CSV
# with open('step2_params.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Group', 'Vc', 'Vp', 'Cl_total', 'Cl_dist'])
#     for i in range(N):
#         writer.writerow([i+1, params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i]])

# Function to calculate Cp for two-compartment
def calculate_cp_2comp(Vc, Vp, Cl_total, Cl_dist, times):
    K10 = Cl_total / Vc
    K12 = Cl_dist / Vc
    K21 = Cl_dist / Vp
    sum_k = K10 + K12 + K21
    disc = np.sqrt(sum_k**2 - 4 * K21 * K10)
    alpha = 0.5 * (sum_k + disc)
    beta = 0.5 * (sum_k - disc)
    A = 334 * (K21 - alpha) * (1 - np.exp(-alpha * 2)) / (2 * Vc * alpha * (beta - alpha) * (1 - np.exp(-alpha * 12)))
    B = 334 * (beta - K21) * (1 - np.exp(-beta * 2)) / (2 * Vc * beta * (beta - alpha) * (1 - np.exp(-beta * 12)))
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
AUC_true = 334 * (24 / 12) / params['Cl_total'] # Calculates AUC based on true Cl_total randomized values

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
    Vdcalc[i] = 334* (1 - np.exp(-Kcalc * 2)) * np.exp(-Kcalc * 2) / (randomized_levels[i, idx_2h] * Kcalc * 2 * (1 - np.exp(-Kcalc * 12)))
    Clcalc[i] = Kcalc * Vdcalc[i]
    AUCcalc[i] = 334 * (24 / 12) / Clcalc[i]

Vdpop = np.mean(Vdcalc[Vdcalc > 0])
Clpop = np.mean(Clcalc[Vdcalc > 0])
Vdpop_std = np.std(Vdcalc[Vdcalc > 0])
Clpop_std = np.std(Clcalc[Vdcalc > 0])
Vdpop_median = np.median(Vdcalc[Vdcalc > 0])
Vdpop_geometric_mean = np.exp(np.mean(np.log(Vdcalc[Vdcalc > 0])))
Vdpop_geometric_std = np.exp(np.std(np.log(Vdcalc[Vdcalc > 0])))


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
    Cp = 334 * (1 - np.exp(-K * 2)) * np.exp(-K * times) / (Cl * 2 * (1 - np.exp(-K * 12)))
    return Cp

# Fit for fixed Vd (step 10), This is fitting all entered times entered just like the bayesian model
fitted_Cl_fixedVd = np.zeros(N)
fitted_levels_fixedVd = np.zeros((N, len(times_fit)))
sse_fixedVd = np.zeros(N)
for i in range(N):
    idx_10h = np.where(times_true == 10)[0][0]
    target = randomized_levels[i, idx_10h]
    def obj(Cl):
        return (calculate_cp_1comp_fixedVd(Cl, 10) - target)**2
    bounds = (Clpop - Clpop * 0.3 * 3, Clpop + Clpop * 0.3 * 3)
    res = optimize.minimize_scalar(obj, bounds=bounds, method='bounded')
    fitted_Cl_fixedVd[i] = res.x
    fitted_levels_fixedVd[i] = calculate_cp_1comp_fixedVd(fitted_Cl_fixedVd[i], times_fit)
    sse_fixedVd[i] = res.fun

AUC_fit_fixedVd = 334 * (24 / 12) / fitted_Cl_fixedVd

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
    Cp = 334 * (1 - np.exp(-K * 2)) * np.exp(-K * times) / (Cl * 2 * (1 - np.exp(-K * 12)))
    return Cp

# Residuals for Bayesian fit
def residuals_1comp(p, times, levels):
    Vd, Cl = p
    cp_fit = calculate_cp_1comp(Cl, Vd, times)
    res_levels = (levels - cp_fit) / (0.1 * levels)
    res_vd = (Vd - Vdpop) / (0.3 * Vdpop)
    res_cl = (Cl - Clpop) / (0.3 * Clpop)
    return np.concatenate([res_levels, [res_vd, res_cl]])

# Fit Bayesian (step 14)
fitted_Vd_bayes = np.zeros(N)
fitted_Cl_bayes = np.zeros(N)
fitted_levels_bayes = np.zeros((N, len(times_fit)))
sse_bayes = np.zeros(N)
for i in range(N):
    x0 = [start_Vd[i], start_Cl[i]]
    bounds = ([Vdpop - Vdpop * 0.3 * 3, Clpop - Clpop * 0.3 * 3], [Vdpop + Vdpop * 0.3 * 3, Clpop + Clpop * 0.3 * 3])
    res = optimize.least_squares(residuals_1comp, x0, bounds=bounds, args=(selected_times, selected_levels[i]))
    fitted_Vd_bayes[i] = res.x[0]
    fitted_Cl_bayes[i] = res.x[1]
    fitted_levels_bayes[i] = calculate_cp_1comp(fitted_Cl_bayes[i], fitted_Vd_bayes[i], times_fit)
    sse_bayes[i] = 2 * res.cost

AUC_fit_bayes = 334 * (24 / 12) / fitted_Cl_bayes

# Save all data to one CSV
with open(f'monte_carlo_all_data_onevrstwocompartment{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = (['Group'] + 
              ['Vc_true', 'Vp_true', 'Cl_total_true', 'Cl_dist_true'] + 
              [f'Cp_{t}_true' for t in times_true] + 
              ['AUC_true'] + 
              [f'Cp_{t}_rand' for t in times_true] + 
              ['Vdcalc', 'Clcalc', 'AUCcalc'] + 
              [f'Cp_{t}_fit_fixed' for t in times_fit] + 
              ['Cl_fit_fixed', 'AUC_fit_fixed', 'SSE_fixed'] + 
              [f'Cp_{t}_fit_bayes' for t in times_fit] + 
              ['Vd_fit_bayes', 'Cl_fit_bayes', 'AUC_fit_bayes', 'SSE_bayes'] +
              ['Cl_diff_calc', 'Cl_diff_calc_sq', 'Cl_diff_fixed', 'Cl_diff_fixed_sq', 'Cl_diff_bayes', 'Cl_diff_bayes_sq'])
    writer.writerow(header)
    for i in range(N):
        cl_diff_calc = params['Cl_total'][i] - Clcalc[i]
        cl_diff_fixed = params['Cl_total'][i] - fitted_Cl_fixedVd[i]
        cl_diff_bayes = params['Cl_total'][i] - fitted_Cl_bayes[i]
        cl_diff_calc_sq = cl_diff_calc ** 2
        cl_diff_fixed_sq = cl_diff_fixed ** 2
        cl_diff_bayes_sq = cl_diff_bayes ** 2
        row = ([i+1] + 
               [params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i]] + 
               list(true_levels[i]) + 
               [AUC_true[i]] + 
               list(randomized_levels[i]) + 
               [Vdcalc[i], Clcalc[i], AUCcalc[i]] + 
               list(fitted_levels_fixedVd[i]) + 
               [fitted_Cl_fixedVd[i], AUC_fit_fixedVd[i], sse_fixedVd[i]] + 
               list(fitted_levels_bayes[i]) + 
               [fitted_Vd_bayes[i], fitted_Cl_bayes[i], AUC_fit_bayes[i], sse_bayes[i]] +
               [cl_diff_calc, cl_diff_calc_sq, cl_diff_fixed, cl_diff_fixed_sq, cl_diff_bayes, cl_diff_bayes_sq])
        writer.writerow(row)

# Grids for each model
bins = [0, 300, 400, 500, 600, np.inf]

# Non Bayesian Peak Trough
hist_nb = np.histogram2d(AUC_true, AUCcalc, bins=[bins, bins])[0]

# Fixed Vd
hist_fixed = np.histogram2d(AUC_true, AUC_fit_fixedVd, bins=[bins, bins])[0]

# Bayesian
hist_bayes = np.histogram2d(AUC_true, AUC_fit_bayes, bins=[bins, bins])[0]

# Save grids to CSV
models = ['Non Bayesian Peak Trough', 'Fixed Vd', 'Bayesian']
hists = [hist_nb, hist_fixed, hist_bayes]

with open(f'grid_onevrstwocompart{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for model, hist in zip(models, hists):
        writer.writerow([f"{model} - Comparison Grid"])
        writer.writerow(['Actual AUC Ranges', 'Number Fit in AUC Ranges Below'])
        writer.writerow(['Number In AUC ranges', 'Total', '< 300', '300 to < 400', '400 to < 500', '500 to < 600', '>= 600'])
        for i in range(len(bins)-1):
            total = int(np.sum(hist[i]))
            row_label = f"{bins[i]} to < {bins[i+1]}" if bins[i+1] != np.inf else f">= {bins[i]}"
            writer.writerow([row_label, total] + list(hist[i].astype(int)))
        
        writer.writerow([])
        writer.writerow([f"{model} - Percentage Grid"])
        writer.writerow(['Actual AUC Ranges', 'Percentage Fit in AUC Ranges Below'])
        writer.writerow(['Number In AUC ranges', 'Total', '< 300', '300 to < 400', '400 to < 500', '500 to < 600', '>= 600'])
        for i in range(len(bins)-1):
            total = np.sum(hist[i])
            if total > 0:
                percs = hist[i] / total * 100
            else:
                percs = np.zeros_like(hist[i])
            row_label = f"{bins[i]} to < {bins[i+1]}" if bins[i+1] != np.inf else f">= {bins[i]}"
            writer.writerow([row_label, f"{total:.0f}"] + [f"{x:.1f}%" for x in percs])
        writer.writerow([])

print("Grids saved to grid_onevrstwocompart.csv")

# Calculate statistics for differences
cl_diff_calc_all = params['Cl_total'] - Clcalc
cl_diff_fixed_all = params['Cl_total'] - fitted_Cl_fixedVd
cl_diff_bayes_all = params['Cl_total'] - fitted_Cl_bayes

cl_diff_calc_sq_all = cl_diff_calc_all ** 2
cl_diff_fixed_sq_all = cl_diff_fixed_all ** 2
cl_diff_bayes_sq_all = cl_diff_bayes_all ** 2

avg_cl_diff_calc = np.mean(cl_diff_calc_all)
sqrt_avg_cl_diff_calc_sq = np.sqrt(np.mean(cl_diff_calc_sq_all))
avg_cl_diff_fixed = np.mean(cl_diff_fixed_all)
sqrt_avg_cl_diff_fixed_sq = np.sqrt(np.mean(cl_diff_fixed_sq_all))
avg_cl_diff_bayes = np.mean(cl_diff_bayes_all)
sqrt_avg_cl_diff_bayes_sq = np.sqrt(np.mean(cl_diff_bayes_sq_all))

# Calculate AIC for Fixed Vd and Bayesian methods
n = len(times_fit)  # Number of observations (time points) per fit
k_fixed = 1  # Parameters: Cl (Vd fixed)
k_bayes = 2  # Parameters: Vd and Cl

aic_fixed = 2 * k_fixed + n * np.log(sse_fixedVd / n)
aic_bayes = 2 * k_bayes + n * np.log(sse_bayes / n)

avg_aic_fixed = np.mean(aic_fixed)
avg_aic_bayes = np.mean(aic_bayes)

# Save statistics to separate CSV
with open(f'cl_differences_statistics{file_suffix}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Statistic', 'Value'])
    writer.writerow(['Average Cl_diff_calc pk trough calc', avg_cl_diff_calc])
    writer.writerow(['Sqrt Average Cl_diff_calc_sq pk trough calc', sqrt_avg_cl_diff_calc_sq])
    writer.writerow(['Average Cl_diff_fixed VD', avg_cl_diff_fixed])
    writer.writerow(['Sqrt Average Cl_diff_fixed_sq fixed VD', sqrt_avg_cl_diff_fixed_sq])
    writer.writerow(['Average Cl_diff_bayes', avg_cl_diff_bayes])
    writer.writerow(['Sqrt Average Cl_diff_bayes_sq', sqrt_avg_cl_diff_bayes_sq])
    writer.writerow(['Average AIC Fixed VD', avg_aic_fixed])
    writer.writerow(['Average AIC Bayesian', avg_aic_bayes])
    writer.writerow(['Vdpop', Vdpop])
    writer.writerow(['Clpop', Clpop])
    writer.writerow(['Vdpop Std Dev', Vdpop_std])
    writer.writerow(['Clpop Std Dev', Clpop_std])
    writer.writerow(['Vdpop Median', Vdpop_median])
    writer.writerow(['Vdpop Geometric Mean', Vdpop_geometric_mean])
    writer.writerow(['Vdpop Geometric Std', Vdpop_geometric_std])

print("Statistics saved to cl_differences_statistics.csv")
