import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt  # optional for plotting

# Population parameters
pop_params = {'Vc': 58.4, 'Vp': 38.4, 'Cl_total': 4.5, 'Cl_dist': 6.5}
cvs = {'Vc': 0.3, 'Vp': 0.3, 'Cl_total': 0.3, 'Cl_dist': 0.4}
limits = {'Vc': (23, 93), 'Vp': (15, 61), 'Cl_total': (1.8, 7.2), 'Cl_dist': (1.3, 11.7)}
fit_bounds = {'Vc': (5.8, 110), 'Vp': (3.8, 72), 'Cl_total': (0.5, 10), 'Cl_dist': (1.3, 11.7)}

N = 30000
times_true = np.array([0, 1, 2, 3, 4, 8, 9, 10])
times_fit = times_true  # User can modify this list for fitting times

# Generate random parameters for true values
params = {}
for key in pop_params:
    mean = pop_params[key]
    sd = mean * cvs[key]
    raw = np.random.normal(mean, sd, N)
    params[key] = np.clip(raw, limits[key][0], limits[key][1])

# Generate starting parameters for fitting
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
    A = 1012.6 * (K21 - alpha) * (1 - np.exp(-alpha * 2)) / (2 * Vc * alpha * (beta - alpha) * (1 - np.exp(-alpha * 12)))
    B = 1012.6 * (beta - K21) * (1 - np.exp(-beta * 2)) / (2 * Vc * beta * (beta - alpha) * (1 - np.exp(-beta * 12)))
    Cp = A * np.exp(-alpha * times) + B * np.exp(-beta * times)
    return Cp

# Calculate true levels
true_levels = np.zeros((N, len(times_true)))
for i in range(N):
    true_levels[i] = calculate_cp(params['Vc'][i], params['Vp'][i], params['Cl_total'][i], params['Cl_dist'][i], times_true)

# Calculate AUC_true
AUC_true = 1012.6 * (24 / 12) / params['Cl_total']

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

# Fit for each group
fitted_params = {'Vc': np.zeros(N), 'Vp': np.zeros(N), 'Cl_total': np.zeros(N), 'Cl_dist': np.zeros(N)}
for i in range(N):
    x0 = [start_params['Vc'][i], start_params['Vp'][i], start_params['Cl_total'][i], start_params['Cl_dist'][i]]
    bounds = ([fit_bounds['Vc'][0], fit_bounds['Vp'][0], fit_bounds['Cl_total'][0], fit_bounds['Cl_dist'][0]],
              [fit_bounds['Vc'][1], fit_bounds['Vp'][1], fit_bounds['Cl_total'][1], fit_bounds['Cl_dist'][1]])
    res = optimize.least_squares(residuals, x0, bounds=bounds,
                                 args=(pop_params['Vc'], pop_params['Vp'], pop_params['Cl_total'], pop_params['Cl_dist'],
                                       times_fit, randomized_levels[i]))
    fitted_params['Vc'][i] = res.x[0]
    fitted_params['Vp'][i] = res.x[1]
    fitted_params['Cl_total'][i] = res.x[2]
    fitted_params['Cl_dist'][i] = res.x[3]

# Calculate AUC_fit
AUC_fit = 1012.6 * (24 / 12) / fitted_params['Cl_total']

# Create the comparison grid
bins = [0, 300, 400, 500, 600, np.inf]
hist, xedges, yedges = np.histogram2d(AUC_true, AUC_fit, bins=[bins, bins])

print("Actual AUC Ranges\tNumber Fit in AUC Ranges Below")
print("Number In AUC ranges\tTotal\t< 300\t300 to < 400\t400 to < 500\t500 to < 600\t>= 600")
for i in range(len(bins)-1):
    total = int(np.sum(hist[i]))
    row_label = f"{bins[i]} to < {bins[i+1]}" if bins[i+1] != np.inf else f">= {bins[i]}"
    print(f"{row_label}\t{total}\t{'\t'.join(map(str, hist[i].astype(int)))}")

# Percentage grid
print("\nPercentage of Fit AUCs in Each Range")
print("Actual AUC Ranges\tPercentage Fit in AUC Ranges Below")
print("Number In AUC ranges\tTotal\t< 300\t300 to < 400\t400 to < 500\t500 to < 600\t>= 600")
for i in range(len(bins)-1):
    total = np.sum(hist[i])
    if total > 0:
        percs = hist[i] / total * 100
    else:
        percs = np.zeros_like(hist[i])
    row_label = f"{bins[i]} to < {bins[i+1]}" if bins[i+1] != np.inf else f">= {bins[i]}"
    print(f"{row_label}\t{total:.0f}\t{'\t'.join(map(lambda x: f'{x:.1f}%', percs))}")