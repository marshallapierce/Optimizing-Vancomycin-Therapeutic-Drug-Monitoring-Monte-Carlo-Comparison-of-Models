import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys

# Statistical comparisons for 1 compartment non-bayesian Peak-trough vrs 2 compartment Bayesian Peak-Trough,
# 1 compartarment Bayesian peak-trough vrs 2 compartment Bayesian peak-trough
# Adapted from statistical_comparisons.py

# Function to load detailed Monte Carlo results and extract metrics
def load_detailed_results(suffix):
    filename = f'monte_carlo_all_data_onevrstwocompartment_geometric_mean{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        # Filter for valid Vdcalc > 0
        valid = df['Vdcalc'] > 0
        df = df[valid]
        
        auc_true = df['AUC_true'].values
        auc_calc = df['AUCcalc'].values  # PK-TR
        auc_bayes = df['AUC_fit_bayes'].values
        auc_diff_calc = auc_calc - auc_true
        auc_diff_bayes = auc_bayes - auc_true
        
        cl_true = df['Cl_total_true'].values
        cl_calc = df['Clcalc'].values
        cl_bayes = df['Cl_fit_bayes'].values
        cl_diff_calc = cl_true - cl_calc
        cl_diff_bayes = cl_true - cl_bayes
        
        # Bayesian fitted levels at trough (10h) and peak trough (2h)
        cp_10_bayes = df['Cp_10.0_fit_bayes'].values
        cp_2_bayes = df['Cp_2.0_fit_bayes'].values
        cp_diff_bayes = cp_10_bayes - cp_2_bayes  # Trough - peak trough
        
        pearson_r_calc, _ = stats.pearsonr(auc_true, auc_calc)
        pearson_r_bayes, _ = stats.pearsonr(auc_true, auc_bayes)
        
        return {
            'auc_diff_calc': auc_diff_calc,
            'auc_diff_bayes': auc_diff_bayes,
            'cl_diff_calc': cl_diff_calc,
            'cl_diff_bayes': cl_diff_bayes,
            'cp_diff_bayes': cp_diff_bayes,
            'auc_true': auc_true,
            'auc_calc': auc_calc,
            'auc_bayes': auc_bayes,
            'cp_10_bayes': cp_10_bayes,
            'cp_2_bayes': cp_2_bayes,
            'pearson_r_calc': pearson_r_calc,
            'pearson_r_bayes': pearson_r_bayes
        }
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

# Function to load statistics
def load_stats_results(suffix):
    filename = f'cl_differences_statistics_Geometric_Mean{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        stats_dict = {}
        for _, row in df.iterrows():
            stat = row['Statistic']
            value = row['Value']
            stats_dict[stat] = value
        return stats_dict
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return {}

# Function to load detailed Monte Carlo results for two-vs-two
def load_detailed_results_two_vrs_two(suffix):
    filename = f'monte_carlo_results_two_vrs_two{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        # Filter for valid Vd (assuming Vc + Vp > 0)
        valid = (df['Vc_fit'] + df['Vp_fit']) > 0
        df = df[valid]
        auc_true = df['AUC_true'].values
        auc_fit = df['AUC_fit'].values
        auc_diff = auc_fit - auc_true
        cl_true = df['Cl_total_true'].values
        cl_fit = df['Cl_total_fit'].values
        cl_diff = cl_true - cl_fit
        cp_2_fit = df['Cp_2_fit'].values
        cp_10_fit = df['Cp_10_fit'].values
        pearson_r, _ = stats.pearsonr(auc_true, auc_fit)
        return {
            'auc_diff': auc_diff,
            'cl_diff': cl_diff,
            'auc_true': auc_true,
            'auc_fit': auc_fit,
            'cp_2_fit': cp_2_fit,
            'cp_10_fit': cp_10_fit,
            'pearson_r': pearson_r
        }
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

# Function to load grid results for two-vs-two
def load_grid_results_two_vrs_two(suffix):
    filename = f'cl_diff_grid_two_vrs_two{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        results = {}
        for _, row in df.iterrows():
            levels = row['Levels']
            results[levels] = {
                'auc_rmse': row['AUC_diff RMSE'],
                'mean_auc_diff': row['Average AUC_diff'],
                'auc_lower_95ci': row['AUC Lower 95% CI'],
                'auc_upper_95ci': row['AUC Upper 95% CI']
            }
        return results
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return {}

def mcnemar_test(auc_true, auc_calc, auc_bayes):
    bins = [0, 400, 601, np.inf]
    bins_true = np.digitize(auc_true, bins) - 1
    bins_calc = np.digitize(auc_calc, bins) - 1
    bins_bayes = np.digitize(auc_bayes, bins) - 1
    calc_correct = (bins_true == bins_calc).astype(int)
    bayes_correct = (bins_true == bins_bayes).astype(int)
    table = np.zeros((2, 2))
    for i in range(len(calc_correct)):
        table[calc_correct[i], bayes_correct[i]] += 1
    result = mcnemar(table, exact=False)
    return result, table

# Main function
def main():
    if len(sys.argv) > 1:
        suffix = sys.argv[1]
    else:
        print("Enter the suffix for the data files (e.g., CltotalfourpointfiveTR):")
        sys.stdout.flush()
        suffix = input().strip()

    # Load data
    data = load_detailed_results(suffix)
    stats_data = load_stats_results(suffix)
    data_two_vrs_two = load_detailed_results_two_vrs_two(suffix)
    grid_two_vrs_two = load_grid_results_two_vrs_two(suffix)

    if data:
        print(f"=== Statistical Comparisons for PK-TR (1 compt) vs Bayesian PK-TR (2 compt), and 1-Compartment (1 compt) Bayesian PK-TR vs 2-Compartment (2 compt) Bayesian PK-TR Peak-Trough (Suffix: {suffix}) ===")

        auc_diff_calc = data['auc_diff_calc']
        auc_diff_bayes = data['auc_diff_bayes']
        cl_diff_calc = data['cl_diff_calc']
        cl_diff_bayes = data['cl_diff_bayes']
        cp_diff_bayes = data['cp_diff_bayes']
        auc_true = data['auc_true']
        auc_calc = data['auc_calc']
        auc_bayes = data['auc_bayes']

        # Section 1: PK-TR vs Bayesian
        print("\n=== PK-TR (1 compt) vs Bayesian PK-TR (2 compt) ===")
        t_auc, p_auc = stats.ttest_rel(auc_diff_calc, auc_diff_bayes)
        p_auc_corrected = min(p_auc * 6, 1.0)
        print(f"AUC Diff: PK-TR mean = {np.mean(auc_diff_calc):.2f}, Bayesian mean = {np.mean(auc_diff_bayes):.2f}")
        print(f"Paired t-test: t = {t_auc:.3f}, p = {p_auc:.3e} (Bonferroni corrected: {p_auc_corrected:.3e})")

        t_cl, p_cl = stats.ttest_rel(cl_diff_calc, cl_diff_bayes)
        p_cl_corrected = min(p_cl * 6, 1.0)
        print(f"Cl Diff: PK-TR mean = {np.mean(cl_diff_calc):.2f}, Bayesian mean = {np.mean(cl_diff_bayes):.2f}")
        print(f"Paired t-test: t = {t_cl:.3f}, p = {p_cl:.3e} (Bonferroni corrected: {p_cl_corrected:.3e})")

        rmse_auc_calc = np.sqrt(np.mean(auc_diff_calc**2))
        rmse_auc_bayes = np.sqrt(np.mean(auc_diff_bayes**2))
        print(f"AUC RMSE: PK-TR = {rmse_auc_calc:.2f}, Bayesian = {rmse_auc_bayes:.2f}")

        rmse_cl_calc = np.sqrt(np.mean(cl_diff_calc**2))
        rmse_cl_bayes = np.sqrt(np.mean(cl_diff_bayes**2))
        print(f"Cl RMSE: PK-TR = {rmse_cl_calc:.2f}, Bayesian = {rmse_cl_bayes:.2f}")

        ci_auc_calc = (np.mean(auc_diff_calc) - 1.96 * np.std(auc_diff_calc), np.mean(auc_diff_calc) + 1.96 * np.std(auc_diff_calc))
        ci_auc_bayes = (np.mean(auc_diff_bayes) - 1.96 * np.std(auc_diff_bayes), np.mean(auc_diff_bayes) + 1.96 * np.std(auc_diff_bayes))
        print(f"AUC 95% CI: PK-TR ({ci_auc_calc[0]:.2f}, {ci_auc_calc[1]:.2f}), Bayesian ({ci_auc_bayes[0]:.2f}, {ci_auc_bayes[1]:.2f})")

        ci_cl_calc = (np.mean(cl_diff_calc) - 1.96 * np.std(cl_diff_calc), np.mean(cl_diff_calc) + 1.96 * np.std(cl_diff_calc))
        ci_cl_bayes = (np.mean(cl_diff_bayes) - 1.96 * np.std(cl_diff_bayes), np.mean(cl_diff_bayes) + 1.96 * np.std(cl_diff_bayes))
        print(f"Cl 95% CI: PK-TR {ci_cl_calc[0]:.2f} to {ci_cl_calc[1]:.2f}, Bayesian {ci_cl_bayes[0]:.2f} to {ci_cl_bayes[1]:.2f}")

        print(f"AUC Pearson's r: PK-TR = {data['pearson_r_calc']:.3f}, Bayesian = {data['pearson_r_bayes']:.3f}")

        mcnemar_result, table = mcnemar_test(auc_true, auc_calc, auc_bayes)
        p_mcnemar_corrected = min(mcnemar_result.pvalue * 6, 1.0)
        print(f"McNemar's Test: chi2 = {mcnemar_result.statistic:.3f}, p = {mcnemar_result.pvalue:.3e} (Bonferroni corrected: {p_mcnemar_corrected:.3e})")
        print(f"Contingency Table:\n{table}")

        # Compute cp_diff for two-compartment if available
        cp_diff_two = None
        auc_diff_two = None
        auc_two = None
        auc_true_trunc = None
        auc_true_two_trunc = None
        if data_two_vrs_two:
            cp_diff_two_full = data_two_vrs_two['cp_10_fit'] - data_two_vrs_two['cp_2_fit']
            auc_diff_two_full = data_two_vrs_two['auc_diff']
            auc_two_full = data_two_vrs_two['auc_fit']
            auc_true_two = data_two_vrs_two['auc_true']
            # Take the same number of samples as the one-compartment data
            min_len = min(len(cp_diff_bayes), len(cp_diff_two_full))
            cp_diff_two = cp_diff_two_full[:min_len]
            cp_diff_bayes_trunc = cp_diff_bayes[:min_len]
            auc_diff_two = auc_diff_two_full[:min_len]
            auc_two = auc_two_full[:min_len]
            auc_true_trunc = auc_true[:min_len]
            auc_true_two_trunc = auc_true_two[:min_len]
            auc_bayes_trunc = data['auc_bayes'][:min_len]

        # Section 2: 1-Compartment Bayesian Peak-Trough vs 2-Compartment Bayesian Peak-Trough
        if cp_diff_two is not None:
            print("\n=== 1-Compartment (1 compt) Bayesian PK-TR Peak-Trough (2,10) vs 2-Compartment (2 compt) Bayesian PK-TR Peak-Trough (2,10) ===")
            mean_diff_one = np.mean(cp_diff_bayes_trunc)
            mean_diff_two = np.mean(cp_diff_two)
            print(f"Mean Cp Diff (Trough - Peak): 1-Compartment = {mean_diff_one:.2f}, 2-Compartment = {mean_diff_two:.2f}")

            # Paired t-test between the two differences
            t_cp_comp, p_cp_comp = stats.ttest_rel(cp_diff_bayes_trunc, cp_diff_two)
            p_cp_comp_corrected = min(p_cp_comp * 6, 1.0)
            print(f"Paired t-test (1-comp vs 2-comp): t = {t_cp_comp:.3f}, p = {p_cp_comp:.3e} (Bonferroni corrected: {p_cp_comp_corrected:.3e})")

            rmse_cp_one = np.sqrt(np.mean(cp_diff_bayes_trunc**2))
            rmse_cp_two = np.sqrt(np.mean(cp_diff_two**2))
            print(f"Cp RMSE (diff): 1-Compartment = {rmse_cp_one:.2f}, 2-Compartment = {rmse_cp_two:.2f}")

            ci_cp_one = (np.mean(cp_diff_bayes_trunc) - 1.96 * np.std(cp_diff_bayes_trunc), np.mean(cp_diff_bayes_trunc) + 1.96 * np.std(cp_diff_bayes_trunc))
            ci_cp_two = (np.mean(cp_diff_two) - 1.96 * np.std(cp_diff_two), np.mean(cp_diff_two) + 1.96 * np.std(cp_diff_two))
            print(f"Cp Diff 95% CI: 1-Compartment ({ci_cp_one[0]:.2f}, {ci_cp_one[1]:.2f}), 2-Compartment ({ci_cp_two[0]:.2f}, {ci_cp_two[1]:.2f})")

            # AUC comparison
            mean_auc_diff_one = np.mean(auc_bayes_trunc - auc_true_trunc)
            mean_auc_diff_two = np.mean(auc_two - auc_true_trunc)
            print(f"Mean AUC Diff: 1-Compartment = {mean_auc_diff_one:.2f}, 2-Compartment = {mean_auc_diff_two:.2f}")

            ci_auc_one = (mean_auc_diff_one - 1.96 * np.std(auc_bayes_trunc - auc_true_trunc), mean_auc_diff_one + 1.96 * np.std(auc_bayes_trunc - auc_true_trunc))
            ci_auc_two = (mean_auc_diff_two - 1.96 * np.std(auc_two - auc_true_two_trunc), mean_auc_diff_two + 1.96 * np.std(auc_two - auc_true_two_trunc))
            print(f"AUC 95% CI: 1-Compartment ({ci_auc_one[0]:.2f}, {ci_auc_one[1]:.2f}), 2-Compartment ({ci_auc_two[0]:.2f}, {ci_auc_two[1]:.2f})")
            print(f"Debug - 2-Compartment AUC diff std: {np.std(auc_two - auc_true_trunc):.2f}, n: {len(auc_two)}")  # Debug

            # McNemar's test for CDA
            mcnemar_result_auc, table_auc = mcnemar_test(auc_true_trunc, auc_bayes_trunc, auc_two)
            p_mcnemar_auc_corrected = min(mcnemar_result_auc.pvalue * 6, 1.0)
            print(f"McNemar's Test for CDA ===")
            print(f"Contingency Table:\n{table_auc}")
            print(f"McNemar's Test: chi2 = {mcnemar_result_auc.statistic:.3f}, p = {mcnemar_result_auc.pvalue:.3e} (Bonferroni corrected: {p_mcnemar_auc_corrected:.3e})")

        # Summary from stats file
        print("\n=== Summary from Stats File ===")
        if 'Bias Average Cl_diff_calc pk trough calc' in stats_data:
            print(f"Cl Bias PK-TR: {stats_data['Bias Average Cl_diff_calc pk trough calc']:.4f}")
        if 'RMSE Sqrt Average Cl_diff_calc_sq pk trough calc' in stats_data:
            print(f"Cl RMSE PK-TR: {stats_data['RMSE Sqrt Average Cl_diff_calc_sq pk trough calc']:.4f}")
        if 'Bias Average Cl_diff_bayes' in stats_data:
            print(f"Cl Bias Bayesian: {stats_data['Bias Average Cl_diff_bayes']:.4f}")
        if 'RMSE Sqrt Average Cl_diff_bayes_sq' in stats_data:
            print(f"Cl RMSE Bayesian: {stats_data['RMSE Sqrt Average Cl_diff_bayes_sq']:.4f}")

        # Section 3: One-Compartment Bayesian vs Two-Compartment Bayesian
        if data_two_vrs_two:
            print("\n=== One-Compartment (1 compt) Bayesian PK-TR vs Two-Compartment (2 compt) Bayesian PK-TR ===")
            auc_diff_one_bayes = data['auc_diff_bayes']
            auc_diff_two_bayes = data_two_vrs_two['auc_diff']
            print(f"One-Compartment Bayesian AUC Diff mean: {np.mean(auc_diff_one_bayes):.2f}")
            print(f"Two-Compartment Bayesian AUC Diff mean: {np.mean(auc_diff_two_bayes):.2f}")
            rmse_one = np.sqrt(np.mean(auc_diff_one_bayes**2))
            rmse_two = np.sqrt(np.mean(auc_diff_two_bayes**2))
            print(f"AUC RMSE: One-Compartment = {rmse_one:.2f}, Two-Compartment = {rmse_two:.2f}")
            print(f"Pearson's r: One-Compartment = {data['pearson_r_bayes']:.3f}, Two-Compartment = {data_two_vrs_two['pearson_r']:.3f}")
            # From grid - keep only the '2,10' level to avoid diluting analysis
            if grid_two_vrs_two:
                # if '10' in grid_two_vrs_two:
                #     print(f"Two-Compartment (Levels '10'): AUC RMSE = {grid_two_vrs_two['10']['auc_rmse']:.2f}, Mean AUC Diff = {grid_two_vrs_two['10']['mean_auc_diff']:.2f}")
                if '2,10' in grid_two_vrs_two:
                    print(f"Two-Compartment (Levels '2,10'): AUC RMSE = {grid_two_vrs_two['2,10']['auc_rmse']:.2f}, Mean AUC Diff = {grid_two_vrs_two['2,10']['mean_auc_diff']:.2f}")

if __name__ == '__main__':
    main()