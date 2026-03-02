import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys

# Function to load detailed Monte Carlo results and extract metrics
def load_detailed_results(suffix, scenario_label):
    filename = f'monte_carlo_results_two_vrs_two{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        auc_true = df['AUC_true'].values
        auc_fit = df['AUC_fit'].values
        auc_diff = auc_fit - auc_true
        cl_true = df['Cl_total_true'].values
        cl_fit = df['Cl_total_fit'].values
        cl_diff = cl_true - cl_fit
        pearson_r, _ = stats.pearsonr(auc_true, auc_fit)
        return {
            'auc_diff': auc_diff,
            'cl_diff': cl_diff,
            'auc_true': auc_true,
            'auc_fit': auc_fit,
            'pearson_r': pearson_r,
            'scenario': scenario_label
        }
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping {scenario_label}.")
        return None

# Function to load grid results for summary metrics
def load_grid_results(suffix):
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

# Function to perform McNemar's test using detailed data
def mcnemar_from_detailed(trough_data, peaktrough_data):
    if trough_data is None or peaktrough_data is None:
        return None
    # Define bins
    bins = [0, 400, 601, np.inf]
    # Get bin indices for trough
    trough_bins_true = np.digitize(trough_data['auc_true'], bins) - 1
    trough_bins_fit = np.digitize(trough_data['auc_fit'], bins) - 1
    trough_correct = (trough_bins_true == trough_bins_fit).astype(int)
    # For peak-trough
    peaktrough_bins_true = np.digitize(peaktrough_data['auc_true'], bins) - 1
    peaktrough_bins_fit = np.digitize(peaktrough_data['auc_fit'], bins) - 1
    peaktrough_correct = (peaktrough_bins_true == peaktrough_bins_fit).astype(int)
    # Contingency table: trough correct/incorrect vs peak-trough correct/incorrect
    table = np.zeros((2, 2))
    for i in range(len(trough_correct)):
        table[trough_correct[i], peaktrough_correct[i]] += 1
    # McNemar's test
    result = mcnemar(table, exact=False)
    return result, table

# Main function
def main():
    import sys
    if len(sys.argv) > 2:
        trough_suffix = sys.argv[1]
        peaktrough_suffix = sys.argv[2]
    else:
        print("Enter the suffix for the two compartment versus two compartment trough scenario (e.g., 10):")
        sys.stdout.flush()
        trough_suffix = input().strip()
        print("Enter the suffix for the two compartment versus two compartment peak-trough scenario (e.g., 2,10):")
        sys.stdout.flush()
        peaktrough_suffix = input().strip()


    # Load detailed data
    trough_data = load_detailed_results(trough_suffix, 'Trough')
    peaktrough_data = load_detailed_results(peaktrough_suffix, 'Peak-Trough')

    if trough_data and peaktrough_data:
        # Continuous metrics comparisons
        print("=== Continuous Metrics Comparisons (Paired t-Tests) ===")

        auc_diff_trough = trough_data['auc_diff']
        auc_diff_peaktrough = peaktrough_data['auc_diff']

        # Paired t-test for mean AUC difference
        t_stat_auc_diff, p_auc_diff = stats.ttest_rel(auc_diff_trough, auc_diff_peaktrough)
        print(f"Mean AUC Difference: Trough mean = {np.mean(auc_diff_trough):.2f}, Peak-Trough mean = {np.mean(auc_diff_peaktrough):.2f}")
        print(f"Paired t-test: t = {t_stat_auc_diff:.3f}, p = {p_auc_diff:.3e}")

        # AUC RMSE comparison
        rmse_trough = np.sqrt(np.mean(auc_diff_trough**2))
        rmse_peaktrough = np.sqrt(np.mean(auc_diff_peaktrough**2))
        print(f"AUC RMSE: Trough = {rmse_trough:.2f}, Peak-Trough = {rmse_peaktrough:.2f}")

        # AUC 95% CI
        auc_lower_trough = np.mean(auc_diff_trough) - 1.96 * np.std(auc_diff_trough)
        auc_upper_trough = np.mean(auc_diff_trough) + 1.96 * np.std(auc_diff_trough)
        auc_lower_peaktrough = np.mean(auc_diff_peaktrough) - 1.96 * np.std(auc_diff_peaktrough)
        auc_upper_peaktrough = np.mean(auc_diff_peaktrough) + 1.96 * np.std(auc_diff_peaktrough)
        print(f"AUC 95% CI: Trough ({auc_lower_trough:.2f}, {auc_upper_trough:.2f}), Peak-Trough ({auc_lower_peaktrough:.2f}, {auc_upper_peaktrough:.2f})")

        # Pearson's r
        r_trough = trough_data['pearson_r']
        r_peaktrough = peaktrough_data['pearson_r']
        print(f"Pearson's r: Trough = {r_trough:.3f}, Peak-Trough = {r_peaktrough:.3f}")

        # McNemar's test
        print("\n=== McNemar's Test for CDA ===")
        mcnemar_result, table = mcnemar_from_detailed(trough_data, peaktrough_data)
        if mcnemar_result:
            print(f"Contingency Table:\n{table}")
            print(f"McNemar's Test: chi2 = {mcnemar_result.statistic:.3f}, p = {mcnemar_result.pvalue:.3e}")
        else:
            print("Could not perform McNemar's test due to missing data.")

    # Load grid results for reference
    grid_trough = load_grid_results(trough_suffix)
    grid_peaktrough = load_grid_results(peaktrough_suffix)
    print("\n=== Summary from Grids ===")
    if '10' in grid_trough:
        print(f"Trough (Levels '10'): AUC RMSE = {grid_trough['10']['auc_rmse']:.2f}, Mean AUC Diff = {grid_trough['10']['mean_auc_diff']:.2f}")
    if '2,10' in grid_peaktrough:
        print(f"Peak-Trough (Levels '2,10'): AUC RMSE = {grid_peaktrough['2,10']['auc_rmse']:.2f}, Mean AUC Diff = {grid_peaktrough['2,10']['mean_auc_diff']:.2f}")

if __name__ == '__main__':
    main()