import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys

# Statistical comparisons for Bayesian trough vs Bayesian peak trough, Bayesian trough vs peak trough, etc.
# Adapted from statistical_comparisons_one_compt_pk_tr_vs_two_compt_pk_tr.py

# Function to load detailed Monte Carlo results
def load_detailed_results(suffix):
    filename = f'monte_carlo_all_data_onevrstwocompartment_geometric_mean{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

# Function to load detailed Monte Carlo results for two-vs-two
def load_detailed_results_two_vrs_two(suffix):
    filename = f'monte_carlo_results_two_vrs_two{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

# Function to load method data
def load_method_data(suffix, compartment, method):
    if compartment == 'one':
        df = load_detailed_results(suffix)
        if df is None:
            return None
        # Select columns based on method
        if method in ['bay_tr', 'bay_pk_tr']:
            auc_col = 'AUC_fit_bayes'
            cl_col = 'Cl_fit_bayes'
        elif method == 'pk_tr':
            auc_col = 'AUCcalc'
            cl_col = 'Clcalc'
        else:
            return None
        # Rename for consistency
        df = df.rename(columns={auc_col: f'AUC_{method}', cl_col: f'Cl_{method}'})
        return df[['Group', 'AUC_true', f'AUC_{method}', 'Cl_total_true', f'Cl_{method}']]
    elif compartment == 'two':
        df = load_detailed_results_two_vrs_two(suffix)
        if df is None:
            return None
        # For two compartment, assume bayesian
        df = df.rename(columns={'AUC_fit': f'AUC_{method}', 'Cl_total_fit': f'Cl_{method}', 'AUC_true_two': 'AUC_true', 'Cl_total_true_two': 'Cl_total_true'})
        return df[['Group', 'AUC_true', f'AUC_{method}', 'Cl_total_true', f'Cl_{method}']]
    else:
        return None

def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return f"p = {p:.3f}"

def mcnemar_test(auc_true, auc1, auc2):
    bins = [0, 400, 601, np.inf]
    bins_true = np.digitize(auc_true, bins) - 1
    bins1 = np.digitize(auc1, bins) - 1
    bins2 = np.digitize(auc2, bins) - 1
    correct1 = (bins_true == bins1).astype(int)
    correct2 = (bins_true == bins2).astype(int)
    table = np.zeros((2, 2))
    for i in range(len(correct1)):
        table[correct1[i], correct2[i]] += 1
    result = mcnemar(table, exact=False)
    return result, table

def create_auc_comparison_csv(auc_true, auc_models, model_names, suffix):
    bins = [0, 400, 600, np.inf]
    bins_true = np.digitize(auc_true, bins) - 1
    lines = []
    lines.append(f"{suffix} Versus True AUC")
    lines.append("")
    for auc_pred, name in zip(auc_models, model_names):
        bins_pred = np.digitize(auc_pred, bins) - 1
        table = np.zeros((3, 3))
        for i in range(len(bins_true)):
            table[bins_true[i], bins_pred[i]] += 1
        # Comparison Grid
        lines.append(f"{name} - Comparison Grid")
        lines.append("Actual AUC Ranges,Number Fit in AUC Ranges Below")
        lines.append("Number In AUC ranges,Total,< 400,400 through 600,> 600")
        ranges = ["< 400", "400 through 600", "> 600"]
        for r in range(3):
            total = int(np.sum(table[r, :]))
            c1 = int(table[r, 0])
            c2 = int(table[r, 1])
            c3 = int(table[r, 2])
            lines.append(f"{ranges[r]},{total},{c1},{c2},{c3}")
        lines.append("")
        # Percentage Grid
        lines.append(f"{name} - Percentage Grid")
        lines.append("Actual AUC Ranges,Percentage Fit in AUC Ranges Below")
        lines.append("Number In AUC ranges,Total,< 400,400 through 600,> 600")
        for r in range(3):
            total = np.sum(table[r, :])
            if total > 0:
                p1 = table[r, 0] / total * 100
                p2 = table[r, 1] / total * 100
                p3 = table[r, 2] / total * 100
                lines.append(f"{ranges[r]},{total:.0f},{p1:.1f}%,{p2:.1f}%,{p3:.1f}%")
            else:
                lines.append(f"{ranges[r]},0,0.0%,0.0%,0.0%")
        lines.append("")
        # Fraction correct
        correct = np.sum(np.diag(table))
        total_all = np.sum(table)
        frac = correct / total_all
        lines.append(f"Fraction of Correct Predictions,,,,,{frac:.4f}")
        # Additional percentages for middle bin
        if np.sum(table[1, :]) > 0:
            pct_over = table[1, 2] / np.sum(table[1, :]) * 100
            pct_under = table[1, 0] / np.sum(table[1, :]) * 100
            lines.append(f"% Predicted AUC >600 when True AUC 400-600,,,,,{pct_over:.2f}%")
            lines.append(f"% Predicted AUC <400 when True AUC 400-600,,,,,{pct_under:.2f}%")
        else:
            lines.append(f"% Predicted AUC >600 when True AUC 400-600,,,,,0.00%")
            lines.append(f"% Predicted AUC <400 when True AUC 400-600,,,,,0.00%")
        lines.append("")
    # Write to file
    filename = f'auc_comparison_grids_{suffix.replace(" ", "_")}.csv'
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"Comparison grids saved to {filename}")

# Main function
def main():
    # No need for suffix input, using fixed suffixes

    # Define the comparisons
    comparisons = [
        {
            'name': '1 compartment Bayesian trough vs 1 compartment Bayesian peak trough',
            'comp1': {'compartment': 'one', 'suffix': 'CltotalfourpointfiveTR', 'method': 'bay_tr'},
            'comp2': {'compartment': 'one', 'suffix': 'Cltotalfourpointfivepktr', 'method': 'bay_pk_tr'}
        },
        {
            'name': '1 compartment Bayesian trough vs 1 compartment peak trough',
            'comp1': {'compartment': 'one', 'suffix': 'CltotalfourpointfiveTR', 'method': 'bay_tr'},
            'comp2': {'compartment': 'one', 'suffix': 'Cltotalfourpointfivepktr', 'method': 'pk_tr'}
        },
        {
            'name': '2 compartment Bayesian trough vs 1 compartment Bayesian peak trough',
            'comp1': {'compartment': 'two', 'suffix': 'CltotalfourpointfiveTR', 'method': 'bay_tr'},
            'comp2': {'compartment': 'one', 'suffix': 'Cltotalfourpointfivepktr', 'method': 'bay_pk_tr'}
        },
        {
            'name': '2 compartment Bayesian trough vs 2 compartment Bayesian peak trough',
            'comp1': {'compartment': 'two', 'suffix': 'CltotalfourpointfiveTR', 'method': 'bay_tr'},
            'comp2': {'compartment': 'two', 'suffix': 'CltotalfourpointfivePKTR', 'method': 'bay_pk_tr'}
        }
    ]

    bonferroni_factor = 8  # Since 8 comparisons

    for comp in comparisons:
        print(f"\n=== {comp['name']} ===")

        # Load data for comp1
        df1 = load_method_data(comp['comp1']['suffix'], comp['comp1']['compartment'], comp['comp1']['method'])
        # Load data for comp2
        df2 = load_method_data(comp['comp2']['suffix'], comp['comp2']['compartment'], comp['comp2']['method'])

        if df1 is None or df2 is None:
            print("Data not found for this comparison.")
            continue

        # Merge on 'Group' assuming it exists
        merged_df = pd.merge(df1, df2, on='Group', suffixes=('_1', '_2'))

        # Assume AUC_true is the same, take from _1
        auc_true = merged_df['AUC_true_1'].values
        auc1 = merged_df[f'AUC_{comp["comp1"]["method"]}'].values
        auc2 = merged_df[f'AUC_{comp["comp2"]["method"]}'].values

        cl_true = merged_df['Cl_total_true_1'].values
        cl1 = merged_df[f'Cl_{comp["comp1"]["method"]}'].values
        cl2 = merged_df[f'Cl_{comp["comp2"]["method"]}'].values

        # Differences
        auc_diff1 = auc1 - auc_true
        auc_diff2 = auc2 - auc_true
        cl_diff1 = cl_true - cl1
        cl_diff2 = cl_true - cl2

        # Paired t-test for AUC diff
        t_auc, p_auc = stats.ttest_rel(auc_diff1, auc_diff2)
        p_auc_corrected = min(p_auc * bonferroni_factor, 1.0)
        print(f"AUC Diff: {comp['comp1']['method']} mean = {np.mean(auc_diff1):.2f}, {comp['comp2']['method']} mean = {np.mean(auc_diff2):.2f}")
        print(f"Paired t-test AUC: t = {t_auc:.3f}, {format_p_value(p_auc)} (Bonferroni corrected: {format_p_value(p_auc_corrected)})")

        # Paired t-test for Cl diff
        t_cl, p_cl = stats.ttest_rel(cl_diff1, cl_diff2)
        p_cl_corrected = min(p_cl * bonferroni_factor, 1.0)
        print(f"Cl Diff: {comp['comp1']['method']} mean = {np.mean(cl_diff1):.2f}, {comp['comp2']['method']} mean = {np.mean(cl_diff2):.2f}")
        print(f"Paired t-test Cl: t = {t_cl:.3f}, {format_p_value(p_cl)} (Bonferroni corrected: {format_p_value(p_cl_corrected)})")

        # RMSE
        rmse_auc1 = np.sqrt(np.mean(auc_diff1**2))
        rmse_auc2 = np.sqrt(np.mean(auc_diff2**2))
        print(f"AUC RMSE: {comp['comp1']['method']} = {rmse_auc1:.2f}, {comp['comp2']['method']} = {rmse_auc2:.2f}")

        rmse_cl1 = np.sqrt(np.mean(cl_diff1**2))
        rmse_cl2 = np.sqrt(np.mean(cl_diff2**2))
        print(f"Cl RMSE: {comp['comp1']['method']} = {rmse_cl1:.2f}, {comp['comp2']['method']} = {rmse_cl2:.2f}")

        # 95% CI for AUC
        ci_auc1 = (np.mean(auc_diff1) - 1.96 * np.std(auc_diff1), np.mean(auc_diff1) + 1.96 * np.std(auc_diff1))
        ci_auc2 = (np.mean(auc_diff2) - 1.96 * np.std(auc_diff2), np.mean(auc_diff2) + 1.96 * np.std(auc_diff2))
        print(f"AUC 95% CI: {comp['comp1']['method']} ({ci_auc1[0]:.2f}, {ci_auc1[1]:.2f}), {comp['comp2']['method']} ({ci_auc2[0]:.2f}, {ci_auc2[1]:.2f})")

        # 95% CI for Cl
        ci_cl1 = (np.mean(cl_diff1) - 1.96 * np.std(cl_diff1), np.mean(cl_diff1) + 1.96 * np.std(cl_diff1))
        ci_cl2 = (np.mean(cl_diff2) - 1.96 * np.std(cl_diff2), np.mean(cl_diff2) + 1.96 * np.std(cl_diff2))
        print(f"Cl 95% CI: {comp['comp1']['method']} ({ci_cl1[0]:.2f}, {ci_cl1[1]:.2f}), {comp['comp2']['method']} ({ci_cl2[0]:.2f}, {ci_cl2[1]:.2f})")

        # Pearson's r
        pearson_r1, _ = stats.pearsonr(auc_true, auc1)
        pearson_r2, _ = stats.pearsonr(auc_true, auc2)
        print(f"Pearson's r AUC: {comp['comp1']['method']} = {pearson_r1:.3f}, {comp['comp2']['method']} = {pearson_r2:.3f}")

        # McNemar's test
        mcnemar_result, table = mcnemar_test(auc_true, auc1, auc2)
        p_mcnemar_corrected = min(mcnemar_result.pvalue * bonferroni_factor, 1.0)
        print(f"McNemar's Test: chi2 = {mcnemar_result.statistic:.3f}, {format_p_value(mcnemar_result.pvalue)} (Bonferroni corrected: {format_p_value(p_mcnemar_corrected)})")
        print(f"Contingency Table:\n{table}")
        correct1 = int(table[1, 0] + table[1, 1])
        correct2 = int(table[0, 1] + table[1, 1])
        print(f"{comp['comp1']['method']} correct classifications: {correct1}, {comp['comp2']['method']} correct classifications: {correct2}")
        if correct1 > correct2:
            print(f"{comp['comp1']['method']} has more correct classifications than {comp['comp2']['method']}.")
        elif correct2 > correct1:
            print(f"{comp['comp2']['method']} has more correct classifications than {comp['comp1']['method']}.")
        else:
            print("Both methods have the same number of correct classifications.")

        # Create AUC comparison grids
        auc_models = [auc1, auc2]
        model_names = comp['name'].split(' vs ')
        create_auc_comparison_csv(auc_true, auc_models, model_names, comp['name'])

if __name__ == '__main__':
    main()