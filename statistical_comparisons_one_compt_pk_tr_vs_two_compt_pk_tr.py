import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys

# Statistical comparisons for 1 compartment non-bayesian Peak-trough vrs 2 compartment Bayesian Peak-Trough, 1 compartarment Bayesian peak-trough vrs 2 compartment Bayesian peak-trough
# Adapted from statistical_comparisons.py

# Function to load detailed Monte Carlo results and extract metrics
def load_detailed_results(suffix):
    filename = f'monte_carlo_all_data_onevrstwocompartment_geometric_mean{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        return df
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
        return df
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

def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return f"p = {p:.3f}"

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

def create_auc_comparison_csv(auc_true, auc_models, model_names, suffix):
    bins = [0, 400, 600, np.inf]
    bins_true = np.digitize(auc_true, bins) - 1
    lines = []
    lines.append("One Compartment & Two Compartment Peak Trough Methods Versus True AUC")
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
    filename = f'auc_comparison_grids_peak_trough{suffix}.csv'
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"Comparison grids saved to {filename}")

# Main function
def main():
    if len(sys.argv) > 1:
        suffix = sys.argv[1]
    else:
        print("Enter the suffix for the data files a peak and trough are required (e.g., CltotalfourpointfivePKTR):")
        sys.stdout.flush()
        suffix = input().strip()

    # Load data
    df_one = load_detailed_results(suffix)
    stats_data = load_stats_results(suffix)
    df_two = load_detailed_results_two_vrs_two(suffix)
    grid_two_vrs_two = load_grid_results_two_vrs_two(suffix)

    if df_one is not None and df_two is not None:
        # Merge on 'Group' to align simulations
        merged_df = pd.merge(df_one, df_two, on='Group', suffixes=('_one', '_two'))
        
        # Apply filters
        valid_one = merged_df['Vdcalc'] > 0
        valid_two = (merged_df['Vc_fit'] + merged_df['Vp_fit']) > 0
        valid = valid_one & valid_two
        df = merged_df[valid]
        
        # Extract metrics for Section 1
        auc_true = df['AUC_true_one'].values
        auc_calc = df['AUCcalc'].values  # PK-TR
        auc_bayes = df['AUC_fit_bayes'].values
        auc_diff_calc = auc_calc - auc_true
        auc_diff_bayes = auc_bayes - auc_true
        
        cl_true = df['Cl_total_true_one'].values
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
        
        data = {
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
        
        # Extract metrics for two-vs-two
        auc_true_two = df['AUC_true_two'].values
        auc_fit_two = df['AUC_fit'].values
        
        # Create AUC comparison grids CSV
        auc_models = [auc_calc, auc_bayes, auc_fit_two]
        model_names = ["One Compartment Peak Trough", "One Compartment Bayesian Peak Trough", "Two Compartment Bayesian Peak Trough"]
        create_auc_comparison_csv(auc_true, auc_models, model_names, suffix)
        
        auc_diff_two = auc_fit_two - auc_true_two
        cl_true_two = df['Cl_total_true_two'].values
        cl_fit_two = df['Cl_total_fit'].values
        cl_diff_two = cl_true_two - cl_fit_two
        cp_2_fit = df['Cp_2_fit'].values
        cp_10_fit = df['Cp_10_fit'].values
        pearson_r_two, _ = stats.pearsonr(auc_true_two, auc_fit_two)
        
        data_two_vrs_two = {
            'auc_diff': auc_diff_two,
            'cl_diff': cl_diff_two,
            'auc_true': auc_true_two,
            'auc_fit': auc_fit_two,
            'cp_2_fit': cp_2_fit,
            'cp_10_fit': cp_10_fit,
            'pearson_r': pearson_r_two
        }
        
        print(f"=== Statistical Comparisons for PK-TR (1 compt) vs Bayesian PK-TR (2 compt), and 1-Compartment (1 compt) Bayesian PK-TR vs 2-Compartment (2 compt) Bayesian PK-TR Peak-Trough (Suffix: {suffix}) ===")
        print(f"Merged and filtered data: {len(df)} simulations")

        # Section 1: PK-TR vs Bayesian
        print("\n=== PK-TR (1 compt) vs Bayesian PK-TR (2 compt) ===")
        t_auc, p_auc = stats.ttest_rel(auc_diff_calc, auc_diff_bayes)
        p_auc_corrected = min(p_auc * 8, 1.0)
        print(f"AUC Diff: PK-TR mean = {np.mean(auc_diff_calc):.2f}, Bayesian mean = {np.mean(auc_diff_bayes):.2f}")
        print(f"Paired t-test: t = {t_auc:.3f}, {format_p_value(p_auc)} (Bonferroni corrected: {format_p_value(p_auc_corrected)})")

        t_cl, p_cl = stats.ttest_rel(cl_diff_calc, cl_diff_bayes)
        p_cl_corrected = min(p_cl * 8, 1.0)
        print(f"Cl Diff: PK-TR mean = {np.mean(cl_diff_calc):.2f}, Bayesian mean = {np.mean(cl_diff_bayes):.2f}")
        print(f"Paired t-test: t = {t_cl:.3f}, {format_p_value(p_cl)} (Bonferroni corrected: {format_p_value(p_cl_corrected)})")

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
        p_mcnemar_corrected = min(mcnemar_result.pvalue * 8, 1.0)
        print(f"McNemar's Test: chi2 = {mcnemar_result.statistic:.3f}, {format_p_value(mcnemar_result.pvalue)} (Bonferroni corrected: {format_p_value(p_mcnemar_corrected)})")
        print(f"Contingency Table:\n{table}")
        correct_pktr = int(table[1, 0] + table[1, 1])
        correct_bayes = int(table[0, 1] + table[1, 1])
        print(f"PK-TR correct classifications: {correct_pktr}, Bayesian correct classifications: {correct_bayes}")
        if correct_pktr > correct_bayes:
            print(f"PK-TR has more correct classifications than Bayesian.")
        elif correct_bayes > correct_pktr:
            print(f"Bayesian has more correct classifications than PK-TR.")
        else:
            print(f"Both methods have the same number of correct classifications.")

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

        # Section 2: 1-Compartment Bayesian Peak-Trough vs 2-Compartment Bayesian Peak-Trough
        if data_two_vrs_two:
            print("\n=== 1-Compartment (1 compt) Bayesian PK-TR Peak-Trough (2,10) vs 2-Compartment (2 compt) Bayesian PK-TR Peak-Trough (2,10) (Paired data set) ===")
            cp_diff_bayes_trunc = cp_diff_bayes  # Already aligned
            cp_diff_two = data_two_vrs_two['cp_10_fit'] - data_two_vrs_two['cp_2_fit']
            auc_bayes_trunc = auc_bayes
            auc_two = data_two_vrs_two['auc_fit']
            auc_true_trunc = auc_true
            auc_diff_two = data_two_vrs_two['auc_diff']
            mean_diff_one = np.mean(cp_diff_bayes_trunc)
            mean_diff_two = np.mean(cp_diff_two)
            print(f"Mean Cp Diff (Trough - Peak): 1-Compartment = {mean_diff_one:.2f}, 2-Compartment = {mean_diff_two:.2f}")

            # Paired t-test between the two differences
            t_cp_comp, p_cp_comp = stats.ttest_rel(cp_diff_bayes_trunc, cp_diff_two)
            p_cp_comp_corrected = min(p_cp_comp * 8, 1.0)
            print(f"Paired t-test (1-comp vs 2-comp): t = {t_cp_comp:.3f}, {format_p_value(p_cp_comp)} (Bonferroni corrected: {format_p_value(p_cp_comp_corrected)})")

            # Paired t-test for AUC diff
            auc_diff_one_sec2 = auc_bayes_trunc - auc_true_trunc
            auc_diff_two_sec2 = auc_two - auc_true_trunc
            t_auc_sec2, p_auc_sec2 = stats.ttest_rel(auc_diff_one_sec2, auc_diff_two_sec2)
            p_auc_sec2_corrected = min(p_auc_sec2 * 8, 1.0)
            print(f"Paired t-test AUC Diff (1-comp vs 2-comp): t = {t_auc_sec2:.3f}, {format_p_value(p_auc_sec2)} (Bonferroni corrected: {format_p_value(p_auc_sec2_corrected)})")

            # Paired t-test for Cl diff
            cl_diff_one_sec2 = cl_diff_bayes
            cl_diff_two_sec2 = cl_diff_two
            t_cl_sec2, p_cl_sec2 = stats.ttest_rel(cl_diff_one_sec2, cl_diff_two_sec2)
            p_cl_sec2_corrected = min(p_cl_sec2 * 8, 1.0)
            print(f"Paired t-test Cl Diff (1-comp vs 2-comp): t = {t_cl_sec2:.3f}, {format_p_value(p_cl_sec2)} (Bonferroni corrected: {format_p_value(p_cl_sec2_corrected)})")

            rmse_cp_one = np.sqrt(np.mean(cp_diff_bayes_trunc**2))
            rmse_cp_two = np.sqrt(np.mean(cp_diff_two**2))
            print(f"Cp RMSE (diff): 1-Compartment = {rmse_cp_one:.2f}, 2-Compartment = {rmse_cp_two:.2f}")

            rmse_auc_one_sec2 = np.sqrt(np.mean((auc_bayes_trunc - auc_true_trunc)**2))
            rmse_auc_two_sec2 = np.sqrt(np.mean((auc_two - auc_true_trunc)**2))
            print(f"AUC RMSE: 1-Compartment = {rmse_auc_one_sec2:.2f}, 2-Compartment = {rmse_auc_two_sec2:.2f}")

            ci_cp_one = (np.mean(cp_diff_bayes_trunc) - 1.96 * np.std(cp_diff_bayes_trunc), np.mean(cp_diff_bayes_trunc) + 1.96 * np.std(cp_diff_bayes_trunc))
            ci_cp_two = (np.mean(cp_diff_two) - 1.96 * np.std(cp_diff_two), np.mean(cp_diff_two) + 1.96 * np.std(cp_diff_two))
            print(f"Cp Diff 95% CI: 1-Compartment ({ci_cp_one[0]:.2f}, {ci_cp_one[1]:.2f}), 2-Compartment ({ci_cp_two[0]:.2f}, {ci_cp_two[1]:.2f})")

            # AUC comparison
            mean_auc_diff_one = np.mean(auc_bayes_trunc - auc_true_trunc)
            mean_auc_diff_two = np.mean(auc_two - auc_true_trunc)
            print(f"Mean AUC Diff: 1-Compartment = {mean_auc_diff_one:.2f}, 2-Compartment = {mean_auc_diff_two:.2f}")

            # CI for 1-compartment using truncated data
            ci_auc_one = (mean_auc_diff_one - 1.96 * np.std(auc_bayes_trunc - auc_true_trunc), mean_auc_diff_one + 1.96 * np.std(auc_bayes_trunc - auc_true_trunc))
            # CI for 2-compartment using full data for consistency with grid
            mean_auc_diff_two_full = np.mean(auc_diff_two)
            std_auc_diff_two_full = np.std(auc_diff_two)
            ci_auc_two = (mean_auc_diff_two_full - 1.96 * std_auc_diff_two_full, mean_auc_diff_two_full + 1.96 * std_auc_diff_two_full)
            print(f"AUC 95% CI: 1-Compartment ({ci_auc_one[0]:.2f}, {ci_auc_one[1]:.2f}), 2-Compartment ({ci_auc_two[0]:.2f}, {ci_auc_two[1]:.2f})")

            # McNemar's test for CDA on paired data
            mcnemar_result_sec2, table_sec2 = mcnemar_test(auc_true_trunc, auc_bayes_trunc, auc_two)
            p_mcnemar_sec2_corrected = min(mcnemar_result_sec2.pvalue * 8, 1.0)
            print(f"McNemar's Test for CDA (Paired Data) ===")
            print(f"Contingency Table:\n{table_sec2}")
            print(f"McNemar's Test: chi2 = {mcnemar_result_sec2.statistic:.3f}, {format_p_value(mcnemar_result_sec2.pvalue)} (Bonferroni corrected: {format_p_value(p_mcnemar_sec2_corrected)})")
            correct_one_compt_sec2 = int(table_sec2[1, 0] + table_sec2[1, 1])
            correct_two_compt_sec2 = int(table_sec2[0, 1] + table_sec2[1, 1])
            print(f"1-Compartment Bayesian correct classifications: {correct_one_compt_sec2}, 2-Compartment Bayesian correct classifications: {correct_two_compt_sec2}")
            if correct_one_compt_sec2 > correct_two_compt_sec2:
                print(f"1-Compartment Bayesian has more correct classifications than 2-Compartment Bayesian.")
            elif correct_two_compt_sec2 > correct_one_compt_sec2:
                print(f"2-Compartment Bayesian has more correct classifications than 1-Compartment Bayesian.")
            else:
                print(f"Both methods have the same number of correct classifications.")

        # Section 3: One-Compartment Bayesian vs Two-Compartment Bayesian
        if data_two_vrs_two:
            print("\n=== One-Compartment (1 compt) Bayesian PK-TR vs Two-Compartment (2 compt) Bayesian PK-TR (Full paired data set) ===")
            print(f"Paired data length: {len(df)}")
            
            # Compute full correct counts for reference
            bins = [0, 400, 601, np.inf]
            # For one-compartment
            bins_true_one = np.digitize(auc_true, bins) - 1
            bins_one = np.digitize(auc_bayes, bins) - 1
            correct_one_full = np.sum(bins_true_one == bins_one)
            # For two-compartment
            bins_true_two = np.digitize(auc_true_two, bins) - 1
            bins_two = np.digitize(auc_fit_two, bins) - 1
            correct_two_full = np.sum(bins_true_two == bins_two)
            print(f"Full paired correct counts: 1-Compartment = {correct_one_full}, 2-Compartment = {correct_two_full}")
            
            print(f"One-Compartment Bayesian AUC Diff mean: {np.mean(auc_diff_bayes):.2f}")
            print(f"Two-Compartment Bayesian AUC Diff mean: {np.mean(auc_diff_two):.2f}")
            
            # AUC 95% CI
            ci_auc_one = (np.mean(auc_diff_bayes) - 1.96 * np.std(auc_diff_bayes), np.mean(auc_diff_bayes) + 1.96 * np.std(auc_diff_bayes))
            ci_auc_two = (np.mean(auc_diff_two) - 1.96 * np.std(auc_diff_two), np.mean(auc_diff_two) + 1.96 * np.std(auc_diff_two))
            print(f"AUC 95% CI: One-Compartment ({ci_auc_one[0]:.2f}, {ci_auc_one[1]:.2f}), Two-Compartment ({ci_auc_two[0]:.2f}, {ci_auc_two[1]:.2f})")
            
            # Paired t-test
            t_auc_comp, p_auc_comp = stats.ttest_rel(auc_diff_bayes, auc_diff_two)
            p_auc_comp_corrected = min(p_auc_comp * 8, 1.0)
            print(f"Mean AUC Diff: One-Compartment = {np.mean(auc_diff_bayes):.2f}, Two-Compartment = {np.mean(auc_diff_two):.2f}")
            print(f"Paired t-test AUC Diff (1-comp vs 2-comp): t = {t_auc_comp:.3f}, {format_p_value(p_auc_comp)} (Bonferroni corrected: {format_p_value(p_auc_comp_corrected)})")
            
            # Paired t-test for Cl diff
            t_cl_comp, p_cl_comp = stats.ttest_rel(cl_diff_bayes, cl_diff_two)
            p_cl_comp_corrected = min(p_cl_comp * 8, 1.0)
            print(f"Paired t-test Cl Diff (1-comp vs 2-comp): t = {t_cl_comp:.3f}, {format_p_value(p_cl_comp)} (Bonferroni corrected: {format_p_value(p_cl_comp_corrected)})")
            
            # Cl Diff
            print(f"Cl Diff: One-Compartment mean = {np.mean(cl_diff_bayes):.2f}, Two-Compartment mean = {np.mean(cl_diff_two):.2f}")
            
            # Cl 95% CI
            ci_cl_one = (np.mean(cl_diff_bayes) - 1.96 * np.std(cl_diff_bayes), np.mean(cl_diff_bayes) + 1.96 * np.std(cl_diff_bayes))
            ci_cl_two = (np.mean(cl_diff_two) - 1.96 * np.std(cl_diff_two), np.mean(cl_diff_two) + 1.96 * np.std(cl_diff_two))
            print(f"Cl 95% CI: One-Compartment ({ci_cl_one[0]:.2f}, {ci_cl_one[1]:.2f}), Two-Compartment ({ci_cl_two[0]:.2f}, {ci_cl_two[1]:.2f})")
            
            # Cl RMSE
            rmse_cl_one = np.sqrt(np.mean(cl_diff_bayes**2))
            rmse_cl_two = np.sqrt(np.mean(cl_diff_two**2))
            print(f"Cl RMSE: One-Compartment = {rmse_cl_one:.2f}, Two-Compartment = {rmse_cl_two:.2f}")
            
            rmse_one = np.sqrt(np.mean(auc_diff_bayes**2))
            rmse_two = np.sqrt(np.mean(auc_diff_two**2))
            print(f"AUC RMSE: One-Compartment = {rmse_one:.2f}, Two-Compartment = {rmse_two:.2f}")
            print(f"Pearson's r: One-Compartment = {data['pearson_r_bayes']:.3f}, Two-Compartment = {data_two_vrs_two['pearson_r']:.3f}")
            
            # McNemar's test for CDA on paired data
            mcnemar_result_full, table_full = mcnemar_test(auc_true, auc_bayes, auc_fit_two)
            p_mcnemar_full_corrected = min(mcnemar_result_full.pvalue * 8, 1.0)
            print(f"McNemar's Test for CDA (Paired Data) ===")
            print(f"Contingency Table:\n{table_full}")
            print(f"McNemar's Test: chi2 = {mcnemar_result_full.statistic:.3f}, {format_p_value(mcnemar_result_full.pvalue)} (Bonferroni corrected: {format_p_value(p_mcnemar_full_corrected)})")
            correct_one_compt_full = int(table_full[1, 0] + table_full[1, 1])
            correct_two_compt_full = int(table_full[0, 1] + table_full[1, 1])
            print(f"1-Compartment Bayesian correct classifications: {correct_one_compt_full}, 2-Compartment Bayesian correct classifications: {correct_two_compt_full}")
            if correct_one_compt_full > correct_two_compt_full:
                print(f"1-Compartment Bayesian has more correct classifications than 2-Compartment Bayesian.")
            elif correct_two_compt_full > correct_one_compt_full:
                print(f"2-Compartment Bayesian has more correct classifications than 1-Compartment Bayesian.")
            else:
                print(f"Both methods have the same number of correct classifications.")
            if grid_two_vrs_two:
                # if '10' in grid_two_vrs_two:
                #     print(f"Two-Compartment (Levels '10'): AUC RMSE = {grid_two_vrs_two['10']['auc_rmse']:.2f}, Mean AUC Diff = {grid_two_vrs_two['10']['mean_auc_diff']:.2f}")
                if '2,10' in grid_two_vrs_two:
                    print(f"Two-Compartment (Levels '2,10') data from Cl_diff_grid_two_vrs_two{suffix}.csv: AUC RMSE = {grid_two_vrs_two['2,10']['auc_rmse']:.2f}, Mean AUC Diff = {grid_two_vrs_two['2,10']['mean_auc_diff']:.2f}")

if __name__ == '__main__':
    main()