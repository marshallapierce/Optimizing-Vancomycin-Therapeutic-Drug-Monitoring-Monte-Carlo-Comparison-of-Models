import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys
import matplotlib.pyplot as plt

# Create output directory
output_dir = 'merged_output'

def print_and_save(text, file):
    """Print to terminal and save to file"""
    print(text)
    file.write(text + '\n')

# Statistical comparisons for peak+trough models
# 1 compartment analytic peak trough vs 1 compartment Bayesian peak trough
# 1 compartment Bayesian peak trough vs 2 compartment Bayesian peak trough

def load_detailed_results():
    filename = f'{output_dir}/combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def load_detailed_results_two_vrs_two():
    filename = f'{output_dir}/combined_monte_carlo_results_two_vrs_two_peaktrough.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

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

def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return f"p = {p:.3f}"

def bland_altman_plot(true_vals, pred_vals, title, filename):
    mean_vals = (true_vals + pred_vals) / 2
    diff_vals = pred_vals - true_vals
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.5)
    plt.axhline(np.mean(diff_vals), color='red', linestyle='-.', label='Mean difference')
    plt.axhline(np.mean(diff_vals) + 1.96 * np.std(diff_vals), color='blue', linestyle='--', label='+1.96 SD')
    plt.axhline(np.mean(diff_vals) - 1.96 * np.std(diff_vals), color='blue', linestyle='--', label='-1.96 SD')
    plt.xlabel('Mean of True and Predicted AUC')
    plt.ylabel('Difference (Predicted - True) AUC')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def create_auc_comparison_csv(auc_true, auc_models, model_names, suffix):
    bins = [0, 400, 600, np.inf]
    bins_true = np.digitize(auc_true, bins) - 1
    lines = []
    lines.append("One Compartment & Two Compartment Peak-Trough Methods Versus True AUC")
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
    filename = f'{output_dir}/auc_comparison_grids_pk_tr_pk_tr_vr_pk_tr_combined.csv'
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"Comparison grids saved to {filename}")

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

def main():
    # Load combined data (no suffix needed)
    df_one = load_detailed_results()
    df_two = load_detailed_results_two_vrs_two()

    # Open output file for statistical results
    results_file = open(f'{output_dir}/statistical_comparison_results_pk_tr_vs_bayesian_combined.txt', 'w')

    try:
        if df_one is not None and df_two is not None:
            # Merge on 'Group', 'Crcl', and 'weight' to align simulations across conditions
            merged_df = pd.merge(df_one, df_two, on=['Group', 'Crcl', 'weight'], suffixes=('_one', '_two'))

            # Apply filters
            valid_one = merged_df['Vdcalc'] > 0
            valid_two = (merged_df['Vc_fit'] + merged_df['Vp_fit']) > 0
            valid = valid_one & valid_two
            df = merged_df[valid]

            # Extract metrics
            auc_true = df['AUC_true_one'].values
            auc_calc = df['AUCcalc'].values  # One-compartment analytic peak trough
            auc_bayes_one = df['AUC_fit_bayes_full'].values  # One-compartment Bayesian peak trough
            auc_bayes_two = df['AUC_fit'].values  # Two-compartment Bayesian peak trough

            # Calculate Cl differences (since no separate Cl statistics file exists)
            cl_true = df['Cl_total_true_one'].values
            cl_calc = df['Clcalc'].values
            cl_bayes_one = df['Cl_fit_bayes_full'].values
            cl_bayes_two = df['Cl_total_fit'].values

            # Generate Bland-Altman plots
            bland_altman_plot(auc_true, auc_calc, 'Bland-Altman: One-Compartment Analytic Peak-Trough AUC vs True AUC', f'{output_dir}/bland_altman_one_compt_analytic_pk_tr_combined.png')
            bland_altman_plot(auc_true, auc_bayes_one, 'Bland-Altman: One-Compartment Bayesian Peak-Trough AUC vs True AUC', f'{output_dir}/bland_altman_one_compt_bayes_pk_tr_combined.png')
            bland_altman_plot(auc_true, auc_bayes_two, 'Bland-Altman: Two-Compartment Bayesian Peak-Trough AUC vs True AUC', f'{output_dir}/bland_altman_two_compt_bayes_pk_tr_combined.png')

            # Create AUC comparison grids CSV
            auc_models = [auc_calc, auc_bayes_one, auc_bayes_two]
            model_names = ["One Compartment Analytic", "One Compartment Bayesian", "Two Compartment Bayesian"]
            create_auc_comparison_csv(auc_true, auc_models, model_names, "")

            # Calculate Pearson's r
            pearson_r_calc, _ = stats.pearsonr(auc_true, auc_calc)
            pearson_r_bayes_one, _ = stats.pearsonr(auc_true, auc_bayes_one)
            pearson_r_bayes_two, _ = stats.pearsonr(auc_true, auc_bayes_two)

            print_and_save(f"=== Peak-Trough Pharmacokinetic Model Comparisons (Combined Data) ===", results_file)
            print_and_save(f"Merged and filtered data: {len(df)} simulations", results_file)

            # Section 1: One-Compartment Analytic Peak-Trough vs One-Compartment Bayesian Peak-Trough
            print_and_save("\n=== One-Compartment Analytic Peak-Trough vs One-Compartment Bayesian Peak-Trough ===", results_file)

            # AUC comparison
            auc_diff_calc = auc_calc - auc_true
            auc_diff_bayes_one = auc_bayes_one - auc_true
            t_auc_1, p_auc_1 = stats.ttest_rel(auc_diff_calc, auc_diff_bayes_one)
            p_auc_1_corrected = min(p_auc_1 * 4, 1.0)  # Bonferroni for 4 tests
            print_and_save(f"AUC Diff: Analytic = {np.mean(auc_diff_calc):.2f}, Bayesian = {np.mean(auc_diff_bayes_one):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_auc_1:.3f}, {format_p_value(p_auc_1)} (Bonferroni corrected: {format_p_value(p_auc_1_corrected)})", results_file)

            # AUC 95% CI
            ci_lower_calc = np.mean(auc_diff_calc) - 1.96 * np.std(auc_diff_calc, ddof=1)
            ci_upper_calc = np.mean(auc_diff_calc) + 1.96 * np.std(auc_diff_calc, ddof=1)
            print_and_save(f"AUC 95% CI (Analytic): ({ci_lower_calc:.2f}, {ci_upper_calc:.2f})", results_file)

            ci_lower_bayes_one = np.mean(auc_diff_bayes_one) - 1.96 * np.std(auc_diff_bayes_one, ddof=1)
            ci_upper_bayes_one = np.mean(auc_diff_bayes_one) + 1.96 * np.std(auc_diff_bayes_one, ddof=1)
            print_and_save(f"AUC 95% CI (Bayesian): ({ci_lower_bayes_one:.2f}, {ci_upper_bayes_one:.2f})", results_file)

            # RMSE calculations
            auc_rmse_calc = np.sqrt(np.mean(auc_diff_calc**2))
            auc_rmse_bayes_one = np.sqrt(np.mean(auc_diff_bayes_one**2))
            print_and_save(f"AUC RMSE: Analytic = {auc_rmse_calc:.2f}, Bayesian = {auc_rmse_bayes_one:.2f}", results_file)

            # Percentage error for AUC
            perc_error_auc_calc = ((auc_calc / auc_true - 1) * 100)
            perc_error_auc_bayes_one = ((auc_bayes_one / auc_true - 1) * 100)
            mean_perc_auc_calc = np.mean(perc_error_auc_calc)
            mean_perc_auc_bayes_one = np.mean(perc_error_auc_bayes_one)
            rmse_perc_auc_calc = np.sqrt(np.mean(perc_error_auc_calc**2))
            rmse_perc_auc_bayes_one = np.sqrt(np.mean(perc_error_auc_bayes_one**2))
            print_and_save(f"AUC Percentage Error Mean: Analytic = {mean_perc_auc_calc:.2f}%, Bayesian = {mean_perc_auc_bayes_one:.2f}%", results_file)
            print_and_save(f"AUC Percentage Error RMSE: Analytic = {rmse_perc_auc_calc:.2f}%, Bayesian = {rmse_perc_auc_bayes_one:.2f}%", results_file)

            # Pearson's r for AUC
            print_and_save(f"AUC Pearson's r: Analytic = {pearson_r_calc:.3f}, Bayesian = {pearson_r_bayes_one:.3f}", results_file)

            # McNemar test
            result_mc_1, table_mc_1 = mcnemar_test(auc_true, auc_calc, auc_bayes_one)
            p_mc_1_corrected = min(result_mc_1.pvalue * 4, 1.0)
            print_and_save(f"McNemar test: statistic = {result_mc_1.statistic:.3f}, {format_p_value(result_mc_1.pvalue)} (Bonferroni corrected: {format_p_value(p_mc_1_corrected)})", results_file)
            print_and_save(f"Contingency table:\n{table_mc_1}", results_file)
            correct_analytic = int(table_mc_1[1, 0] + table_mc_1[1, 1])
            correct_bayes_one = int(table_mc_1[0, 1] + table_mc_1[1, 1])
            print_and_save(f"Correct classifications: Analytic = {correct_analytic}, Bayesian = {correct_bayes_one}", results_file)

            # Cl comparison
            cl_diff_calc = cl_calc - cl_true
            cl_diff_bayes_one = cl_bayes_one - cl_true
            t_cl_1, p_cl_1 = stats.ttest_rel(cl_diff_calc, cl_diff_bayes_one)
            p_cl_1_corrected = min(p_cl_1 * 4, 1.0)
            print_and_save(f"Cl Diff: Analytic = {np.mean(cl_diff_calc):.2f}, Bayesian = {np.mean(cl_diff_bayes_one):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_cl_1:.3f}, {format_p_value(p_cl_1)} (Bonferroni corrected: {format_p_value(p_cl_1_corrected)})", results_file)

            # Cl 95% CI
            ci_cl_lower_calc = np.mean(cl_diff_calc) - 1.96 * np.std(cl_diff_calc, ddof=1)
            ci_cl_upper_calc = np.mean(cl_diff_calc) + 1.96 * np.std(cl_diff_calc, ddof=1)
            print_and_save(f"Cl 95% CI (Analytic): ({ci_cl_lower_calc:.2f}, {ci_cl_upper_calc:.2f})", results_file)

            ci_cl_lower_bayes_one = np.mean(cl_diff_bayes_one) - 1.96 * np.std(cl_diff_bayes_one, ddof=1)
            ci_cl_upper_bayes_one = np.mean(cl_diff_bayes_one) + 1.96 * np.std(cl_diff_bayes_one, ddof=1)
            print_and_save(f"Cl 95% CI (Bayesian): ({ci_cl_lower_bayes_one:.2f}, {ci_cl_upper_bayes_one:.2f})", results_file)

            # Cl RMSE
            cl_rmse_calc = np.sqrt(np.mean(cl_diff_calc**2))
            cl_rmse_bayes_one = np.sqrt(np.mean(cl_diff_bayes_one**2))
            print_and_save(f"Cl RMSE: Analytic = {cl_rmse_calc:.2f}, Bayesian = {cl_rmse_bayes_one:.2f}", results_file)

            # Section 2: One-Compartment Bayesian Peak-Trough vs Two-Compartment Bayesian Peak-Trough
            print_and_save("\n=== One-Compartment Bayesian Peak-Trough vs Two-Compartment Bayesian Peak-Trough ===", results_file)

            # AUC comparison
            auc_diff_bayes_two = auc_bayes_two - auc_true
            t_auc_2, p_auc_2 = stats.ttest_rel(auc_diff_bayes_one, auc_diff_bayes_two)
            p_auc_2_corrected = min(p_auc_2 * 4, 1.0)
            print_and_save(f"AUC Diff: 1-Compt Bayes = {np.mean(auc_diff_bayes_one):.2f}, 2-Compt Bayes = {np.mean(auc_diff_bayes_two):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_auc_2:.3f}, {format_p_value(p_auc_2)} (Bonferroni corrected: {format_p_value(p_auc_2_corrected)})", results_file)

            # AUC 95% CI
            print_and_save(f"AUC 95% CI (1-Compt Bayes): ({ci_lower_bayes_one:.2f}, {ci_upper_bayes_one:.2f})", results_file)

            ci_lower_bayes_two = np.mean(auc_diff_bayes_two) - 1.96 * np.std(auc_diff_bayes_two, ddof=1)
            ci_upper_bayes_two = np.mean(auc_diff_bayes_two) + 1.96 * np.std(auc_diff_bayes_two, ddof=1)
            print_and_save(f"AUC 95% CI (2-Compt Bayes): ({ci_lower_bayes_two:.2f}, {ci_upper_bayes_two:.2f})", results_file)

            # RMSE calculations
            auc_rmse_bayes_two = np.sqrt(np.mean(auc_diff_bayes_two**2))
            print_and_save(f"AUC RMSE: 1-Compt Bayes = {auc_rmse_bayes_one:.2f}, 2-Compt Bayes = {auc_rmse_bayes_two:.2f}", results_file)

            # Percentage error for AUC
            perc_error_auc_bayes_two = ((auc_bayes_two / auc_true - 1) * 100)
            mean_perc_auc_bayes_two = np.mean(perc_error_auc_bayes_two)
            rmse_perc_auc_bayes_two = np.sqrt(np.mean(perc_error_auc_bayes_two**2))
            print_and_save(f"AUC Percentage Error Mean: 1-Compt Bayes = {mean_perc_auc_bayes_one:.2f}%, 2-Compt Bayes = {mean_perc_auc_bayes_two:.2f}%", results_file)
            print_and_save(f"AUC Percentage Error RMSE: 1-Compt Bayes = {rmse_perc_auc_bayes_one:.2f}%, 2-Compt Bayes = {rmse_perc_auc_bayes_two:.2f}%", results_file)

            # Pearson's r for AUC
            print_and_save(f"AUC Pearson's r: 1-Compt Bayes = {pearson_r_bayes_one:.3f}, 2-Compt Bayes = {pearson_r_bayes_two:.3f}", results_file)

            # McNemar test
            result_mc_2, table_mc_2 = mcnemar_test(auc_true, auc_bayes_one, auc_bayes_two)
            p_mc_2_corrected = min(result_mc_2.pvalue * 4, 1.0)
            print_and_save(f"McNemar test: statistic = {result_mc_2.statistic:.3f}, {format_p_value(result_mc_2.pvalue)} (Bonferroni corrected: {format_p_value(p_mc_2_corrected)})", results_file)
            print_and_save(f"Contingency table:\n{table_mc_2}", results_file)
            correct_bayes_one_2 = int(table_mc_2[1, 0] + table_mc_2[1, 1])
            correct_bayes_two = int(table_mc_2[0, 1] + table_mc_2[1, 1])
            print_and_save(f"Correct classifications: 1-Compt Bayes = {correct_bayes_one_2}, 2-Compt Bayes = {correct_bayes_two}", results_file)

            # Cl comparison
            cl_diff_bayes_two = cl_bayes_two - cl_true
            t_cl_2, p_cl_2 = stats.ttest_rel(cl_diff_bayes_one, cl_diff_bayes_two)
            p_cl_2_corrected = min(p_cl_2 * 4, 1.0)
            print_and_save(f"Cl Diff: 1-Compt Bayes = {np.mean(cl_diff_bayes_one):.2f}, 2-Compt Bayes = {np.mean(cl_diff_bayes_two):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_cl_2:.3f}, {format_p_value(p_cl_2)} (Bonferroni corrected: {format_p_value(p_cl_2_corrected)})", results_file)

            # Cl 95% CI
            print_and_save(f"Cl 95% CI (1-Compt Bayes): ({ci_cl_lower_bayes_one:.2f}, {ci_cl_upper_bayes_one:.2f})", results_file)

            ci_cl_lower_bayes_two = np.mean(cl_diff_bayes_two) - 1.96 * np.std(cl_diff_bayes_two, ddof=1)
            ci_cl_upper_bayes_two = np.mean(cl_diff_bayes_two) + 1.96 * np.std(cl_diff_bayes_two, ddof=1)
            print_and_save(f"Cl 95% CI (2-Compt Bayes): ({ci_cl_lower_bayes_two:.2f}, {ci_cl_upper_bayes_two:.2f})", results_file)

            # Cl RMSE
            cl_rmse_bayes_two = np.sqrt(np.mean(cl_diff_bayes_two**2))
            print_and_save(f"Cl RMSE: 1-Compt Bayes = {cl_rmse_bayes_one:.2f}, 2-Compt Bayes = {cl_rmse_bayes_two:.2f}", results_file)

    finally:
        results_file.close()

if __name__ == '__main__':
    main()