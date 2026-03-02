import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Create output directory
output_dir = 'merged_output'

def print_and_save(text, file):
    """Print to terminal and save to file"""
    print(text)
    file.write(text + '\n')

# Statistical comparisons for peak-trough pharmacokinetic models
# 1-compartment analytic peak-trough vs 1-compartment Bayesian peak-trough
# 1-compartment Bayesian peak-trough vs 2-compartment Bayesian peak-trough

def load_detailed_results(suffix):
    filename = f'merged_output/combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def load_detailed_results_two_vrs_two(suffix):
    filename = f'merged_output/combined_monte_carlo_results_two_vrs_two_peaktrough.csv'
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

def main():
    # Load combined data (no suffix needed)
    df_one = load_detailed_results("")
    df_two = load_detailed_results_two_vrs_two("")
    
    print(f"Loaded df_one: {len(df_one) if df_one is not None else 'None'}")
    print(f"Loaded df_two: {len(df_two) if df_two is not None else 'None'}")
    
    # Open output file for statistical results
    results_file = open(f'{output_dir}/statistical_comparison_results_pk_tr_vr_pk_tr_combined.txt', 'w')
    
    try:
        if df_one is not None and df_two is not None:
            print("Merging dataframes...")
            # Merge on 'Group', 'Crcl', and 'weight' to align simulations across conditions
            merged_df = pd.merge(df_one, df_two, on=['Group', 'Crcl', 'weight'], suffixes=('_one', '_two'))
            print(f"Merged df length: {len(merged_df)}")
            
            # Apply filters
            valid_one = merged_df['Vdcalc'] > 0
            valid_two = (merged_df['Vc_fit'] + merged_df['Vp_fit']) > 0
            valid = valid_one & valid_two
            df = merged_df[valid]
            print(f"Filtered df length: {len(df)}")
            
            if len(df) == 0:
                print("No valid data after filtering. Exiting.")
                return
            
            print("Data loaded successfully. Now processing...")
            
            # Extract metrics
            auc_true = df['AUC_true_one'].values
            auc_calc = df['AUCcalc'].values  # PK-TR analytic
            auc_bayes = df['AUC_fit_bayes_full'].values  # 1-compt Bayesian peak-trough
            auc_fit_two = df['AUC_fit'].values  # 2-compt Bayesian peak-trough
            
            print(f"AUC true shape: {auc_true.shape}")
            print("Starting statistical analysis...")
            
            # Create AUC comparison grids CSV
            auc_models = [auc_calc, auc_bayes, auc_fit_two]
            model_names = ["Analytic Peak-Trough", "One Compartment Bayesian Peak-Trough", "Two Compartment Bayesian Peak-Trough"]
            create_auc_comparison_csv(auc_true, auc_models, model_names, "")
            
            print("CSV created. Starting main analysis...")
            
            # Rest of the code...
            
            cl_true = df['Cl_total_true_one'].values
            cl_calc = df['Clcalc'].values
            cl_bayes = df['Cl_fit_bayes_full'].values
            cl_fit_two = df['Cl_total_fit'].values
            
            cp_10_calc = df['Cp_10.0_fit_fixed'].values  # Assuming this is for analytic, but may need adjustment
            cp_10_bayes = df['Cp_10.0_fit_bayes_full'].values  # 1-compt Bayesian peak-trough
            cp_10_fit_two = df['Cp_10_fit'].values  # 2-compt Bayesian peak-trough
            
            pearson_r_calc, _ = stats.pearsonr(auc_true, auc_calc)
            pearson_r_bayes, _ = stats.pearsonr(auc_true, auc_bayes)
            pearson_r_two, _ = stats.pearsonr(auc_true, auc_fit_two)
            
            print_and_save(f"=== Peak-Trough Pharmacokinetic Model Comparisons (Combined Data) ===", results_file)
            print_and_save(f"Merged and filtered data: {len(df)} simulations", results_file)

            # Section 1: One-Compartment Analytic Peak-Trough vs One-Compartment Bayesian Peak-Trough
            print_and_save("\n=== One-Compartment Analytic Peak-Trough vs One-Compartment Bayesian Peak-Trough ===", results_file)

            # AUC comparison
            auc_diff_calc = auc_calc - auc_true
            auc_diff_bayes = auc_bayes - auc_true
            t_auc_cb, p_auc_cb = stats.ttest_rel(auc_diff_calc, auc_diff_bayes)
            p_auc_cb_corrected = min(p_auc_cb * 8, 1.0)  # Bonferroni for 8 tests
            print_and_save(f"AUC Diff: Analytic = {np.mean(auc_diff_calc):.2f}, Bayesian = {np.mean(auc_diff_bayes):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_auc_cb:.3f}, {format_p_value(p_auc_cb)} (Bonferroni corrected: {format_p_value(p_auc_cb_corrected)})", results_file)

            # AUC 95% CI
            ci_lower_calc = np.mean(auc_diff_calc) - 1.96 * np.std(auc_diff_calc, ddof=1)
            ci_upper_calc = np.mean(auc_diff_calc) + 1.96 * np.std(auc_diff_calc, ddof=1)
            print_and_save(f"AUC 95% CI (Analytic): ({ci_lower_calc:.2f}, {ci_upper_calc:.2f})", results_file)

            ci_lower_bayes = np.mean(auc_diff_bayes) - 1.96 * np.std(auc_diff_bayes, ddof=1)
            ci_upper_bayes = np.mean(auc_diff_bayes) + 1.96 * np.std(auc_diff_bayes, ddof=1)
            print_and_save(f"AUC 95% CI (Bayesian): ({ci_lower_bayes:.2f}, {ci_upper_bayes:.2f})", results_file)

            # RMSE calculations
            auc_rmse_calc = np.sqrt(np.mean(auc_diff_calc**2))
            auc_rmse_bayes = np.sqrt(np.mean(auc_diff_bayes**2))
            print_and_save(f"AUC RMSE: Analytic = {auc_rmse_calc:.2f}, Bayesian = {auc_rmse_bayes:.2f}", results_file)

            # Percentage error for AUC
            perc_error_auc_calc = ((auc_calc / auc_true - 1) * 100)
            perc_error_auc_bayes = ((auc_bayes / auc_true - 1) * 100)
            mean_perc_auc_calc = np.mean(perc_error_auc_calc)
            mean_perc_auc_bayes = np.mean(perc_error_auc_bayes)
            rmse_perc_auc_calc = np.sqrt(np.mean(perc_error_auc_calc**2))
            rmse_perc_auc_bayes = np.sqrt(np.mean(perc_error_auc_bayes**2))
            print_and_save(f"AUC Percentage Error Mean: Analytic = {mean_perc_auc_calc:.2f}%, Bayesian = {mean_perc_auc_bayes:.2f}%", results_file)
            print_and_save(f"AUC Percentage Error RMSE: Analytic = {rmse_perc_auc_calc:.2f}%, Bayesian = {rmse_perc_auc_bayes:.2f}%", results_file)

            # Pearson's r for AUC
            print_and_save(f"AUC Pearson's r: Analytic = {pearson_r_calc:.3f}, Bayesian = {pearson_r_bayes:.3f}", results_file)

            # McNemar test
            result_mc_cb, table_mc_cb = mcnemar_test(auc_true, auc_calc, auc_bayes)
            p_mc_cb_corrected = min(result_mc_cb.pvalue * 8, 1.0)
            print_and_save(f"McNemar test: statistic = {result_mc_cb.statistic:.3f}, {format_p_value(result_mc_cb.pvalue)} (Bonferroni corrected: {format_p_value(p_mc_cb_corrected)})", results_file)
            print_and_save(f"Contingency table:\n{table_mc_cb}", results_file)
            correct_calc = int(table_mc_cb[1, 0] + table_mc_cb[1, 1])
            correct_bayes_cb = int(table_mc_cb[0, 1] + table_mc_cb[1, 1])
            print_and_save(f"Correct classifications: Analytic = {correct_calc}, Bayesian = {correct_bayes_cb}", results_file)

            # Cl comparison
            cl_diff_calc = cl_calc - cl_true
            cl_diff_bayes = cl_bayes - cl_true
            t_cl_cb, p_cl_cb = stats.ttest_rel(cl_diff_calc, cl_diff_bayes)
            p_cl_cb_corrected = min(p_cl_cb * 8, 1.0)
            print_and_save(f"Cl Diff: Analytic = {np.mean(cl_diff_calc):.2f}, Bayesian = {np.mean(cl_diff_bayes):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_cl_cb:.3f}, {format_p_value(p_cl_cb)} (Bonferroni corrected: {format_p_value(p_cl_cb_corrected)})", results_file)

            # Cl 95% CI
            ci_cl_lower_calc = np.mean(cl_diff_calc) - 1.96 * np.std(cl_diff_calc, ddof=1)
            ci_cl_upper_calc = np.mean(cl_diff_calc) + 1.96 * np.std(cl_diff_calc, ddof=1)
            print_and_save(f"Cl 95% CI (Analytic): ({ci_cl_lower_calc:.2f}, {ci_cl_upper_calc:.2f})", results_file)

            ci_cl_lower_bayes = np.mean(cl_diff_bayes) - 1.96 * np.std(cl_diff_bayes, ddof=1)
            ci_cl_upper_bayes = np.mean(cl_diff_bayes) + 1.96 * np.std(cl_diff_bayes, ddof=1)
            print_and_save(f"Cl 95% CI (Bayesian): ({ci_cl_lower_bayes:.2f}, {ci_cl_upper_bayes:.2f})", results_file)

            # Cl RMSE
            cl_rmse_calc = np.sqrt(np.mean(cl_diff_calc**2))
            cl_rmse_bayes = np.sqrt(np.mean(cl_diff_bayes**2))
            print_and_save(f"Cl RMSE: Analytic = {cl_rmse_calc:.2f}, Bayesian = {cl_rmse_bayes:.2f}", results_file)

            # Section 2: One-Compartment Bayesian Peak-Trough vs Two-Compartment Bayesian Peak-Trough
            print_and_save("\n=== One-Compartment Bayesian Peak-Trough vs Two-Compartment Bayesian Peak-Trough ===", results_file)

            # AUC comparison
            t_auc_bt, p_auc_bt = stats.ttest_rel(auc_diff_bayes, auc_fit_two - auc_true)
            p_auc_bt_corrected = min(p_auc_bt * 8, 1.0)
            auc_diff_two = auc_fit_two - auc_true
            print_and_save(f"AUC Diff: 1-Compt Bayes = {np.mean(auc_diff_bayes):.2f}, 2-Compt Bayes = {np.mean(auc_diff_two):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_auc_bt:.3f}, {format_p_value(p_auc_bt)} (Bonferroni corrected: {format_p_value(p_auc_bt_corrected)})", results_file)

            # AUC 95% CI
            print_and_save(f"AUC 95% CI (1-Compt Bayes): ({ci_lower_bayes:.2f}, {ci_upper_bayes:.2f})", results_file)

            ci_lower_two = np.mean(auc_diff_two) - 1.96 * np.std(auc_diff_two, ddof=1)
            ci_upper_two = np.mean(auc_diff_two) + 1.96 * np.std(auc_diff_two, ddof=1)
            print_and_save(f"AUC 95% CI (2-Compt Bayes): ({ci_lower_two:.2f}, {ci_upper_two:.2f})", results_file)

            # RMSE calculations
            auc_rmse_two = np.sqrt(np.mean(auc_diff_two**2))
            print_and_save(f"AUC RMSE: 1-Compt Bayes = {auc_rmse_bayes:.2f}, 2-Compt Bayes = {auc_rmse_two:.2f}", results_file)

            # Percentage error for AUC
            perc_error_auc_two = ((auc_fit_two / auc_true - 1) * 100)
            mean_perc_auc_two = np.mean(perc_error_auc_two)
            rmse_perc_auc_two = np.sqrt(np.mean(perc_error_auc_two**2))
            print_and_save(f"AUC Percentage Error Mean: 1-Compt Bayes = {mean_perc_auc_bayes:.2f}%, 2-Compt Bayes = {mean_perc_auc_two:.2f}%", results_file)
            print_and_save(f"AUC Percentage Error RMSE: 1-Compt Bayes = {rmse_perc_auc_bayes:.2f}%, 2-Compt Bayes = {rmse_perc_auc_two:.2f}%", results_file)

            # Pearson's r for AUC
            print_and_save(f"AUC Pearson's r: 1-Compt Bayes = {pearson_r_bayes:.3f}, 2-Compt Bayes = {pearson_r_two:.3f}", results_file)

            # McNemar test
            result_mc_bt, table_mc_bt = mcnemar_test(auc_true, auc_bayes, auc_fit_two)
            p_mc_bt_corrected = min(result_mc_bt.pvalue * 8, 1.0)
            print_and_save(f"McNemar test: statistic = {result_mc_bt.statistic:.3f}, {format_p_value(result_mc_bt.pvalue)} (Bonferroni corrected: {format_p_value(p_mc_bt_corrected)})", results_file)
            print_and_save(f"Contingency table:\n{table_mc_bt}", results_file)
            correct_bayes_bt = int(table_mc_bt[1, 0] + table_mc_bt[1, 1])
            correct_two = int(table_mc_bt[0, 1] + table_mc_bt[1, 1])
            print_and_save(f"Correct classifications: 1-Compt Bayes = {correct_bayes_bt}, 2-Compt Bayes = {correct_two}", results_file)

            # Cl comparison
            cl_diff_two = cl_fit_two - cl_true
            t_cl_bt, p_cl_bt = stats.ttest_rel(cl_diff_bayes, cl_diff_two)
            p_cl_bt_corrected = min(p_cl_bt * 8, 1.0)
            print_and_save(f"Cl Diff: 1-Compt Bayes = {np.mean(cl_diff_bayes):.2f}, 2-Compt Bayes = {np.mean(cl_diff_two):.2f}", results_file)
            print_and_save(f"Paired t-test: t = {t_cl_bt:.3f}, {format_p_value(p_cl_bt)} (Bonferroni corrected: {format_p_value(p_cl_bt_corrected)})", results_file)

            # Cl 95% CI
            print_and_save(f"Cl 95% CI (1-Compt Bayes): ({ci_cl_lower_bayes:.2f}, {ci_cl_upper_bayes:.2f})", results_file)

            ci_cl_lower_two = np.mean(cl_diff_two) - 1.96 * np.std(cl_diff_two, ddof=1)
            ci_cl_upper_two = np.mean(cl_diff_two) + 1.96 * np.std(cl_diff_two, ddof=1)
            print_and_save(f"Cl 95% CI (2-Compt Bayes): ({ci_cl_lower_two:.2f}, {ci_cl_upper_two:.2f})", results_file)

            # Cl RMSE
            cl_rmse_two = np.sqrt(np.mean(cl_diff_two**2))
            print_and_save(f"Cl RMSE: 1-Compt Bayes = {cl_rmse_bayes:.2f}, 2-Compt Bayes = {cl_rmse_two:.2f}", results_file)

    finally:
        results_file.close()

if __name__ == '__main__':
    main()