import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import sys
import matplotlib.pyplot as plt

# Create output directory
output_dir = 'output'

# Statistical comparisons for trough-based pharmacokinetic models
# 1-compartment Bayesian with trough vs 2-compartment Bayesian with trough
# 1-compartment Fixed VD with trough vs 2-compartment Bayesian with trough

def load_detailed_results(suffix):
    filename = f'output/monte_carlo_all_data_onevrstwocompartment_geometric_mean{suffix}.csv'
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def load_detailed_results_two_vrs_two(suffix):
    filename = f'output/monte_carlo_results_two_vrs_two{suffix}.csv'
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
    plt.xlabel('Mean of True and Predicted AUC') # 'Mean of True and Predicted AUC'
    plt.ylabel('Difference (Predicted - True) AUC')# 'Difference (Predicted - True) AUC'
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def create_auc_comparison_csv(auc_true, auc_models, model_names, suffix):
    bins = [0, 400, 600, np.inf]
    bins_true = np.digitize(auc_true, bins) - 1
    lines = []
    lines.append("One Compartment & Two Compartment Trough Methods Versus True AUC")
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
    filename = f'{output_dir}/auc_comparison_grids_troughs{suffix}.csv'
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"Comparison grids saved to {filename}")

def main():
    if len(sys.argv) > 1:
        suffix = sys.argv[1]
    else:
        print("Enter the suffix for the data files (e.g., CltotalfourpointfiveTR):")
        sys.stdout.flush()
        suffix = input().strip()

    # Load data
    df_one = load_detailed_results(suffix)
    df_two = load_detailed_results_two_vrs_two(suffix)

    if df_one is not None and df_two is not None:
        # Merge on 'Group' to align simulations
        merged_df = pd.merge(df_one, df_two, on='Group', suffixes=('_one', '_two'))
        
        # Apply filters
        valid_one = merged_df['Vdcalc'] > 0
        valid_two = (merged_df['Vc_fit'] + merged_df['Vp_fit']) > 0
        valid = valid_one & valid_two
        df = merged_df[valid]
        
        # Extract metrics
        auc_true = df['AUC_true_one'].values
        auc_calc = df['AUCcalc'].values  # PK-TR
        auc_bayes = df['AUC_fit_bayes'].values  # 1-compt Bayesian
        auc_fixed = df['AUC_fit_fixed'].values  # 1-compt Fixed VD
        auc_fit_two = df['AUC_fit'].values  # 2-compt Bayesian
        
        # Create AUC comparison grids CSV
        auc_models = [auc_fixed, auc_bayes, auc_fit_two]
        model_names = ["Fixed Vd", "One Compartment Bayesian", "Two Compartment Bayesian"]
        create_auc_comparison_csv(auc_true, auc_models, model_names, suffix)
        
        # Generate Bland-Altman plots
        bland_altman_plot(auc_true, auc_bayes, 'Bland-Altman: One-Compartment Bayesian Trough Model AUC vs True AUC', f'output/bland_altman_one_compt_bayes{suffix}.png')
        bland_altman_plot(auc_true, auc_fit_two, 'Bland-Altman: Two-Compartment Bayesian Trough Model AUC vs True AUC', f'output/bland_altman_two_compt_bayes{suffix}.png')
        bland_altman_plot(auc_true, auc_fixed, 'Bland-Altman: One-Compartment Fixed VD Trough Model AUC vs True AUC', f'output/bland_altman_one_compt_fixed{suffix}.png')
        
        cl_true = df['Cl_total_true_one'].values
        cl_calc = df['Clcalc'].values
        cl_bayes = df['Cl_fit_bayes'].values
        cl_fixed = df['Cl_fit_fixed'].values
        cl_fit_two = df['Cl_total_fit'].values
        
        cp_10_bayes = df['Cp_10.0_fit_bayes'].values  # 1-compt Bayesian trough
        cp_10_fixed = df['Cp_10.0_fit_fixed'].values  # 1-compt Fixed VD trough
        cp_10_fit_two = df['Cp_10_fit'].values  # 2-compt Bayesian trough
        
        pearson_r_calc, _ = stats.pearsonr(auc_true, auc_calc)
        pearson_r_bayes, _ = stats.pearsonr(auc_true, auc_bayes)
        pearson_r_fixed, _ = stats.pearsonr(auc_true, auc_fixed)
        pearson_r_two, _ = stats.pearsonr(auc_true, auc_fit_two)
        
        print(f"=== Trough-Based Pharmacokinetic Model Comparisons (Suffix: {suffix}) ===")
        print(f"Merged and filtered data: {len(df)} simulations")

        # Section 1: 1-Compartment Bayesian with Trough vs 2-Compartment Bayesian with Trough
        print("\n=== 1-Compartment Bayesian Trough vs 2-Compartment Bayesian Trough ===")

        # AUC comparison
        auc_diff_1_bayes = auc_bayes - auc_true
        auc_diff_2 = auc_fit_two - auc_true
        t_auc_12, p_auc_12 = stats.ttest_rel(auc_diff_1_bayes, auc_diff_2)
        p_auc_12_corrected = min(p_auc_12 * 8, 1.0)  # Bonferroni for 8 tests
        print(f"AUC Diff: 1-Compt Bayes = {np.mean(auc_diff_1_bayes):.2f}, 2-Compt Bayes = {np.mean(auc_diff_2):.2f}")
        print(f"Paired t-test: t = {t_auc_12:.3f}, {format_p_value(p_auc_12)} (Bonferroni corrected: {format_p_value(p_auc_12_corrected)})")

        # AUC 95% CI
        ci_lower_1 = np.mean(auc_diff_1_bayes) - 1.96 * np.std(auc_diff_1_bayes, ddof=1)
        ci_upper_1 = np.mean(auc_diff_1_bayes) + 1.96 * np.std(auc_diff_1_bayes, ddof=1)
        print(f"AUC 95% CI (1-Compt Bayes): ({ci_lower_1:.2f}, {ci_upper_1:.2f})")

        ci_lower_2 = np.mean(auc_diff_2) - 1.96 * np.std(auc_diff_2, ddof=1)
        ci_upper_2 = np.mean(auc_diff_2) + 1.96 * np.std(auc_diff_2, ddof=1)
        print(f"AUC 95% CI (2-Compt Bayes): ({ci_lower_2:.2f}, {ci_upper_2:.2f})")

        # RMSE calculations
        auc_rmse_1_bayes = np.sqrt(np.mean(auc_diff_1_bayes**2))
        auc_rmse_2 = np.sqrt(np.mean(auc_diff_2**2))
        print(f"AUC RMSE: 1-Compt Bayes = {auc_rmse_1_bayes:.2f}, 2-Compt Bayes = {auc_rmse_2:.2f}")

        # Percentage error for AUC
        perc_error_auc_1_bayes = ((auc_bayes / auc_true - 1) * 100)
        perc_error_auc_2 = ((auc_fit_two / auc_true - 1) * 100)
        mean_perc_auc_1_bayes = np.mean(perc_error_auc_1_bayes)
        mean_perc_auc_2 = np.mean(perc_error_auc_2)
        rmse_perc_auc_1_bayes = np.sqrt(np.mean(perc_error_auc_1_bayes**2))
        rmse_perc_auc_2 = np.sqrt(np.mean(perc_error_auc_2**2))
        print(f"AUC Percentage Error Mean: 1-Compt Bayes = {mean_perc_auc_1_bayes:.2f}%, 2-Compt Bayes = {mean_perc_auc_2:.2f}%")
        print(f"AUC Percentage Error RMSE: 1-Compt Bayes = {rmse_perc_auc_1_bayes:.2f}%, 2-Compt Bayes = {rmse_perc_auc_2:.2f}%")

        # Pearson's r for AUC
        print(f"AUC Pearson's r: 1-Compt Bayes = {pearson_r_bayes:.3f}, 2-Compt Bayes = {pearson_r_two:.3f}")

        # McNemar test
        result_mc, table_mc = mcnemar_test(auc_true, auc_bayes, auc_fit_two)
        p_mc_corrected = min(result_mc.pvalue * 8, 1.0)
        print(f"McNemar test: statistic = {result_mc.statistic:.3f}, {format_p_value(result_mc.pvalue)} (Bonferroni corrected: {format_p_value(p_mc_corrected)})")
        print(f"Contingency table:\n{table_mc}")
        correct_1_compt = int(table_mc[1, 0] + table_mc[1, 1])
        correct_2_compt = int(table_mc[0, 1] + table_mc[1, 1])
        print(f"Correct classifications: 1-Compt Bayes = {correct_1_compt}, 2-Compt Bayes = {correct_2_compt}")

        # Cl comparison
        cl_diff_1_bayes = cl_bayes - cl_true
        cl_diff_2 = cl_fit_two - cl_true
        t_cl_12, p_cl_12 = stats.ttest_rel(cl_diff_1_bayes, cl_diff_2)
        p_cl_12_corrected = min(p_cl_12 * 8, 1.0)
        print(f"Cl Diff: 1-Compt Bayes = {np.mean(cl_diff_1_bayes):.2f}, 2-Compt Bayes = {np.mean(cl_diff_2):.2f}")
        print(f"Paired t-test: t = {t_cl_12:.3f}, {format_p_value(p_cl_12)} (Bonferroni corrected: {format_p_value(p_cl_12_corrected)})")

        # Cl 95% CI
        ci_cl_lower_1 = np.mean(cl_diff_1_bayes) - 1.96 * np.std(cl_diff_1_bayes, ddof=1)
        ci_cl_upper_1 = np.mean(cl_diff_1_bayes) + 1.96 * np.std(cl_diff_1_bayes, ddof=1)
        print(f"Cl 95% CI (1-Compt Bayes): ({ci_cl_lower_1:.2f}, {ci_cl_upper_1:.2f})")

        ci_cl_lower_2 = np.mean(cl_diff_2) - 1.96 * np.std(cl_diff_2, ddof=1)
        ci_cl_upper_2 = np.mean(cl_diff_2) + 1.96 * np.std(cl_diff_2, ddof=1)
        print(f"Cl 95% CI (2-Compt Bayes): ({ci_cl_lower_2:.2f}, {ci_cl_upper_2:.2f})")

        # Cl RMSE
        cl_rmse_1_bayes = np.sqrt(np.mean(cl_diff_1_bayes**2))
        cl_rmse_2 = np.sqrt(np.mean(cl_diff_2**2))
        print(f"Cl RMSE: 1-Compt Bayes = {cl_rmse_1_bayes:.2f}, 2-Compt Bayes = {cl_rmse_2:.2f}")

        # Trough concentration comparison
        t_cp_12, p_cp_12 = stats.ttest_rel(cp_10_bayes, cp_10_fit_two)
        p_cp_12_corrected = min(p_cp_12 * 8, 1.0)
        print(f"Trough Cp: 1-Compt Bayes = {np.mean(cp_10_bayes):.2f}, 2-Compt Bayes = {np.mean(cp_10_fit_two):.2f}")
        print(f"Paired t-test: t = {t_cp_12:.3f}, {format_p_value(p_cp_12)} (Bonferroni corrected: {format_p_value(p_cp_12_corrected)})")

        # Section 2: 1-Compartment Fixed VD with Trough vs 2-Compartment Bayesian with Trough
        print("\n=== 1-Compartment Fixed VD Trough vs 2-Compartment Bayesian Trough ===")

        # AUC comparison
        auc_diff_fixed = auc_fixed - auc_true
        t_auc_f2, p_auc_f2 = stats.ttest_rel(auc_diff_fixed, auc_diff_2)
        p_auc_f2_corrected = min(p_auc_f2 * 8, 1.0)
        print(f"AUC Diff: Fixed VD = {np.mean(auc_diff_fixed):.2f}, 2-Compt Bayes = {np.mean(auc_diff_2):.2f}")
        print(f"Paired t-test: t = {t_auc_f2:.3f}, {format_p_value(p_auc_f2)} (Bonferroni corrected: {format_p_value(p_auc_f2_corrected)})")

        # AUC 95% CI
        ci_lower_fixed = np.mean(auc_diff_fixed) - 1.96 * np.std(auc_diff_fixed, ddof=1)
        ci_upper_fixed = np.mean(auc_diff_fixed) + 1.96 * np.std(auc_diff_fixed, ddof=1)
        print(f"AUC 95% CI (Fixed VD): ({ci_lower_fixed:.2f}, {ci_upper_fixed:.2f})")
        print(f"AUC 95% CI (2-Compt Bayes): ({ci_lower_2:.2f}, {ci_upper_2:.2f})")

        # RMSE calculations
        auc_rmse_fixed = np.sqrt(np.mean(auc_diff_fixed**2))
        print(f"AUC RMSE: Fixed VD = {auc_rmse_fixed:.2f}, 2-Compt Bayes = {auc_rmse_2:.2f}")

        # Percentage error for AUC
        perc_error_auc_fixed = ((auc_fixed / auc_true - 1) * 100)
        mean_perc_auc_fixed = np.mean(perc_error_auc_fixed)
        rmse_perc_auc_fixed = np.sqrt(np.mean(perc_error_auc_fixed**2))
        print(f"AUC Percentage Error Mean: Fixed VD = {mean_perc_auc_fixed:.2f}%, 2-Compt Bayes = {mean_perc_auc_2:.2f}%")
        print(f"AUC Percentage Error RMSE: Fixed VD = {rmse_perc_auc_fixed:.2f}%, 2-Compt Bayes = {rmse_perc_auc_2:.2f}%")

        # Pearson's r for AUC
        print(f"AUC Pearson's r: Fixed VD = {pearson_r_fixed:.3f}, 2-Compt Bayes = {pearson_r_two:.3f}")

        # McNemar test
        result_mc_f, table_mc_f = mcnemar_test(auc_true, auc_fixed, auc_fit_two)
        p_mc_f_corrected = min(result_mc_f.pvalue * 8, 1.0)
        print(f"McNemar test: statistic = {result_mc_f.statistic:.3f}, {format_p_value(result_mc_f.pvalue)} (Bonferroni corrected: {format_p_value(p_mc_f_corrected)})")
        print(f"Contingency table:\n{table_mc_f}")
        correct_fixed = int(table_mc_f[1, 0] + table_mc_f[1, 1])
        correct_2_compt_f = int(table_mc_f[0, 1] + table_mc_f[1, 1])
        print(f"Correct classifications: Fixed VD = {correct_fixed}, 2-Compt Bayes = {correct_2_compt_f}")

        # Cl comparison
        cl_diff_fixed = cl_fixed - cl_true
        t_cl_f2, p_cl_f2 = stats.ttest_rel(cl_diff_fixed, cl_diff_2)
        p_cl_f2_corrected = min(p_cl_f2 * 8, 1.0)
        print(f"Cl Diff: Fixed VD = {np.mean(cl_diff_fixed):.2f}, 2-Compt Bayes = {np.mean(cl_diff_2):.2f}")
        print(f"Paired t-test: t = {t_cl_f2:.3f}, {format_p_value(p_cl_f2)} (Bonferroni corrected: {format_p_value(p_cl_f2_corrected)})")

        # Cl 95% CI
        ci_cl_lower_fixed = np.mean(cl_diff_fixed) - 1.96 * np.std(cl_diff_fixed, ddof=1)
        ci_cl_upper_fixed = np.mean(cl_diff_fixed) + 1.96 * np.std(cl_diff_fixed, ddof=1)
        print(f"Cl 95% CI (Fixed VD): ({ci_cl_lower_fixed:.2f}, {ci_cl_upper_fixed:.2f})")
        print(f"Cl 95% CI (2-Compt Bayes): ({ci_cl_lower_2:.2f}, {ci_cl_upper_2:.2f})")

        # Cl RMSE
        cl_rmse_fixed = np.sqrt(np.mean(cl_diff_fixed**2))
        print(f"Cl RMSE: Fixed VD = {cl_rmse_fixed:.2f}, 2-Compt Bayes = {cl_rmse_2:.2f}")

        # Trough concentration comparison
        t_cp_f2, p_cp_f2 = stats.ttest_rel(cp_10_fixed, cp_10_fit_two)
        p_cp_f2_corrected = min(p_cp_f2 * 8, 1.0)
        print(f"Trough Cp: Fixed VD = {np.mean(cp_10_fixed):.2f}, 2-Compt Bayes = {np.mean(cp_10_fit_two):.2f}")
        print(f"Paired t-test: t = {t_cp_f2:.3f}, {format_p_value(p_cp_f2)} (Bonferroni corrected: {format_p_value(p_cp_f2_corrected)})")

if __name__ == '__main__':
    main()