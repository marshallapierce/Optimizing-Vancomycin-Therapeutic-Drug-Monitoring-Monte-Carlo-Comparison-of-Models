import pandas as pd
import numpy as np

# Load a small sample of both datasets
print('Loading samples...')
df1 = pd.read_csv('merged_output/combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv', nrows=5000)
df2 = pd.read_csv('merged_output/combined_monte_carlo_results_two_vrs_two.csv', nrows=5000)

print('One-compartment AUC_true range:', df1['AUC_true'].min(), 'to', df1['AUC_true'].max())
print('Two-compartment AUC_true range:', df2['AUC_true'].min(), 'to', df2['AUC_true'].max())

print('One-compartment AUC_fit_bayes_last range:', df1['AUC_fit_bayes_last'].min(), 'to', df1['AUC_fit_bayes_last'].max())
print('Two-compartment AUC_fit range:', df2['AUC_fit'].min(), 'to', df2['AUC_fit'].max())

# Check distribution
print('One-compartment AUC_true distribution:')
print(df1['AUC_true'].describe())
print('Two-compartment AUC_true distribution:')
print(df2['AUC_true'].describe())

# Check if AUC_fit values are reasonable predictions
print('Two-compartment AUC_fit vs AUC_true correlation:', df2['AUC_fit'].corr(df2['AUC_true']))
print('One-compartment AUC_fit_bayes_last vs AUC_true correlation:', df1['AUC_fit_bayes_last'].corr(df1['AUC_true']))

# Check mean absolute error
print('Two-compartment MAE:', abs(df2['AUC_fit'] - df2['AUC_true']).mean())
print('One-compartment MAE:', abs(df1['AUC_fit_bayes_last'] - df1['AUC_true']).mean())