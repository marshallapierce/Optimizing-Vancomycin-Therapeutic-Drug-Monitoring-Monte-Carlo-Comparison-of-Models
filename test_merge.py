import pandas as pd
import sys

def load_detailed_results():
    filename = 'merged_output/combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv'
    df = pd.read_csv(filename)
    return df

def load_detailed_results_two_vrs_two():
    filename = 'merged_output/combined_monte_carlo_results_two_vrs_two.csv'
    df = pd.read_csv(filename)
    return df

def load_method_data(compartment, method):
    if compartment == 'one':
        df = load_detailed_results()
        if method == 'bay_pk_tr':
            df = df.rename(columns={'AUC_fit_bayes_full': 'AUC_bay_pk_tr', 'Cl_fit_bayes_full': 'Cl_bay_pk_tr'})
        return df[['Group', 'AUC_true', 'AUC_bay_pk_tr', 'Cl_total_true', 'Cl_bay_pk_tr']]
    elif compartment == 'two':
        df = load_detailed_results_two_vrs_two()
        df = df.rename(columns={'AUC_fit': 'AUC_bay_tr', 'Cl_total_fit': 'Cl_bay_tr'})
        return df[['Group', 'AUC_true', 'AUC_bay_tr', 'Cl_total_true', 'Cl_bay_tr']]

print("Loading data for comparison 2...")
df1 = load_method_data('two', 'bay_tr')  # 2-compartment Bayesian trough
df2 = load_method_data('one', 'bay_pk_tr')  # 1-compartment Bayesian peak trough

print(f"df1 (2-compartment): {len(df1)} rows")
print(f"df2 (1-compartment): {len(df2)} rows")
print(f"df1 Group range: {df1['Group'].min()} - {df1['Group'].max()}")
print(f"df2 Group range: {df2['Group'].min()} - {df2['Group'].max()}")

print("Checking Group overlap...")
group1 = set(df1['Group'].unique())
group2 = set(df2['Group'].unique())
overlap = group1.intersection(group2)
print(f'Overlapping groups: {len(overlap)}')
if len(overlap) > 0:
    print('Sample overlapping groups:', sorted(list(overlap))[:10])
else:
    print('NO overlapping groups!')

print("Attempting merge...")
merged_df = pd.merge(df1, df2, on='Group', suffixes=('_1', '_2'))
print(f"Merged dataframe has {len(merged_df)} rows")