import pandas as pd

print('Loading one-compartment file...')
df1 = pd.read_csv('merged_output/combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv', nrows=10)
print('One-compartment columns:', list(df1.columns))
print('One-compartment Group values:', df1['Group'].tolist())

print('\nLoading two-compartment file...')
df2 = pd.read_csv('merged_output/combined_monte_carlo_results_two_vrs_two.csv', nrows=10)
print('Two-compartment columns:', list(df2.columns))
print('Two-compartment Group values:', df2['Group'].tolist())

print('\nChecking Group overlap...')
group1 = set(df1['Group'].unique())
group2 = set(df2['Group'].unique())
overlap = group1.intersection(group2)
print(f'One-compartment unique groups: {len(group1)}')
print(f'Two-compartment unique groups: {len(group2)}')
print(f'Overlapping groups: {len(overlap)}')
if len(overlap) > 0:
    print('Sample overlapping groups:', list(overlap)[:5])
else:
    print('NO overlapping groups found!')