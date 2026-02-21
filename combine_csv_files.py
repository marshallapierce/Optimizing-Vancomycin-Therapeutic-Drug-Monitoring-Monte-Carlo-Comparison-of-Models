import os
import pandas as pd
import glob
import re

def extract_crcl_wt(dirname):
    """Extract Crcl, wt, and time_config from directory name"""
    # Remove trailing backslash if present
    dirname = dirname.rstrip('\\')
    match = re.search(r'output(?:_twocompartment)?_Clcr(\d+)_wt(\d+)(?:_(.*))?', dirname)
    if match:
        crcl = int(match.group(1))
        wt = int(match.group(2))
        time_config = match.group(3) if match.group(3) else 'default'
        # Clean up time_config by removing any remaining backslashes
        time_config = time_config.rstrip('\\') if time_config else 'default'
        return crcl, wt, time_config
    return None, None, None

def combine_csv_files():
    # Create merged_output directory if it doesn't exist
    os.makedirs('merged_output', exist_ok=True)

    # Patterns for directories
    one_compt_dirs = glob.glob('output_Clcr*_wt*/')
    two_compt_dirs = glob.glob('output_twocompartment_Clcr*_wt*/')

    # Group two-compartment directories by time config
    time_configs = {}
    for dirname in two_compt_dirs:
        crcl, wt, time_config = extract_crcl_wt(dirname)
        if crcl is None:
            continue
        if time_config not in time_configs:
            time_configs[time_config] = []
        time_configs[time_config].append(dirname)

    # Combine one-compartment data (assuming no time configs for one-compartment)
    print("Processing one-compartment data...")
    one_compt_files = []
    for dirname in one_compt_dirs:
        crcl, wt, _ = extract_crcl_wt(dirname)
        if crcl is None:
            continue

        csv_path = os.path.join(dirname, 'monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv')
        if os.path.exists(csv_path):
            one_compt_files.append((csv_path, crcl, wt))
            print(f"Found {csv_path}")

    if one_compt_files:
        output_path = 'merged_output/combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.csv'
        first_file = True
        total_rows = 0

        for csv_path, crcl, wt in one_compt_files:
            try:
                # Read in chunks to handle large files
                chunk_size = 10000
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    chunk['Crcl'] = crcl
                    chunk['weight'] = wt
                    # Write header only for first chunk of first file
                    chunk.to_csv(output_path, mode='a' if not first_file else 'w',
                               header=first_file, index=False)
                    first_file = False
                    total_rows += len(chunk)
                    print(f"Processed chunk from {os.path.basename(csv_path)}: {len(chunk)} rows")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                continue

        print(f"Combined one-compartment data saved to {output_path} with {total_rows} total rows")
    else:
        print("No one-compartment data found")

    # Combine two-compartment data by time config
    print(f"Found time configs: {list(time_configs.keys())}")
    for time_config, directories in time_configs.items():
        print(f"Processing config '{time_config}' with {len(directories)} directories")
        output_path = f'merged_output/combined_monte_carlo_results_two_vrs_two_{time_config}.csv'
        first_file = True
        total_rows = 0

        for dirname in directories:
            crcl, wt, _ = extract_crcl_wt(dirname)
            print(f"  Processing directory: {dirname} -> Crcl: {crcl}, Wt: {wt}")

            # Look for CSV files that match the time config in the filename
            if time_config == 'trough':
                csv_pattern = os.path.join(dirname, 'monte_carlo_results_two_vrs_two_10_trough.csv')
            elif time_config == 'peaktrough':
                csv_pattern = os.path.join(dirname, 'monte_carlo_results_two_vrs_two_2_10_peaktrough.csv')
            else:
                # For default config, look for the original filename
                csv_pattern = os.path.join(dirname, 'monte_carlo_results_two_vrs_two.csv')

            print(f"    Looking for file: {csv_pattern}")
            if os.path.exists(csv_pattern):
                try:
                    # Read in chunks to handle large files
                    chunk_size = 10000
                    for chunk in pd.read_csv(csv_pattern, chunksize=chunk_size):
                        chunk['Crcl'] = crcl
                        chunk['weight'] = wt
                        chunk['time_config'] = time_config  # Add time config column
                        # Write header only for first chunk of first file
                        chunk.to_csv(output_path, mode='a' if not first_file else 'w',
                                   header=first_file, index=False)
                        first_file = False
                        total_rows += len(chunk)
                        print(f"    Processed chunk from {os.path.basename(csv_pattern)}: {len(chunk)} rows")
                except Exception as e:
                    print(f"    Error processing {csv_pattern}: {e}")
                    continue
                print(f"    ✓ Loaded {csv_pattern} (config: {time_config})")
            else:
                print(f"    ✗ No matching CSV file found: {csv_pattern}")

        if total_rows > 0:
            print(f"✓ Combined two-compartment data ({time_config}) saved to {output_path} with {total_rows} total rows")
        else:
            print(f"✗ No two-compartment data found for config: {time_config}")

if __name__ == '__main__':
    combine_csv_files()