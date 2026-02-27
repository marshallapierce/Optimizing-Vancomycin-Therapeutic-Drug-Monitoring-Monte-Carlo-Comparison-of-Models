# Monte Carlo Simulation Data for Vancomycin Pharmacokinetic Modeling

This repository contains compressed data files from Monte Carlo simulations comparing one-compartment and two-compartment pharmacokinetic models for vancomycin therapeutic drug monitoring.

## Data Access

The simulation results are stored as compressed 7-Zip (.7z) archives in the `merged_output/` directory.

### Prerequisites
- **Git LFS**: Required to download the full files. Install from https://git-lfs.github.com/
- **7-Zip**: Required to extract the archives. Download from https://www.7-zip.org/

### Cloning with Data
```bash
git lfs install
git clone https://github.com/marshallapierce/Optimizing-Vancomycin-Therapeutic-Drug-Monitoring-Monte-Carlo-Comparison-of-Models.git
```

### Extracting Data Files
1. Navigate to the `merged_output/` directory.
2. Right-click each .7z file and select "7-Zip > Extract Here" to decompress.
3. The extracted CSV files contain the merged Monte Carlo simulation results.

### Data Files
- `combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.7z`: Combined results for one vs two compartment geometric mean comparisons (~414 MB extracted).
- `combined_monte_carlo_results_two_vrs_two_default.7z`: Two vs two compartment default model results (~312 MB extracted).
- `combined_monte_carlo_results_two_vrs_two_peaktrough.7z`: Two vs two compartment peak-trough results (~313 MB extracted).
- `combined_monte_carlo_results_two_vrs_two_trough.7z`: Two vs two compartment trough results (~303 MB extracted).

## Code Repository

The Python code and scripts for running these simulations are available in a separate private repository. Contact the maintainer for access if needed.

## License

This data is provided under the MIT License.