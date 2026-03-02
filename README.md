# Python Monte Carlo Simulation for Vancomycin Pharmacokinetic Modeling

This repository contains Monte Carlo simulation code for comparing one-compartment and two-compartment pharmacokinetic models in vancomycin therapeutic drug monitoring (TDM). The project evaluates the accuracy of Area Under the Curve (AUC) predictions using different modeling approaches, including 1 compartment analytic peak-trough, 1 compartment fixed Vd trough, and 1 and 2 compartment Bayesian methods, againt the 2 compartment Goti reference model.

## Overview

Vancomycin is a critical antibiotic used for treating serious infections, and accurate dosing is essential to ensure therapeutic efficacy while minimizing toxicity. This project uses Monte Carlo simulations to:

- Compare one-compartment vs. two-compartment pharmacokinetic models using sparse data sampling (trough, peak-trough).
- Evaluate 1-compartment analytic peak-trough non-bayesian, 1 compartment Bayesian trough and peak-trough, 2 compartment Bayesian trough and peak-trough vs. reference 2 compartment Bayesian Goti model for AUC estimation. Statistically compare the difference models performance against the reference model and again each other.
- Assess model performance across different renal function levels (crcl ml/min 30, 60, 90, 120) and patient weights (50, 70, 90, 110 kg)
- Generate statistical comparisons and visualizations for individual models vs reference model in folders (1 compartment models folder names output_clcrxx_wtxx, 2 compartment models folder names output_twocompartment_clcrxx_wtxx) and merged data sets of all 480,000 simulations in the merged_output folder. The statistical_comparsion text files in the merged_output folder contain the statistical comparison data for the specific models being compared.
- 

## Features

- **Monte Carlo Simulations**: Use the Goti reference model radomized mean population parameters within 2 standard deviations of the mean to generates 30,000 thousand simulated patient scenarios for each weight and repeated this for each crcl with a total number of simulations of 480,000.
- **Model Comparisons**: Compares AUC predictions of models and 2 compartment Bayesian reference model:
  - One-compartment analytic model using a peak an trough
  - One-compartment fix Vd usinng a trough
  - One-compartment Bayesian model using a trough and a peak and a trough
  - Two-compartment Goti Bayesian model using a trough and a peak and a trough
- **Statistical Analysis**: Performs McNemar tests and other statistical comparisons
- **Visualization**: Creates plots and HTML comparison tables
- **Flexible Input**: Supports different dosing regimens, renal functions, and patient weights

## Dependencies

The code requires Python 3.7+ and the following packages:

- numpy
- pandas
- scipy
- matplotlib
- statsmodels

Install dependencies using pip:

```bash
pip install numpy pandas scipy matplotlib statsmodels
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marshallapierce/Optimizing-Vancomycin-Therapeutic-Drug-Monitoring-Monte-Carlo-Comparison-of-Models.git
cd Optimizing-Vancomycin-Therapeutic-Drug-Monitoring-Monte-Carlo-Comparison-of-Models
```

2. Install required packages (see Dependencies section)

## Data Access

This repository includes large simulation data files compressed with 7-Zip (.7z) and stored using Git LFS for efficient handling.

### Prerequisites for Data Access
- **Git LFS**: Required to download the full data files. Install from https://git-lfs.github.com/
- **7-Zip**: Required to extract the .7z archives. Download from https://www.7-zip.org/

### Cloning with Data
```bash
git lfs install
git clone https://github.com/marshallapierce/Optimizing-Vancomycin-Therapeutic-Drug-Monitoring-Monte-Carlo-Comparison-of-Models.git
```

### Extracting Data Files
1. After cloning, navigate to the `merged_output/` directory.
2. Right-click each .7z file and select "7-Zip > Extract Here" to decompress.
3. The extracted CSV files contain the merged Monte Carlo simulation results for analysis.

### Data Files Description
- `combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean.7z`: Combined results for one vs two compartment geometric mean comparisons (extracts to ~414 MB CSV).
- `combined_monte_carlo_results_two_vrs_two_default.7z`: Two vs two compartment default model results (extracts to ~312 MB CSV).
- `combined_monte_carlo_results_two_vrs_two_peaktrough.7z`: Two vs two compartment peak-trough results (extracts to ~313 MB CSV).
- `combined_monte_carlo_results_two_vrs_two_trough.7z`: Two vs two compartment trough results (extracts to ~303 MB CSV).

These files are used by the statistical comparison scripts in the repository.

## Usage

### Running Individual and Group Comparisons

The repository contains several Python scripts for different types of comparisons:

- `OneVrsTwoCompartmentModelComparisonGeometricMean.py`: Main comparison script for analytic peak-trough, fixed Vd trough, 1 compartment Bayesian trough and peak-trough vs 2 compartment Bayesian reference model
- `TwoCompartmentModelLevelComparisons.py`: Main comparison script for 2 compartment Bayesian trough and peak-trough vs 2 compartment Bayesian reference model
- `run_multiple_weights_crcls_OneVrsTwoCompartmentModelComparisonGeometricMean.py`: Runs OneVrsTwoCompartmentModelComparisonGeometricMean.py with multiple weights and Crcls output is stored in separate file folders (output_clcrxx_wtxx)
- `run_two_compartment_multiple_weights_crcls.py`: Runs TwoCompartmentModelLevelComparisons.py with multiple weights and Crcls output is stored in separate files folders output_twocompartment_Clcrxx_wt_trough, output_twocompartment_Clcrxx_wtxx_peaktrough
- `combin_csv_files.py`: combines the data for the output of the above two files in the merged_output directory. Creating three files comnined_monte_carlo_results_two_vrs_two_trough, combined_onte_carlo_results_two_vrs_two_peaktrough, and combined_monte_carlo_all_data_onevrstwocompartment_geometric_mean
- `statistical_comparisons_trough_models.py`: Statistical analysis of trough models using merged data set; 1 compartment Bayesian trough vs 2 compartment Bayesian trough and 1 compartment fixed Vd vs 2 compartment Bayesian trough
- `statistical_comparisons_pk_tr_vs_pk_tr_merged_data_sets.py`: Statistical analysis of peak-trough models using merged data sets; 1 compartment analytic peak-trough vs 1 compartment Bayesian peak-trough and 1 compartment Bayesian peak-trough vs 2 compartment Bayesian peak-trough models
- `run_statistical_comparison_one_two_comp_bay_tr_vr_one_two_compart_nonbay_pktr_bay_pktr_merged_data_sets.py`: Statistical comparison of trough vs peak-trough models using merged data sets; 2 compartment Bayesian trough vs 1 compartment Bayesian peak-trough, 2 compartment Bayesian trough vs 2 compartment Bayesian peak-trough, 1 compartment Bayesian trough vs 1 compartment Bayesian peak-trough, 1 compartment Bayesian trough vs 1 compartment analytic peak-trough


### Example Usage

1. Run a basic one vs two compartment comparison:
```bash
python OneVrsTwoCompartmentModelComparisonGeometricMean.py
```

2. Follow the prompts to enter:
   - Patient weight (kg)
   - Creatinine clearance (mL/min)
   - Fitting times
   - File suffix for output

### Output Files

The simulations generate various output files in the `rootfolder/` and `merged_output/` directories:

- **CSV files**: Raw simulation data and statistical results
- **PNG plots**: Accuracy plots and AUC comparisons
- **HTML files**: Interactive comparison tables

### Key Output Directories

- `output_Clcr{value}_wt{value}/`: Results for specific renal function and weight combinations
- `merged_output/`: Combined results across all simulations
- `auc_comparison_grids_*.csv`: AUC comparison data

## Project Structure

```
├── OneVrsTwoCompartmentModelComparison.py          # Main comparison script
├── statistical_comparisons_*.py                     # Statistical analysis scripts
├── run_*.py                                         # Batch execution scripts
├── auc_*.py                                         # AUC analysis scripts
├── combined_accuracy_plot.py                        # Plotting utilities
├── output/                                          # Individual simulation outputs
├── merged_output/                                   # Combined results
├── auc_comparison_grids_*.csv                      # AUC data files
├── *.html                                          # Comparison tables
└── README.md                                        # This file
```

## Methodology

The Monte Carlo simulations:

1. Generate random pharmacokinetic parameters within physiological ranges
2. Simulate vancomycin concentration-time profiles
3. Fit models using different approaches (analytic vs Bayesian)
4. Calculate AUC and compare predictions
5. Perform statistical tests to evaluate model performance

### Pharmacokinetic Models

- **Referemce model**: Goti 2 compartment Bayesian model
- **One-compartment**: Simple elimination model (1 compartment open model)
- **Two-compartment**: Distribution and elimination phases (2 compartment open model)
- **Bayesian**: Incorporates prior knowledge and multiple concentration measurements
- **Analytic**: 1 compartment Sawchuck-Zaske model, from which are dervied population parameters used in 1 compartment Bayesian model and fixed Vd model estimates

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

The pharmacokinetic models and parameters are based on published literature in vancomycin therapeutic drug monitoring. See `vancomycin_references.ris` for citation details.

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.