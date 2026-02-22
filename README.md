# Python Monte Carlo Simulation for Vancomycin Pharmacokinetic Modeling

This repository contains Monte Carlo simulation code for comparing one-compartment and two-compartment pharmacokinetic models in vancomycin therapeutic drug monitoring (TDM). The project evaluates the accuracy of Area Under the Curve (AUC) predictions using different modeling approaches, including Bayesian and non-Bayesian methods.

## Overview

Vancomycin is a critical antibiotic used for treating serious infections, and accurate dosing is essential to ensure therapeutic efficacy while minimizing toxicity. This project uses Monte Carlo simulations to:

- Compare one-compartment vs. two-compartment pharmacokinetic models
- Evaluate 1-compartment peak-trough non-bayesian, 1 compartment bayesian trough and peak-trough, two-compartment Bayesian trough and peak-trough vs. reference model for AUC estimation
- Assess model performance across different renal function levels and patient weights
- Generate statistical comparisons and visualizations

## Features

- **Monte Carlo Simulations**: Generates thousands of simulated patient scenarios with varying pharmacokinetic parameters
- **Model Comparisons**: Compares AUC predictions between:
  - One-compartment analytic models using a peak an trough
  - One-compartment Bayesian models using a trough and a peak and a trough
  - Two-compartment Bayesian models using a trough and a peak and a trough
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
git clone https://github.com/yourusername/Python_Monte_Carlo_2.git
cd Python_Monte_Carlo_2
```

2. Install required packages (see Dependencies section)

## Usage

### Running Individual Comparisons

The repository contains several Python scripts for different types of comparisons:

- `OneVrsTwoCompartmentModelComparisonGeometricMean.py`: Main comparison script for analytic peak-trough, fixed Vd trough, 1 compartment Bayesian trough and peak-trough vs 2 compartment Bayesian reference model
- `TwoCompartmentModelLevelComparisons.py`: Main comparison script for 2 compartment Bayesian trough and peak-trough vs 2 compartment Bayesian reference model
- `run_multiple_weights_crcls_OneVrsTwoCompartmentModelComparisonGeometricMean.py`: Runs OneVrsTwoCompartmentModelComparisonGeometricMean.py with multiple weights and CrCLs
- `run_two_compartment_multiple_weights_crcls.py`: Runs TwoCompartmentModelLevelComparisons.py with multiple weights and CrCLs
- `statistical_comparisons_trough_models.py`: Statistical analysis of trough models; 1 compartment Bayesian trough vs 2 compartment Bayesian trough and 1 compartment fixed Vd vs 2 compartment Bayesian trough
- `statistical_comparisons_pk_tr_vs_bayesian.py`: Statistical analysis of peak-trough models; 1 compartment analytic peak-trough vs 1 compartment Bayesian peak-trough and 1 compartment Bayesian peak-trough vs 2 compartment Bayesian peak-trough models
- `run_statistical_comparison_one_two_comp_bay_tr_vr_one_two_compart_nonbay_pktr_bay_pktr_merged_data_sets.py`: Statistical comparison of trough vs peak-trough models; 2 compartment Bayesian trough vs 1 compartment Bayesian peak-trough, 2 compartment Bayesian trough vs 2 compartment Bayesian peak-trough, 1 compartment Bayesian trough vs 1 compartment Bayesian peak-trough, 1 compartment Bayesian trough vs 1 compartment analytic peak-trough


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

The simulations generate various output files in the `output/` and `merged_output/` directories:

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

- **One-compartment**: Simple elimination model
- **Two-compartment**: Distribution and elimination phases
- **Bayesian**: Incorporates prior knowledge and multiple concentration measurements
- **Analytic**: Uses population parameters and single-point estimates

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