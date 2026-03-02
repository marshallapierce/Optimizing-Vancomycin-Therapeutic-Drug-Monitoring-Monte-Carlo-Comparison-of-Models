#!/usr/bin/env python3
"""
Script to run all four statistical comparisons sequentially.
This script executes the vancomycin pharmacokinetic Monte Carlo simulation
statistical comparisons for all model/method combinations.
"""

import subprocess
import sys
import os
import time

def run_comparison(comparison_index):
    """Run a single comparison using the virtual environment's Python."""
    print(f"\n{'='*60}")
    print(f"Starting Comparison {comparison_index}")
    print(f"{'='*60}")

    # Use the virtual environment's Python executable
    python_exe = os.path.join('.venv', 'Scripts', 'python.exe')
    script_path = 'statistical_comparison_one_two_compart_bay_tr_vs_one_two_compart_nonbay_pk_tr_bay_pk_tr.py'

    try:
        # Run the comparison (don't capture output to avoid hanging)
        result = subprocess.run([python_exe, script_path, str(comparison_index)],
                              cwd=os.getcwd())

        if result.returncode == 0:
            print(f"✅ Comparison {comparison_index} completed successfully")
            return True
        else:
            print(f"❌ Comparison {comparison_index} failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error running comparison {comparison_index}: {e}")
        return False

def main():
    """Run all four comparisons sequentially."""
    print("Starting all statistical comparisons for vancomycin pharmacokinetic models")
    print("This will run 4 comparisons sequentially, which may take several minutes each.")

    # Define the comparison descriptions
    comparisons = [
        "1 compartment Bayesian trough vs 1 compartment Bayesian peak trough",
        "1 compartment Bayesian trough vs 1 compartment peak trough",
        "2 compartment Bayesian trough vs 1 compartment Bayesian peak trough",
        "2 compartment Bayesian trough vs 2 compartment Bayesian peak trough"
    ]

    start_time = time.time()
    success_count = 0

    for i in range(4):
        print(f"\n📊 Preparing to run: {comparisons[i]}")

        if run_comparison(i):
            success_count += 1
        else:
            print(f"⚠️  Comparison {i} failed, but continuing with remaining comparisons...")

        # Small delay between comparisons
        time.sleep(2)

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Completed {success_count}/4 comparisons successfully")
    print(".1f")
    print("\nOutput files created in merged_output/ directory:")
    print("- statistical_comparison_results_*.txt (statistical results)")
    print("- auc_comparison_grids_*.csv (classification grids)")
    print("- bland_altman_*.png (Bland-Altman plots)")

    if success_count == 4:
        print("\n🎉 All comparisons completed successfully!")
    else:
        print(f"\n⚠️  {4-success_count} comparison(s) failed. Check the output above for details.")

if __name__ == "__main__":
    main()