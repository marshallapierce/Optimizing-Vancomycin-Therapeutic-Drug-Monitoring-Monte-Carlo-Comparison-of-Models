#!/usr/bin/env python3
"""
Script to run individual statistical comparisons one at a time
"""
import sys
import os

def run_comparison(comparison_index):
    """Run a specific comparison by index"""
    venv_python = r"c:\Users\mpier\Python_Monte_Carlo_2\.venv\Scripts\python.exe"
    script_path = r"c:\Users\mpier\Python_Monte_Carlo_2\statistical_comparison_one_two_compart_bay_tr_vs_one_two_compart_nonbay_pk_tr_bay_pk_tr.py"
    cmd = f'"{venv_python}" "{script_path}" {comparison_index}'
    print(f"Running comparison {comparison_index}: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single_comparison.py <comparison_index>")
        print("Comparison indices:")
        print("  0: 1 compartment Bayesian trough vs 1 compartment Bayesian peak trough")
        print("  1: 1 compartment Bayesian trough vs 1 compartment peak trough")
        print("  2: 2 compartment Bayesian trough vs 1 compartment Bayesian peak trough")
        print("  3: 2 compartment Bayesian trough vs 2 compartment Bayesian peak trough")
        sys.exit(1)

    try:
        comparison_index = int(sys.argv[1])
        if comparison_index < 0 or comparison_index > 3:
            print("Comparison index must be 0-3")
            sys.exit(1)
    except ValueError:
        print("Comparison index must be a number")
        sys.exit(1)

    run_comparison(comparison_index)