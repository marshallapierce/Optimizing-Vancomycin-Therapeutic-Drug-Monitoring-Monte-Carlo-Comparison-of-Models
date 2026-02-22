import subprocess
import shutil
import os

# This script runs the OneVrsTwoCompartmentModelComparisonGeometricMean.py script for multiple combinations of weights and Crcl values,
#  using default time points and suffix for the outputs in a nested loop.

# List of weights and Crcl to run
weights = [50, 70, 90, 110] # Set the weight values here (e.g., 50, 70, 90, 110 kg) Please adjust as needed for different weights.
crcls = [120, 90, 60, 30]  # Set the Crcl values here (e.g., 120, 90, 60, 30 mL/min) Please adjust as needed for different Crcl values.

# Default inputs for the script (adjust as needed)
default_times = "2,10"  # e.g., "2,10" for peak and trough
default_suffix = ""  # e.g., "" for no suffix, or "_test"

for crcl in crcls:
    for wt in weights:
        print(f"Running for Crcl {crcl} mL/min, weight {wt} kg...")
        
        # Read the original script
        with open('OneVrsTwoCompartmentModelComparisonGeometricMean.py', 'r') as f:
            content = f.read()
        
        # Replace the weight and Crcl
        content = content.replace('weight = 110', f'weight = {wt}')
        content = content.replace('Crcl = 30', f'Crcl = {crcl}')
        
        # Write to a temp script
        with open('temp_script.py', 'w') as f:
            f.write(content)
        
        # Prepare inputs for the script
        inputs = f"{default_times}\n{default_suffix}\n"
        
        # Run the temp script with inputs
        result = subprocess.run(
            ['.venv\\Scripts\\python.exe', 'temp_script.py'],
            input=inputs,
            text=True,
            capture_output=True
        )
        
        # Print output for debugging (optional)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Move outputs to a weight-specific folder
        output_dir = f'output_Clcr{crcl}_wt{wt}'
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists('output'):
            for file in os.listdir('output'):
                shutil.move(os.path.join('output', file), os.path.join(output_dir, file))
        
        # Clean up temp script
        os.remove('temp_script.py')
        
        print(f"Completed for Crcl {crcl}, weight {wt}. Outputs in {output_dir}")

print("All runs completed!")