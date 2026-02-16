import subprocess
import shutil
import os

# List of weights and Crcl to run
weights = [50, 70, 90, 110] # Set the weight values here (e.g., 50, 70, 90, 110 kg) Please adjust as needed for different weights.
crcl = 120 # Set the Crcl value here (e.g., 30 mL/min) Please adjust as needed for different Crcl values.

# Default inputs for the script (adjust as needed)
default_suffix = ""  # e.g., "" for no suffix, or "_test"

for wt in weights:
    print(f"Running for Crcl {crcl} mL/min, weight {wt} kg...")
    
    # Read the original script
    with open('TwoCompartmentModelLevelComparisons.py', 'r') as f:
        content = f.read()
    
    # Replace the weight and Crcl
    content = content.replace('weight = 110', f'weight = {wt}')
    content = content.replace('Crcl = 30', f'Crcl = {crcl}')
    
    # Write to a temp script
    with open('temp_script.py', 'w') as f:
        f.write(content)
    
    # Prepare inputs for the script
    inputs = f"{default_suffix}\n"
    
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
    output_dir = f'output_twocompartment_Clcr{crcl}_wt{wt}'
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists('output'):
        for file in os.listdir('output'):
            shutil.move(os.path.join('output', file), os.path.join(output_dir, file))
    
    # Clean up temp script
    os.remove('temp_script.py')
    
    print(f"Completed for weight {wt}. Outputs in {output_dir}")

print("All runs completed!")