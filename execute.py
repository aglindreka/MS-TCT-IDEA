import os
import subprocess
import re

# Directory containing the .pkl files
# pkl_directory = "/data/stars/user/areka/MS-TCT/save_logit_ALL_KL/"
pkl_directory = '/data/stars/user/areka/MS-TCT/save_logit_ALL_mae_46'
# Get a list of all .pkl files in the directory
pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith('.pkl')]

# Initialize variables to store the maximum value and the corresponding file
max_value = -1
max_file = ""

# Iterate through each .pkl file
for pkl_file in pkl_files:
    # Construct the full path to the .pkl file
    pkl_path = os.path.join(pkl_directory, pkl_file)
    
    # Run the Evaluation.py script and capture the output
    result = subprocess.run(
        ["python", "Evaluation.py", "-pkl_path", pkl_path],
        capture_output=True,
        text=True
    )
    
    # Extract the printed value from the output using regex
    match = re.search(r"Test Frame-based map tensor\((\d+\.\d+)\)", result.stdout)
    if match:
        value = float(match.group(1))
        print(f"File: {pkl_file}, Value: {value}")
        
        # Update the maximum value and corresponding file if necessary
        if value > max_value:
            max_value = value
            max_file = pkl_file
    else:
        print(f"File: {pkl_file}, No value found in output.")

# Print the largest value and the corresponding file
print(f"\nLargest value: {max_value} from file: {max_file}")
