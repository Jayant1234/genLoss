import re

# Function to remove lines with progress bars
def remove_progress_bars(input_file, output_file):
    # Define the pattern for detecting progress bars (this example looks for the █ character)
    progress_bar_pattern = r"█|▒|░|┈"  # Add more characters here if necessary
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Skip lines that match the progress bar pattern
            if not re.search(progress_bar_pattern, line):
                outfile.write(line)

# Usage example
input_file = 'sgd_baseline_ouput_o.log'
output_file = 'cleaned_file.txt'
remove_progress_bars(input_file, output_file)
