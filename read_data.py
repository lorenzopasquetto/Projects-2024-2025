

import numpy as np

def load_data_and_parameters(file_path):
    # Create lists to store the numerical data (SPECTRA) and parameters
    data_list = []
    parameters_list = []
    for index, line in enumerate(file):
        if index != 61692 and index != 61693:  # <--- INSPECT YOUR DATASET AND CHECK FOR MISSING LINES OR NaN LINES

            # Split the line at the '|' symbol
            parts = line.split('|')
            
            if index%5000 == 0: print(index)

            # Process the part before '|' (numerical data)
            line_before_pipe = parts[0]
            line_values = line_before_pipe.split()[1:]  # Skip the first value ("solo_data")
            if np.array(line_values).shape[0] != 4500:
                print(index, np.array(line_values).shape[0])

            data_list.append([float(value) for value in line_values])
            
            # Process the part after '|' (parameters)
            if len(parts) > 1:
                parameters = parts[1].strip().split()  # Extract and split the parameters
                if len(parameters) == 6:  # Ensure there are exactly 6 parameters
                    parameters_list.append(parameters)
                else:
                    raise ValueError(f"Expected 6 parameters, but got {len(parameters)} in row: {line}")
    
    # Convert the numerical data list to a NumPy array
    data = np.array(data_list)
    
    parameters = np.array(parameters_list)
    
    return data, parameters
