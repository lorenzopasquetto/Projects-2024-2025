

import numpy as np

def load_data_and_parameters(file_path):
    """
    Spectrum and parameters stored in text like file. For each row the first 4501 numbers correspond to the spctra points.
    The spectra is separated fromm the parameters (six) by "|". 
    
    
    """
    data_list = []
    parameters_list = []
    for index, line in enumerate(file):
        if index != 61692 and index != 61693:  # <--- INSPECT YOUR DATASET AND CHECK FOR MISSING LINES OR NaN LINES

            parts = line.split('|')
            
            if index%5000 == 0: print(index)

            line_before_pipe = parts[0]
            line_values = line_before_pipe.split()[1:]  # Skip the first value ("solo_data") -> each row start with "solo_data"
            if np.array(line_values).shape[0] != 4500:
                print(index, np.array(line_values).shape[0])

            data_list.append([float(value) for value in line_values])
            
            if len(parts) > 1:
                parameters = parts[1].strip().split()  # Extract and split the parameters
                if len(parameters) == 6:  # Ensure there are exactly 6 parameters
                    parameters_list.append(parameters)
                else:
                    raise ValueError(f"Expected 6 parameters, but got {len(parameters)} in row: {line}")
    
    data = np.array(data_list)
    
    parameters = np.array(parameters_list)
    
    return data, parameters # data and parameters of shape (n_spectra, 4501) and (n_spectra, 6)
