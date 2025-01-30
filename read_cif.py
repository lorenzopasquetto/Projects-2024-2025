import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm  # For progress tracking
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial


from pymatgen.io.cif import CifParser
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import warnings

# Ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Functions used inside parallel process

def extrapolate_cs(path):
    try:
        parser = CifParser(path)
        structure = parser.parse_structures()[0]
        analyzer = SpacegroupAnalyzer(structure)
        lattice = structure.lattice
        crystal_system = analyzer.get_crystal_system() # Crystal system defined by the space group. See https://github.com/materialsproject/pymatgen/blob/v2025.1.23/src/pymatgen/symmetry/analyzer.py#L202-L230
        
        return [crystal_system, lattice.abc, lattice.angles]
    
    except (TypeError, ValueError) as e:
        print("Error: ", e)
        return None


def create_spectra(path):
    try:
        parser = CifParser(path)
        structure = parser.parse_structures()[0]
        xrd_calc = XRDCalculator(wavelength="CuKa")
        pattern = xrd_calc.get_pattern(structure)
        lattice = structure.lattice
        pattern = [pattern, lattice.abc, lattice.angles]
        
        return pattern
    
    except (TypeError, ValueError) as e:
        print("Error: ", e)
        return None



# Intensive CPU task -> ProcessPoolExecutor
# Ideal to merge the crystal systems reading process and the spectra creation -> modify extrapolate_cs and create_spectra functions

def process_files_parallel(folder_path, max_workers, limit = 8000):
    with os.scandir(folder_path) as entries:
        file_entries = [entry.path for entry in entries if entry.is_file() and entry.name.endswith(".CIF")][:limit]
 


    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(create_spectra, file_entries),
            total=len(file_entries),
            desc="Processing CIF files"
        ))

    return results

def process_files_parallel_cs(folder_path, max_workers, limit = 540000):
    with os.scandir(folder_path) as entries:
        file_entries = [entry.path for entry in entries if entry.is_file() and entry.name.endswith(".CIF")][:limit]
 


    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(extrapolate_cs, file_entries),
            total=len(file_entries),
            desc="Processing CIF files"
        ))

    return results




dir = "/home/lopasq/data/GNoME_data/by_composition"


def translate_spectra(pattern):

        peaks_ = np.round(pattern[0].x * 4500/90).astype(int)

        init_array = np.zeros(4501)
        init_array[peaks_] = pattern[0].y  # This is the spectra of Delta peaks with different intensities

        spectra = convolve(init_array, lorentzian, mode='same')
        spectra = (spectra * 1000) / (np.max(spectra))
        lattice_params = [pattern[1], pattern[2]]
        lattice_params = [item for sublist in lattice_params for item in sublist]
        return spectra, lattice_params

def lorentzian_kernel(x, gamma = 0.01):
        return gamma / (np.pi * (x**2 + gamma**2))





if __name__ == "__main__":
    # Run the parallel CIF processing
    print("Cores available: ", os.cpu_count())
    limit = 540000
    peak_arrays = process_files_parallel(dir, max_workers=os.cpu_count(), limit = limit)  # Str of arrays
    cs_lattice = process_files_parallel_cs(dir, max_workers=os.cpu_count(), limit = limit)




    
    # reduce memory load saving the np arrays of dimensions (20000,4501)
    spectrum = []
    lattices = []
    index_file = 0
    
    a = np.arange(-2, 2, 0.01)

    lorentzian = lorentzian_kernel(a)
    lorentzian /= np.sum(lorentzian_kernel(a))
        
    for ind, i in enumerate(peak_arrays):

        if (ind % 20000 == 0 and ind != 0) or ind == limit-1:
            X_data = np.stack(spectrum, axis = 0)
            y_data = np.stack(lattices, axis=0)
            print(index_file)
            np.save(f'/home/lopasq/data/GNoME_data/GNoME_comp/X_data_{index_file}', X_data)
            np.save(f'/home/lopasq/data/GNoME_data/GNoME_comp/y_data_{index_file}', y_data)
            index_file +=1
            spectrum = []
            lattices = []



        spectra, lattice_params = translate_spectra(i)
        spectrum.append(spectra)
        lattices.append(lattice_params)








