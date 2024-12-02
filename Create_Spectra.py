mport numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
from scipy.signal import convolve


def lorentzian_kernel(x, gamma = 0.01):
    return gamma / (np.pi * (x**2 + gamma**2))

a = np.arange(-2, 2, 0.01)
lorentzian = lorentzian_kernel(a)
lorentzian /= np.sum(lorentzian_kernel(a))



def create_spectra_cubic(lattice_parameter, species):
    
    a = lattice_parameter  
    lattice = Lattice.cubic(a)

    # Atomic species and their coordinates in the unit cell
    species = [species, species]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]

    # Create structure
    structure = Structure(lattice, species, coords)

    xrd_calculator = XRDCalculator(wavelength="CuKa")

    pattern = xrd_calculator.get_pattern(structure)

    peaks_ = np.round(pattern.x * 4500/90).astype(int)


    init_array = np.zeros(4501)
    init_array[peaks_] = pattern.y  # This is the spectra of Delta peaks with different intensities

    spectra = convolve(init_array, lorentzian, mode='same')
    spectra = (spectra * 1000) / (np.max(spectra))



def create_dataset(elements, range_val=25, min_displ = 0.1):
    X_data = [None]
    parameters = [None]
    for i in elements:
            for j in range(range_val):
                    
                    lattice_parameters = [2.5 + min_displ*j, 2.5 + min_displ*j, 2.5 +  min_displ*j, 90, 90, 90]
                    X = create_spectra_cubic(lattice_parameter=2.5 + (0.1 * j), species=i)
                    X_data.append(X)
                    parameters.append(lattice_parameters)
                    print("Element: ", i, "Iteration: ", j, "--> ", lattice_parameters[0], "\u212B")
    return np.stack(X_data[1:]), np.stack(parameters[1:])
                    
                    
    
