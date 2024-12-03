import numpy as np

import os
from pymatgen.io.cif import CifParser


from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
from scipy.signal import convolve


def lorentzian_kernel(x, gamma = 0.01):
    return gamma / (np.pi * (x**2 + gamma**2))

a = np.arange(-2, 2, 0.01)
lorentzian = lorentzian_kernel(a)
lorentzian /= np.sum(lorentzian_kernel(a))


def create_spectra_cubic(lattice_parameter, species, species2 = None):
    if species2 == None: species2 = species
    
    a = lattice_parameter  
    lattice = Lattice.cubic(a)

    # Atomic species and their coordinates in the unit cell
    species = [species, species2]
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
    return spectra
    


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
                    
elements = [
    "H", "D", "T", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf"
    ]


cif_folder = "/Users/loernzopasquetto/Desktop/cod/cif/7/00/01"

def Lattice_fp(x):
    
    
    return  Lattice.from_parameters(x[0], x[1], x[2], x[3], x[4], x[5])

def read_folder(cif_folder):
    """
    reads the folder and gives:
    1- the coords of elements in the basis
    2- the lattice parameters
    3- the species
    """
    species = []
    coord = []
    lattice_params = []


    for ind, i in enumerate(os.listdir(cif_folder)):
        cif_path = os.path.join(cif_folder, i)
        
                
        try:
            parser = CifParser(cif_path)
            structures = parser.get_structures()
            if structures:
                #print("Parsed structure:", structures[0])
                for struc in structures:
                    species.append(struc.species)
                    coord.append(struc.frac_coords)
                    lattice_params.append(struc.lattice.parameters)
                    
            else:
                print("No structures found in the CIF file.")
        except Exception as e:
            print("Error parsing CIF file:", e)
    
    return species, coord, lattice_params


"""
Example:
# Read all the .cif file in the folder and plot the spectra number 33


cif_folder = "~/Desktop/cod/cif/7/00/01"
X, Y, Z = read_folder(cif_folder)

structure = Structure(Lattice_fp(Z[33]), X[33], Y[33])
xrd_calculator = XRDCalculator(wavelength="CuKa")

pattern = xrd_calculator.get_pattern(structure) 

peaks_ = np.round(pattern.x * 4500/90).astype(int)
init_array = np.zeros(4501)
init_array[peaks_] = pattern.y  # This is the spectra of Delta peaks with different intensities

spectra = convolve(init_array, lorentzian, mode='same')
spectra = (spectra * 1000) / (np.max(spectra))
final_spectra = spectra[:4500] # final shape is (4500, 1)

"""



#################################################
count = 0
range_val=25
min_displ = 0.1
X_data = [None]
parameters = [None]

for i in elements:
    for k in range(len(elements)):
        
        species = i
        species2 = elements[k]
        
        for j in range(range_val):
                    
                    lattice_parameters = [2.5 + min_displ*j, 2.5 + min_displ*j, 2.5 +  min_displ*j, 90, 90, 90]
                    X = create_dataset_cubic(lattice_parameter=2.5 + (0.1 * j), species=i)
                    X_data.append(X)
                    if len(X_data)%10: print(len(X_data))
                    parameters.append(lattice_parameters)
                    #print("Elements: ", i, elements[k], "Iteration: ", j, "--> ", lattice_parameters[0], "\u212B")
        
        
    
