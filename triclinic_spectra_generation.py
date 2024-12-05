import numpy as np
import os
import random
import multiprocessing as mp

from tqdm import tqdm


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


print("Number of CPUs: ", mp.cpu_count())



def generate_coords_basis(n_basis):
    init_tri = [[0, 0, 0]]
    tri = [[0, 0, 0.25], [0, 0.25, 0.25], [0.25, 0.25, 0.25], [0, 0, 0.5], [0, 0.25, 0.5], [0, 0.5, 0.5], [0.25, 0.5, 0.5], [0.5, 0.5, 0.5]]
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

    coords = init_tri + random.sample(tri, n_basis-1)
    elems = random.sample(elements, n_basis)

    return coords, elems

def Lattice_fp(x):
    
    return  Lattice.from_parameters(x[0], x[1], x[2], x[3], x[4], x[5])





def gen_spectra_1(args):
    lattice_params, coords, species = args
    
    lattice = Lattice_fp(lattice_params)
    structures = Structure(lattice, species, coords)
    xrd_calculator = XRDCalculator(wavelength="CuKa")

    pattern = xrd_calculator.get_pattern(structures)

    peaks_ = np.round(pattern.x * 4500/90).astype(int)

    init_array = np.zeros(4501)
    init_array[peaks_] = pattern.y  # This is the spectra of Delta peaks with different intensities

    spectra = convolve(init_array, lorentzian, mode='same')
    spectra = (spectra * 1000) / (np.max(spectra))

    return spectra[:4500], lattice_params


def main():
    """
    Parallel generation of spectra using mp with progression bar 
    """
    
    num_spectra = 100000


    # define parameters:
    coords_l = []
    elems_l = []
    lattice_params_ = []

    mu_l = 10
    mu_ang = 90
    sigma_l = 3
    sigma_ang = 12


    for i in range(num_spectra):
        x, y = generate_coords_basis(n_basis= random.sample([1, 2, 3], 1)[0])
        z  = np.array([mu_l, mu_l, mu_l, mu_ang, mu_ang, mu_ang]) + [sigma_l, sigma_l, sigma_l, sigma_ang, sigma_ang, sigma_ang] * np.random.randn(6)
        coords_l.append(x)
        elems_l.append(y)
        lattice_params_.append(z)
    
    
    params_ = list(zip(lattice_params_, coords_l, elems_l))
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        with tqdm(total=num_spectra) as pbar:  # Progress bar in the main process
            for result in pool.imap_unordered(gen_spectra_1, params_):
                results.append(result)
                pbar.update(1)
    print(f"Generated {len(results)} spectra.")
    return results



if __name__ == "__main__":
    
    results_ = main()

    
    spect = []
    params_ = []
    for i in results_:
        spect.append(i[0])
        params_.append(i[1])

    X_data = np.stack(spect, axis = 0)
    y_data = np.stack(params_, axis = 0)
    np.save(".../X_data_tric_created", X_data)
    np.save(".../y_data_tric_created", y_data)
    print(X_data.shape, y_data.shape)
    






