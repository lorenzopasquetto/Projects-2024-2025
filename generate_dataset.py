

def generate_coords_basis(n_basis):

    tri = [[0, 0, 0.25], [0, 0.25, 0.25], [0.25, 0.25, 0.25], [0, 0, 0.5], [0, 0.25, 0.5], [0, 0.5, 0.5], [0.25, 0.5, 0.5]]
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


def gen_spectra(lattice_params, coords, species):
    
    lattice = Lattice_fp(lattice_params)
    structures = Structure(lattice, species, coords)
    xrd_calculator = XRDCalculator(wavelength="CuKa")

    pattern = xrd_calculator.get_pattern(structure)

    peaks_ = np.round(pattern.x * 4500/90).astype(int)

    init_array = np.zeros(4501)
    init_array[peaks_] = pattern.y  # This is the spectra of Delta peaks with different intensities

    spectra = convolve(init_array, lorentzian, mode='same')
    spectra = (spectra * 1000) / (np.max(spectra))

    return spectra[:4500], lattice_params

def generate_dataset(n_spectra, mu_l, mu_ang):
    X = []
    y = []
    for i in range(n_spectra):
        if i%100 == 0: print(i)
        lattice_params = np.array([mu_l, mu_l, mu_l, mu_ang, mu_ang, mu_ang]) + [sigma_l, sigma_l, sigma_l, sigma_ang, sigma_ang, sigma_ang] * np.random.randn(6)
        coords, species = generate_coords_basis(n_basis=3)
        X_temp, y_temp = gen_spectra(lattice_params, coords, species)
        X.append(X_temp)
        y.append(y_temp)

    X_data = np.stack(X, axis = 0)
    y_data = np.stack(y, axis = 0)
    return X_data, y_data
