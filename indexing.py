import numpy as np


def reciprocal_metric_tensor_v2(a, b, c, alpha, beta, gamma):
  """
  alpha, beta, gamma angles are given in degree. Here we calculate the reciprocal metric tensor --> G * G^* = I -> G^* = G^-1
  Once we have the reciprocal metric tensor we can calculate the interplanar distance, and then the 2theta angle. 
  """
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    
    G = np.array([
        [a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
        [a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
        [a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
    ])
    
    G_reciprocal = np.linalg.inv(G)
    
    return G_reciprocal





def interplanar_spacing_v2(h, k, l, G):
    """
    Compute the interplanar spacing d_hkl given Miller indices (h, k, l)
    and the reciprocal metric tensor G.
    """
    d_hkl_sq_inv = (G[0, 0] * h**2 + G[1, 1] * k**2 + G[2, 2] * l**2 + 
                    2 * G[0, 1] * h * k + 2 * G[0, 2] * h * l + 2 * G[1, 2] * k * l) 
    
    if d_hkl_sq_inv <= 0:
        return np.nan  
    
    return 1 / np.sqrt(d_hkl_sq_inv)



def bragg_2theta(d_hkl, wavelength=1.5406e-10):
    """
    Compute the 2-theta diffraction angle given the interplanar spacing d_hkl
    and the X-ray wavelength (default is Cu K-alpha, 1.5406 Å).
    """
    if d_hkl <= 0 or wavelength / (2 * d_hkl) > 1:
        return np.nan  
    
    theta = np.arcsin(wavelength / (2 * d_hkl))
    return np.degrees(2 * theta)  





