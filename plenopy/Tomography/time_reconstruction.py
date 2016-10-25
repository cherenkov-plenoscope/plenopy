import numpy as np

def refractive_index_atmosphere(z_asl):
    # refractive index of atmosphere vs. altitude
    z0 = 6500.0
    n_air_depth_1030g = 1.000233
    return 1.0 + (n_air_depth_1030g-1.0)*np.exp(-z_asl/z0)

def t_given_s(s, obs_level=5000):
    c0 = 299792458
    z0 = 6500.0
    n0 = 1.000233
    return 1/c0*(s+z0*(n0-1)*(1 - np.exp(-s/z0)))