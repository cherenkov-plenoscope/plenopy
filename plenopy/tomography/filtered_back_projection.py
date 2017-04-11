import numpy as np


def histogram(rays, binning, intensities=None):
    if intensities is None:
        intensities = np.ones(rays.support.shape[0])

    hist = np.zeros(shape=(
        binning.number_xy_bins,
        binning.number_xy_bins,
        binning.number_z_bins))

    for idx_z, z in enumerate(binning.z_bin_centers):

        xys = rays.xy_intersections_in_object_distance(z)
        hist[:,:,idx_z] = np.histogram2d(
            xys[:,0],
            xys[:,1],
            weights=intensities,
            bins=(binning.xy_bin_edges, binning.xy_bin_edges))[0]

    return hist


def max_intensity_vs_z(hist):
    max_intensity_on_z_slice = np.zeros(hist.shape[2])
    for idx_z in range(hist.shape[2]):
        max_intensity_on_z_slice[idx_z] = np.max(hist[:,:,idx_z])
    return max_intensity_on_z_slice


def normalize_ray_histograms(hist_intensities, hist_rays):
    max_rays_vs_z = max_intensity_vs_z(hist_rays)
    hist_nrom = np.zeros(shape=hist_intensities.shape)

    for z in range(hist_intensities.shape[2]):
        hi = hist_intensities[:,:,z]
        hist_nrom[:,:,z] = hi/(max_rays_vs_z[z])**(1/3)
    return hist_nrom


def ramp_kernel_in_frequency_space(binning):
    """
    ramp_pos = binning.flat_xyz_voxel_positions_in_frequency_space()

    dist_to_origin = np.linalg.norm(ramp_pos, axis=1)

    ramp = np.zeros(
        binning.number_xy_bins*binning.number_xy_bins*binning.number_z_bins,
        dtype='float64')
    
    d = dist_to_origin
    #ramp = (-d**2 + 1*d)*4
    ramp = d**2.0
    ramp = ramp/ramp.max()
    return ramp.reshape(
        binning.number_xy_bins,
        binning.number_xy_bins,
        binning.number_z_bins)
    """
    ramp = np.zeros(shape=(
        binning.number_xy_bins, 
        binning.number_xy_bins, 
        binning.number_z_bins),
        dtype='float64')

    xw = np.linspace(1,-1,binning.number_xy_bins)
    xw = np.abs(xw)

    yw = np.linspace(1,-1,binning.number_xy_bins)
    yw = np.abs(yw)

    zw = np.linspace(1,-1,binning.number_z_bins)
    zw = np.abs(zw)

    for x in range(ramp.shape[0]):
        for y in range(ramp.shape[1]):
            ramp[x,y,:]+=zw

    for y in range(ramp.shape[1]):
        for z in range(ramp.shape[2]):
            ramp[:,y,z]+=xw

    for z in range(ramp.shape[2]):
        for x in range(ramp.shape[0]):
            ramp[x,:,z]+=yw
    
    ramp = ramp/ramp.max()
    return ramp

def frequency_filter(hist, kernel):
    ft_hist = np.fft.fftn(hist)
    ft_hist_filtered = ft_hist*kernel
    return np.abs(np.fft.ifftn(ft_hist_filtered))