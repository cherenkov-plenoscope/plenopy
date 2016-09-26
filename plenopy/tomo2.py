import plenopy as plp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

run = plp.Run('demo_big/Fe8/')
evt = run[1]

# cutting ----------------------------------------------------------------------
intensity_threshold = 1
valid_geom = evt.light_field.valid_lixel.flatten()
valid_intensity = evt.light_field.intensity.flatten() >= intensity_threshold
valid_arrival_time = (evt.light_field.arrival_time.flatten() > 30e-9)*(evt.light_field.arrival_time.flatten() < 40e-9)
valid = valid_geom*valid_intensity*valid_arrival_time

rays = plp.Tomography.Rays(
    xs=evt.light_field.x_mean.flatten()[valid],
    ys=evt.light_field.y_mean.flatten()[valid], 
    z=0.0, 
    cxs=evt.light_field.cx_mean.flatten()[valid], 
    cys=evt.light_field.cy_mean.flatten()[valid]) 
    
intensities = evt.light_field.intensity.flatten()[valid]

binning = plp.Tomography.Binning3D(
    z_min=0.0, 
    z_max=25e3, 
    number_z_bins=256,
    xy_diameter=5e2, 
    number_xy_bins=64)

def histogram3D(rays, bins, intensities=None):
    if intensities is None:
        intensities = np.ones(rays.supports.shape[0])

    hist = np.zeros(shape=(
        bins.number_xy_bins,
        bins.number_xy_bins,
        bins.number_z_bins))

    for idx_z, z in enumerate(bins.z_bin_centers):

        xys = rays.xy_intersections_in_z(z)
        hist[:,:,idx_z] = np.histogram2d(
            xys[:,0],
            xys[:,1],
            weights=intensities,
            bins=(bins.xy_bin_edges, bins.xy_bin_edges))[0]

    return hist

def max_intensity_vs_z(hist):
    max_intensity_on_z_slice = np.zeros(hist.shape[2])
    for idx_z in range(hist.shape[2]):
        max_intensity_on_z_slice[idx_z] = np.max(hist[:,:,idx_z])
    return max_intensity_on_z_slice

def med_intensity_vs_z(hist):
    max_intensity_on_z_slice = np.zeros(hist.shape[2])
    for idx_z in range(hist.shape[2]):
        max_intensity_on_z_slice[idx_z] = np.median(hist[:,:,idx_z])
    return max_intensity_on_z_slice

def normalize_ray_histograms(hist_intensities, hist_rays, binning, min_rays_per_xy_area=1.3):
    max_rays_vs_z = max_intensity_vs_z(hist_rays)

    #max_rays_at_z_max = np.max(hist_rays[:,:,-1])
    #min_rays = max_rays_at_z_max
    hist_nrom = np.zeros(shape=hist_intensities.shape)

    for z in range(hist_intensities.shape[2]):
        hi = hist_intensities[:,:,z]
        hist_nrom[:,:,z] = hi/(max_rays_vs_z[z])**(1/3)
    #hist_nrom[hist_rays>min_rays] = hist_intensities[hist_rays>min_rays]/hist_rays[hist_rays>min_rays]
    return hist_nrom

vol_i = histogram3D(rays, binning, intensities)
vol_r = histogram3D(rays, binning)

vol_n = normalize_ray_histograms(vol_i, vol_r, binning)

"""
def ramp_kernel(rx=1, ry=1, rz=2):  
    kernel3D = np.zeros(shape=(rx*2+1, ry*2+1, rz*2+1), dtype='float64')
    for x in range(kernel3D.shape[0]):
        for y in range(kernel3D.shape[1]):
            for z in range(kernel3D.shape[2]):
                kernel3D[x,y,z] = np.abs(z-rz) + np.abs(x-rx) + np.abs(y-ry) 
    kernel3D = kernel3D/kernel3D.sum()
    return kernel3D

kernel3D = ramp_kernel(rx=1,ry=1,rz=1)
"""

def ramp_kernel_frequency(binning):
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

def flatten(hist, binning, threshold=0):
    xyzi = []
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            for z in range(hist.shape[2]):
                if hist[x,y,z] > threshold:
                    xyzi.append(np.array([
                        binning.xy_bin_centers[x],
                        binning.xy_bin_centers[y],
                        binning.z_bin_centers[z],
                        hist[x,y,z]]))
    xyzi = np.array(xyzi)
    return xyzi

def plot(xyzIs, xyzIs2=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    inte = xyzIs[:,3]
    inte = 500*inte/inte.max()

    ax.scatter(
        xyzIs[:,0], xyzIs[:,1], xyzIs[:,2],
        s=inte,
        depthshade=False,
        alpha=0.01,
        lw=0)

    if xyzIs2 is not None:
        inte2 = xyzIs2[:,3]
        inte2 = 500*inte2/inte2.max()

        ax.scatter(
            xyzIs2[:,0], xyzIs2[:,1], xyzIs2[:,2],
            s=inte,
            c='r',
            depthshade=False,
            alpha=0.01,
            lw=0)        

    plt.show()

def histogram3DAirShowerPhotons(photons, binning, observation_level=5e3):

    supports = np.array([
        photons.x,
        photons.y,
        observation_level*np.ones(photons.x.shape[0])]).T

    directions = np.array([
        photons.cx,
        photons.cy,
        np.sqrt(1.0 - photons.cx**2 - photons.cy**2)]).T

    a = (photons.emission_height - supports[:,2])/directions[:,2]

    emission_positions = np.array([
            supports[:,0] - a*directions[:,0],
            supports[:,1] - a*directions[:,1],
            photons.emission_height
        ]).T

    # transform to plenoscope frame
    emission_positions[:,2] = emission_positions[:,2] - observation_level

    hist = np.histogramdd(
        emission_positions, bins=(
            binning.xy_bin_edges, 
            binning.xy_bin_edges, 
            binning.z_bin_edges))

    return hist[0]

#volIFil = ndimage.convolve(vol_n, kernel3D)

ramp = ramp_kernel_frequency(binning)

ft_vol_n = np.fft.fftn(vol_n)
ft_vol_n_filtered = ft_vol_n*ramp
vol_n_filtered = np.abs(np.fft.ifftn(ft_vol_n_filtered))

shower = flatten(histogram3DAirShowerPhotons(evt.simulation_truth.air_shower_photon_bunches, binning), binning)
#recons = flatten(volIFil, binning)
recon2 = flatten(vol_n_filtered, binning)