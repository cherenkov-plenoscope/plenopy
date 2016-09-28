import plenopy as plp
import numpy as np
import matplotlib.pyplot as plt

run = plp.Run('demo_big/Fe8/')
evt = run[5]

# cutting ----------------------------------------------------------------------
intensity_threshold = 1
valid_geom = evt.light_field.valid_lixel.flatten()
valid_intensity = evt.light_field.intensity.flatten() >= intensity_threshold
valid_arrival_time = (evt.light_field.arrival_time.flatten() > 30e-9)*(evt.light_field.arrival_time.flatten() < 40e-9)
valid = valid_geom*valid_intensity*valid_arrival_time

rays = plp.LixelRays(
    x=evt.light_field.x_mean.flatten()[valid],
    y=evt.light_field.y_mean.flatten()[valid],
    cx=evt.light_field.cx_mean.flatten()[valid], 
    cy=evt.light_field.cy_mean.flatten()[valid]) 
    
intensities = evt.light_field.intensity.flatten()[valid]

binning = plp.Tomography.Binning(
    z_min=0.0, 
    z_max=25e3, 
    number_z_bins=256,
    xy_diameter=5e2, 
    number_xy_bins=64)

shower_hist = plp.Tomography.histogram3DAirShowerPhotons(
    evt.simulation_truth.air_shower_photon_bunches, 
    binning)

shower = plp.Tomography.flatten(shower_hist, binning)
recon_hist = plp.Tomography.filtered_back_projection(rays, intensities, binning)
recon = plp.Tomography.flatten(recon_hist, binning)