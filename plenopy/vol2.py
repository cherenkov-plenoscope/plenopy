import plenopy as pop
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

run = pop.Run('demo_big/Fe8/')
evt = run[1]

class Voxel(object):
    def __init__(self, pos):
        self.pos = pos
        self.rays = []

intensityThresh = 1
# only valid lixels
validGeom = evt.light_field.valid_lixel.flatten()
validIntensity = evt.light_field.intensity.flatten() >= intensityThresh
validArrivalTime = (evt.light_field.arrival_time.flatten() > 30e-9)*(evt.light_field.arrival_time.flatten() < 40e-9)

valid = validGeom*validIntensity*validArrivalTime

# set up the lixel rays
support = evt.light_field.rays.support[valid]
direction = evt.light_field.rays.direction[valid]
intensity = evt.light_field.intensity.flatten()[valid]
idx = np.arange(intensity.shape[0])

fast = 2
nbins_xy = 256/fast
nbins_z = 256/fast
voxel_xy_radius = 2*fast
voxel_z_radius = 39.0625*fast

raysRefVoxel = []
for i in intensity:
    raysRefVoxel.append([])

def xyOnSlice(z):
    scale_factors = z/direction[:, 2]
    pos3D = support - (scale_factors*direction.T).T
    return pos3D[:, 0:2]

def initVoxelPlane(z):
    voxels = []
    xybins = np.linspace(
        -nbins_xy*voxel_xy_radius, 
        nbins_xy*voxel_xy_radius, 
        nbins_xy)
    for x in xybins:
        voxelSlice = []
        for y in xybins:
            voxelSlice.append(Voxel(np.array([x,y,z])))
        voxels.append(voxelSlice)
    return voxels

def xy2VoxelIdx(xy):
    return np.array([
        np.round((xy[:,0]/voxel_xy_radius)+(nbins_xy/2)).astype('int64'),
        np.round((xy[:,1]/voxel_xy_radius)+(nbins_xy/2)).astype('int64')]).T

def voxelPlane(z):
    xys = xyOnSlice(z)
    voxelPlane = initVoxelPlane(z)
    xyVoxelIdxs = xy2VoxelIdx(xys)
    for i, xyVoxelIdx in enumerate(xyVoxelIdxs):
        
        if xyVoxelIdx[0] >= 0 and xyVoxelIdx[0] < nbins_xy and xyVoxelIdx[1] >= 0 and xyVoxelIdx[1] < nbins_xy:
            voxelPlane[xyVoxelIdx[0]][xyVoxelIdx[1]].rays.append(i)
    voxelArea = voxel_xy_radius**2.0*4.0
    voxelAboveThreshold = []
    for voxelSlice in voxelPlane:
        for voxel in voxelSlice:

            voxLixelDensity = len(voxel.rays)/voxelArea
            voxPhotonDensity = intensity[voxel.rays].sum()/voxelArea
            if voxPhotonDensity > 5.0*voxLixelDensity and voxLixelDensity > 0.075: 
                voxelAboveThreshold.append(voxel)

                for ray in voxel.rays:
                    raysRefVoxel[ray].append(voxel)

    return voxelAboveThreshold

def makeVoxelVolume():
    zbins = np.linspace(
        voxel_z_radius, 
        2.0*nbins_z*voxel_z_radius, 
        nbins_z)
    voxelVolume = []

    for z in tqdm(zbins):
        voxelVolume.append(voxelPlane(z))

    return voxelVolume
    
def xyzIVoxel(voxelVolume):
    xyzIs = []

    for plane in voxelVolume:
        for voxel in plane:
            xyzI = np.array([
                voxel.pos[0],
                voxel.pos[1],
                voxel.pos[2],
                intensity[voxel.rays].sum()])
            xyzIs.append(xyzI)

    xyzIs = np.array(xyzIs)
    return xyzIs

def maxIntensityVsZ(voxelVolume):

    mIz = np.zeros(shape=(len(voxelVolume), 3))

    for z, plane in enumerate(voxelVolume):
        mI = 0
        mIdx = 0
        mz = 0
        for v, voxel in enumerate(plane):
            inte = intensity[voxel.rays].sum()
            if inte > mI:
                mI = inte
                mIdx = v
                mz = voxel.pos[2]

        mIz[z] = np.array([mI, mz, mIdx])

    return mIz 

def plot(xyzIs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    inte = xyzIs[:,3]
    inte = 500*inte/inte.max()

    ax.scatter(
        xyzIs[:,0], xyzIs[:,1], xyzIs[:,2],
        s=inte,
        depthshade=False,
        alpha=0.05,
        lw=0)
    plt.show()

def voxelToBeSubtracted(vol):

    mIz = maxIntensityVsZ(vol)
    dI_dz = np.gradient(mIz[:,0])

    if (dI_dz > 0.0).sum() == 0:
        return None
    else:
        plane = dI_dz.argmax()
        return vol[plane][mIz[plane, 2].astype('int64')]

def removeRay(ray):
    for voxel in raysRefVoxel[ray]:
        voxel.rays.remove(ray)

def subtractLixel(vol):
    shower = []

    while True:

        voxelSub = voxelToBeSubtracted(vol)

        if voxelSub is None:
            break

        inte = intensity[voxelSub.rays].sum()
        if inte < 1:
            break

        print('intensity', inte, 'in', len(voxelSub.rays), 'rays at', voxelSub.pos[2], 'm')
        shower.append(
            np.array([
                voxelSub.pos[0],
                voxelSub.pos[1],
                voxelSub.pos[2],
                inte])
            )

        for ray in voxelSub.rays:
            removeRay(ray)

    return np.array(shower)

vol = makeVoxelVolume()
shower = subtractLixel(vol)




