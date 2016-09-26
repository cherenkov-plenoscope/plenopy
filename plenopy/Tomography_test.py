import plenopy as pop
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

run = pop.Run('demo_big/Fe8/')
evt = run[1]

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
#idx = np.arange(intensity.shape[0])

fast = 1
nbins_xy = 64/fast
nbins_z = 256/fast
voxel_xy_radius = 1*fast
voxel_z_radius = 6.6*fast
#voxelArea = (voxel_xy_radius**2.0)*4.0

xyBinEdges = np.linspace(
    -nbins_xy*voxel_xy_radius, 
    nbins_xy*voxel_xy_radius, 
    nbins_xy+1)

zBinEdges = np.linspace(
    5e3+3.0*evt.light_field.expected_focal_length_of_imaging_system,
    5e3+3.0*evt.light_field.expected_focal_length_of_imaging_system + 2.0*nbins_z*voxel_z_radius,
    nbins_z+1)

xyBinCenters = xyBinEdges[:-1]+voxel_xy_radius
zBinCenters = zBinEdges[:-1]+voxel_z_radius

#raysRefVoxel = []
#for i in intensity:
#    raysRefVoxel.append([])

def xyRayIntersectOnSlice(z):
    scale_factors = z/direction[:, 2]
    pos3D = support - (scale_factors*direction.T).T
    return pos3D[:, 0:2]

def histogramSlice(z):
    xy = xyRayIntersectOnSlice(z)
    return np.histogram2d(
        xy[:,0],
        xy[:,1],
        bins=(xyBinEdges,xyBinEdges))[0]

def histogramVolume():
    volRayCount = []
    for z in tqdm(zBinEdges[:-1]):
        volRayCount.append(histogramSlice(z))
    volRayCount = np.array(volRayCount)
    return volRayCount

def histogramSliceIntensity(z):
    xy = xyRayIntersectOnSlice(z)
    return np.histogram2d(
        xy[:,0],
        xy[:,1],
        weights=intensity,
        bins=(xyBinEdges,xyBinEdges))[0]

def histogramVolumeIntensity():
    volIntensity = []
    for z in tqdm(zBinEdges[:-1]):
        volIntensity.append(histogramSliceIntensity(z))
    volIntensity = np.array(volIntensity)
    return volIntensity

"""def rampFilterZ(vol, kernel):

    kernelWidth = kernel.shape[0] - 1

    xs = vol.shape[1]
    ys = vol.shape[2]

    volFil = np.zeros(shape=(vol.shape[0]- kernelWidth, xs, ys))

    for x in range(vol.shape[1]):
        for y in range(vol.shape[2]):
            volFil[:,x,y] = np.convolve(vol[:,x,y], kernel, mode='valid')

    return volFil
"""
def flatten(vol, threshold=0):
    xyzi = []
    for z in tqdm(range(vol.shape[0])):
        for x in range(vol.shape[1]):
            for y in range(vol.shape[2]):
                if vol[z,x,y] > threshold:
                    xyzi.append(np.array([
                        xyBinCenters[x],
                        xyBinCenters[y],
                        zBinCenters[z],
                        vol[z,x,y]]))
    xyzi = np.array(xyzi)
    return xyzi

"""
def stdIvsZ(vol):
    maxI = np.zeros(vol.shape[0])
    for i, zSlice in enumerate(vol):
        maxI[i] = np.std(zSlice)
    return maxI

def meanIvsZ(vol):
    maxI = np.zeros(vol.shape[0])
    for i, zSlice in enumerate(vol):
        maxI[i] = np.mean(zSlice)
    return maxI
"""
def maxIvsZ(vol):
    maxI = np.zeros(vol.shape[0])
    for i, zSlice in enumerate(vol):
        maxI[i] = np.max(zSlice)
    return maxI

def trueShower():
    x = evt.simulation_truth.air_shower_photons.x
    y = evt.simulation_truth.air_shower_photons.y
    r = np.sqrt(x**2 + y**2)
    cx = evt.simulation_truth.air_shower_photons.cx
    cy = evt.simulation_truth.air_shower_photons.cy
    h = evt.simulation_truth.air_shower_photons.emission_height

    # restrict to aperture 
    apertutre_radius = 50
    validR = r < apertutre_radius
    validC = np.abs(np.sqrt(cx**2 + cy**2)) < np.deg2rad(3.35)
    valid = validR*validC
    x = x[valid]
    y = y[valid]
    cx = cx[valid]
    cy = cy[valid]
    h = h[valid]

    sups = np.array([x,y,5e3*np.ones(x.shape[0])]).T

    dirs = np.array([cx,cy,np.sqrt(1.0 - cx**2 - cy**2)]).T

    a = (h - sups[:,2])/dirs[:,2]

    pos = np.array([
            sups[:,0] - a*dirs[:,0],
            sups[:,1] - a*dirs[:,1],
            h
        ]).T

    # transform to plenoscope frame
    pos[:,2] = pos[:,2] - 5e3

    hist = np.histogramdd(pos, bins=(xyBinEdges, xyBinEdges, zBinEdges))

    bins = hist[0]

    trueShower = []
    for x in tqdm(range(bins.shape[0])):
        for y in range(bins.shape[1]):
            for z in range(bins.shape[2]):
                if bins[x,y,z] > 0:
                    trueShower.append(np.array([
                        xyBinCenters[x],
                        xyBinCenters[y],
                        zBinCenters[z],
                        bins[x,y,z]]))

    trueShower = np.array(trueShower)
    return trueShower

def normLF(volI, volC):
    volO = np.zeros(shape=volI.shape)
    mCz = maxIvsZ(volC)
    mCz /= mCz.mean()
    mCz[mCz < 2.0] = 2.0
    mCz /= mCz.mean()
    for z, mc in enumerate(mCz):
        volO[z] = volI[z]/mc
    return volO

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

volI = histogramVolumeIntensity()
volC = histogramVolume()

volN = normLF(volI, volC)

kx = 1
ky = kx
kz = 2
kernel3D = np.zeros(shape=(kz*2+1,kx*2+1,ky*2+1), dtype='float64')
#kernel3D *= -1.0 
#kernel3D[kz*2-1,kx,ky] = 1 - kernel3D.sum()
for z in range(kernel3D.shape[0]):
    for x in range(kernel3D.shape[1]):
        for y in range(kernel3D.shape[2]):
            kernel3D[z,x,y] = np.abs(z-kz) + np.abs(x-kx) + np.abs(y-ky) 
kernel3D = kernel3D/kernel3D.sum()

kernelDZ = np.array([
    [[+1.0]],
    [[-0.5]],
    ]
    )

volIFil = ndimage.convolve(volN, kernel3D)
volIFil = ndimage.convolve(volN, kernelDZ)


reconsI = flatten(volIFil, 25)
shower = trueShower()

plot(reconsI[reconsI[:,3]>500], shower[shower[:,3]>10])


"""
def initVoxelPlane(z):
    voxels = []
    for x in xyBinEdges:
        voxelSlice = []
        for y in xyBinEdges:
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
    voxelAboveThreshold = []
    for voxelSlice in voxelPlane:
        for voxel in voxelSlice:

            voxLixelDensity = len(voxel.rays)/voxelArea
            voxPhotonDensity = intensity[voxel.rays].sum()/voxelArea
            if voxPhotonDensity > 3*voxLixelDensity and voxLixelDensity > 0.075: 
            #if len(voxel.rays) > 5 and voxPhotonDensity > 5.0*voxLixelDensity:
                voxelAboveThreshold.append(voxel)

                for ray in voxel.rays:
                    raysRefVoxel[ray].append(voxel)

    return voxelAboveThreshold

def makeVoxelVolume():
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

def plot(xyzIs, xyzIs2=None):
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

    if xyzIs2 is not None:
        inte2 = xyzIs2[:,3]
        inte2 = 500*inte2/inte2.max()

        ax.scatter(
            xyzIs2[:,0], xyzIs2[:,1], xyzIs2[:,2],
            s=inte,
            c='r',
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
        idx = dI_dz.shape[0]
        for i in range(dI_dz.shape[0]):
            idx -= 1
            if dI_dz[idx] > 2.5:
                break

        plane = idx
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

        print('intensity', inte, 'in', len(voxelSub.rays), 'rays at', voxelSub.pos, 'm')
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
"""



