import numpy as np

def ramp_kernel(kx=1, ky=1, kz=2):  
    kernel3D = np.zeros(shape=(kz*2+1,kx*2+1,ky*2+1), dtype='float64')
    for z in range(kernel3D.shape[0]):
        for x in range(kernel3D.shape[1]):
            for y in range(kernel3D.shape[2]):
                kernel3D[z,x,y] = np.abs(z-kz) + np.abs(x-kx) + np.abs(y-ky) 
    kernel3D = kernel3D/kernel3D.sum()
