#!/usr/bin/env ipython

import yaml
import numpy as np
import tifffile
import os
from scipy.io import loadmat
from time import time

from pyolaf.geometry import LFM_computeGeometryParameters, LFM_setCameraParams
from pyolaf.lf import LFM_computeLFMatrixOperators
from pyolaf.transform import LFM_retrieveTransformation, format_transform, get_transformed_shape, transform_img
from pyolaf.aliasing import lanczosfft, LFM_computeDepthAdaptiveWidth

from pyolaf.project import LFM_forwardProject, LFM_backwardProject

try:
    import cupy
    from cupy.fft import fftshift, ifft2, fft2
    has_cupy = True
except ImportError:
    print('Cupy not available. Falling back to numpy and scipy.')
    cupy = np
    from numpy.fft import fftshift, ifft2, fft2
    has_cupy = False

if has_cupy:
    mempool = cupy.get_default_memory_pool()
    mempool.set_limit(7.5 * 2**30)



# see this url for samples:
#   https://drive.google.com/drive/folders/1clAUjal3P0a2owQrwGvdpUAoCHYwSecb?usp=share_link

# change this to your data path
data_path = '/home/lili/data/light-field/samples-pyolaf/fly-muscles-GFP'

fname_calib = os.path.join(data_path, 'calib.tif')
fname_config = os.path.join(data_path, 'config.yaml')
fname_img = os.path.join(data_path, 'example_fly.tif')

WhiteImage = tifffile.imread(fname_calib)
LensletImage = tifffile.imread(fname_img, key=0)

## PARAMETERS
# depth range (in mm)
depthRange = [-150, 200]
# depth step
depthStep = 50

# choose lenslet spacing (in  pixels) to downsample the number of pixels between mlens for speed up
newSpacingPx = 15
# choose super-resolution factor as a multiple of lenslet resolution (= 1 voxel/lenslet)
superResFactor = 10

# window size for anti-aliasing filter
lanczosWindowSize = 2
# whether to enable the antialiasing filter
filterFlag = True

# number of iterations of deconvolution
niter = 1

DebugBuildGridModel = False


## start of computations
# Specific LFM configuration and camera parameters (um units)
Camera = LFM_setCameraParams(fname_config, newSpacingPx)

# Compute LFPSF Patterns and other prerequisites: lenslets centers, resolution related
LensletCenters, Resolution, LensletGridModel, NewLensletGridModel = \
    LFM_computeGeometryParameters(
        Camera, WhiteImage, depthRange, depthStep, superResFactor, False)

H, Ht = LFM_computeLFMatrixOperators(Camera, Resolution, LensletCenters)

## Correct the input image
# obtain the transformation between grid models
FixAll = LFM_retrieveTransformation(LensletGridModel, NewLensletGridModel)

# precompute image/volume sizes
trans = format_transform(FixAll)
imgSize = get_transformed_shape(WhiteImage.shape, trans)
imgSize = imgSize + (1-np.remainder(imgSize, 2))

texSize = np.ceil(np.multiply(imgSize, Resolution['texScaleFactor'])).astype('int32')
texSize = texSize + (1-np.remainder(texSize, 2))

ndepths = len(Resolution['depths'])
volumeSize = np.append(texSize, ndepths).astype('int32')

# build anti-aliasing filter kernels
widths = LFM_computeDepthAdaptiveWidth(Camera, Resolution);
kernelFFT = lanczosfft(volumeSize, widths, lanczosWindowSize)

## setup the image
img = cupy.array(LensletImage, dtype='float32')
new = transform_img(img, trans, LensletCenters['offset'])
newnorm = (new - np.min(new)) / (np.max(new) - np.min(new))
LFimage = newnorm

## start deconvolving
crange = Camera['range']
initVolume = np.ones(volumeSize, dtype='float32')

print('precomputing partial forward and back projections...')
onesForward = LFM_forwardProject(H, initVolume, LensletCenters, Resolution, imgSize, crange, step=8)
onesBack = LFM_backwardProject(Ht, onesForward, LensletCenters, Resolution, texSize, crange, step=8)

# deconv algorithm
LFimage = cupy.asarray(LFimage)
reconVolume = cupy.asarray(np.copy(initVolume))

t1 = time()

for i in range(niter):
    if i == 0:
        LFimageGuess = onesForward
    else:
        LFimageGuess = LFM_forwardProject(H, reconVolume, LensletCenters, Resolution, imgSize, crange, step=10)
    if has_cupy:
        mempool.free_all_blocks()

    errorLFimage = LFimage / LFimageGuess * onesForward
    errorLFimage[~cupy.isfinite(errorLFimage)] = 0

    errorBack = LFM_backwardProject(Ht, errorLFimage, LensletCenters, Resolution, texSize, crange, step=10)
    if has_cupy:
        mempool.free_all_blocks()

    errorBack = errorBack / onesBack
    errorBack[~cupy.isfinite(errorBack)] = 0

    # update
    reconVolume = reconVolume * errorBack

    if filterFlag:
        for j in range(errorBack.shape[2]):
            reconVolume[:, :, j] = cupy.abs(fftshift(ifft2(kernelFFT[:,:,j] * fft2(reconVolume[:,:,j]))))

    reconVolume[~cupy.isfinite(reconVolume)] = 0
    if has_cupy:
        mempool.free_all_blocks()

if has_cupy:
    reconVolume_np = cupy.asnumpy(reconVolume)
else:
    reconVolume_np = reconVolume

t2 = time()
print('time', t2 - t1)

import matplotlib.pyplot as plt

m = np.max(vref['reconVolume'])

plt.figure(1)
plt.clf()
plt.imshow(reconVolume_np[:, :, 0])
plt.draw()
plt.show(block=False)
