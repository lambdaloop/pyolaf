#!/usr/bin/env python3

import numpy as np
try:
    import cupy
except:
    cupy = np

def lanczosfft(filterSize, widths, n):
    n_depths = filterSize[2]
    size2 = np.int64(np.floor(filterSize / 2))

    x, y = cupy.meshgrid(cupy.arange(-size2[1], size2[1]+1),
                         cupy.arange(-size2[0], size2[0]+1))

    lanczos2FFT = cupy.zeros(filterSize, dtype='complex128')

    dxy = cupy.sqrt(cupy.square(x) + cupy.square(y))

    for i in range(n_depths):
        fy = 1.0/widths[i, 0]
        fx = 1.0/widths[i, 1]

        x_f = x*fx
        y_f = y*fy
        kernelSinc = cupy.sinc(x_f) * cupy.sinc(y_f)

        # nX Lanczos window -- depending on the texture features
        current_window = cupy.sinc(x_f/n) * cupy.sinc(y_f/n)
        check = dxy > (3 * widths[i, 1])
        current_window[check] = 0

        # final windowed ideal filter
        lanczos2 = kernelSinc * current_window
        lanczos2 = lanczos2 / cupy.sum(lanczos2)
        lanczos2FFT[:, :, i] = cupy.fft.fft2(lanczos2)
    return lanczos2FFT

def LFM_computeDepthAdaptiveWidth(Camera, Resolution):

    ## compute the depth dependent witdth of the anti-aliasing filters
    dz = Resolution["depths"]
    zobj = (Camera["fobj"] - dz).astype('float')

    # avoid division by zero
    for i in range(len(zobj)):
        if(np.isclose(zobj[i], Camera["fobj"]) or np.isclose(zobj[i], Camera["dof"])):
            zobj[i] = zobj[i] + 0.00001*Camera["fobj"]

    # z1 -> where the objective will focus
    z1 = (zobj * Camera["fobj"])/(zobj - Camera["fobj"])

    # effective radius of the tube lens
    tubeRad = Camera["objRad"] * Camera["Delta_ot"]*np.abs(1./z1 - 1/Camera["Delta_ot"])

    # z2 -> where the tl will focus
    z2 = Camera["ftl"]*(Camera["Delta_ot"] - z1)/(Camera["Delta_ot"] - z1 - Camera["ftl"])

    # main blur (at the mla)
    B = tubeRad * Camera["tube2mla"] * np.abs(1./z2 - 1/Camera["tube2mla"])

    # z3 -> where the mla focuses
    z3 = Camera["fm"]*(Camera["tube2mla"] - z2)/(Camera["tube2mla"] - z2 - Camera["fm"])

    # miclolens blur radius
    b = Camera["lensPitch"]/2 * np.abs(1./z3 - 1/Camera["mla2sensor"])
    # b = Camera["lensPitch"]/2  * np.abs((z3 - Camera["mla2sensor"])/z3)

    # microlens array to sensor magnification
    lambda_ = z2*Camera["mla2sensor"]/(Camera["tube2mla"] * np.abs(Camera["tube2mla"] - z2))

    # cut-off freq
    d = Camera["lensPitch"]
    f0 = 1/(2*d)

    pinhole_filt_rad = d*(np.abs(lambda_)) #1./(2*f0*abs(lambda_));
    final_rad = np.abs(pinhole_filt_rad - b)

    # Size of filters
    widths = np.minimum(d/2,final_rad)

    ## filter size in object space (voxels)
    widthsX = widths * Resolution["TexNnum"][1]/d
    widthsY = widths * Resolution["TexNnum"][0]/d

    widths = np.zeros((len(widths), 2), dtype='int64')
    widths[:,0] = np.floor(widthsY*2)
    widths[:,1] = np.floor(widthsX*2)
    widths[widths%2 == 0] = widths[widths%2 == 0] + 1

    return widths
