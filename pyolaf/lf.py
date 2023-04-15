#!/usr/bin/env python3

import numpy as np
import time
from tqdm import trange

def LFM_computePSFsize(maxDepth, Camera):
    ## geometric blur radius at the MLA
    maxDepth = maxDepth - Camera["offsetFobj"]
    zobj = Camera["fobj"] - maxDepth
    # avoid division by zero
    if(zobj == Camera["fobj"] or zobj == Camera["dof"]):
        zobj = zobj + 0.00001*Camera["fobj"]
    # z1 -> where the objective will focus
    z1 = (zobj * Camera["fobj"]) / (zobj - Camera["fobj"])
    # effective radius of the tube lens
    tubeRad = Camera["objRad"] * Camera["Delta_ot"] * np.abs(1.0/z1 - 1/Camera["Delta_ot"])

    # z2 -> where the tl will focus
    z2 = Camera["ftl"] * (Camera["Delta_ot"] - z1) / (Camera["Delta_ot"] - z1 - Camera["ftl"])

    ## main blur (at the mla)
    BlurRad = tubeRad * Camera["tube2mla"] * np.abs(1.0/z2 - 1/Camera["tube2mla"])
    # allow for some extra extent as the psf decays smoothly compared to rays prediction
    PSFsize = np.ceil(BlurRad / Camera["lensPitch"]) + 2
    print(f"Size of PSF radius ~= {PSFsize} [microlens pitch]")
    return PSFsize

def LFM_getUsedCenters(PSFsize, lensletCenters):
    usedLens = np.array([PSFsize + 3, PSFsize + 3])
    centerOfMatrix = np.round(0.01+np.array(lensletCenters["px"].shape[:2])/2).astype('int')

    # calculate the indices of the lenslets within the PSFsize+3 region around the center
    usedLensletIndeces_y = np.arange(centerOfMatrix[0]-usedLens[0]-1, centerOfMatrix[0]+usedLens[0]).astype(int)
    usedLensletIndeces_x = np.arange(centerOfMatrix[1]-usedLens[1]-1, centerOfMatrix[1]+usedLens[1]).astype(int)

    # make sure indices are within the image dimensions
    usedLensletIndeces_y = usedLensletIndeces_y[(usedLensletIndeces_y >= 0) & (usedLensletIndeces_y < lensletCenters["px"].shape[0])]
    usedLensletIndeces_x = usedLensletIndeces_x[(usedLensletIndeces_x >= 0) & (usedLensletIndeces_x < lensletCenters["px"].shape[1])]

    # use the calculated indices to index the lenslet centers for the used region
    usedLensletCenters = {"px": lensletCenters["px"][usedLensletIndeces_y[:, None], usedLensletIndeces_x, :],
                          "vox": lensletCenters["vox"][usedLensletIndeces_y[:, None], usedLensletIndeces_x, :]}

    return usedLensletCenters

def LFM_calcPSFAllDepths(Camera, Resolution):
    # Offsets the depths in the case of a defocused (2.0) LFM setup; offsetFobj is zero for original (1.0) LFM setup
    Resolution["depths"] = Resolution["depths"] + Camera["offsetFobj"]

    psfWaveStack = np.zeros((len(Resolution["yspace"]), len(Resolution["xspace"]), len(Resolution["depths"])), dtype='complex128')
    print('Computing PSF for main objective:')
    print('...')

    for i in range(len(Resolution["depths"])):
        compute_psf = True
        idx = 0

        # Check if the abs(depth) was previously computed, as zero-symmetric depths are just conjugates.
        if i > 0:
            idx = np.where(np.abs(Resolution["depths"][:i]) == np.abs(Resolution["depths"][i]))[0]
            if idx.size != 0:
                compute_psf = False
                idx = idx[0]

        # If depth has not been computed, compute it
        if compute_psf:
            tic = time.time()
            psfWAVE = LFM_calcPSF(0, 0, Resolution["depths"][i], Camera, Resolution)
            print('PSF: {}/{} in {:.2f}s'.format(i + 1, len(Resolution["depths"]), time.time() - tic))
        else:
            # If it is exactly the same depth just copy
            if Resolution["depths"][i] == Resolution["depths"][idx]:
                psfWAVE = psfWaveStack[:, :, idx]
            else:
                # If it is the negative, conjugate
                psfWAVE = np.conjugate(psfWaveStack[:, :, idx])
            print('PSF: {}/{} already computed for depth {}'.format(i + 1, len(Resolution["depths"]), Resolution["depths"][idx]))

        psfWaveStack[:, :, i]  = psfWAVE

    print('...')
    return psfWaveStack

from scipy import integrate, special
import scipy.special


def LFM_calcPSF(p1, p2, p3, Camera, Resolution):
    """
    Computes the PSF at the Camera.Native image plane for a source point (p1, p2, p3) using wave optics theory
    camera['NA'] -> main lens numerical aperture
    yspace, xspace -> sensor space coordinates
    camera['WaveLength'] -> wavelength
    M -> objective magnification
    n -> refractive index
    """
    k = 2*np.pi*Camera['n']/Camera['WaveLength']  # wave number
    alpha = np.arcsin(Camera['NA']/Camera['n'])  # maximal half-angle of the cone of light entering the lens
    demag = 1/Camera['M']

    ylength = len(Resolution['yspace'])
    xlength = len(Resolution['xspace'])
    centerPT = np.ceil(len(Resolution['yspace'])/2).astype(int)
    pattern = np.zeros((centerPT, centerPT))
    zeroline = np.zeros((1, centerPT))

    yspace = Resolution['yspace'][0:centerPT]
    xspace = Resolution['xspace'][0:centerPT]

    d1 = Camera['dof'] - p3  # as per Advanced optics book (3.4.2)

    # compute the PSF for one quarted on the sensor area and replicate (symmetry)
    u = 4*k*(p3*1)*(np.sin(alpha/2)**2)
    Koi = demag/((d1*Camera['WaveLength'])**2)*np.exp(-1j*u/(4*(np.sin(alpha/2)**2)))

    # create meshgrid of x and y values
    x, y = np.meshgrid(xspace, yspace)

    xL2normsq = (((y+Camera['M']*p1)**2+(x+Camera['M']*p2)**2)**0.5)/Camera['M']

    # radial and axial optical coodinates
    v = k*xL2normsq*np.sin(alpha)

    def intgrand(theta, alpha, u, v):
        return np.sqrt(np.cos(theta)) * (1+np.cos(theta))  * \
            np.exp((1j*u/2) * (np.sin(theta/2)**2) / (np.sin(alpha/2)**2)) * \
            scipy.special.j0(np.sin(theta)/np.sin(alpha)*v) *  (np.sin(theta))

    # integrate intgrand() over theta for all x and y values
    # compute PSF
    # integrate over theta: 0->alpha
    I0x, _ = integrate.quad_vec(intgrand, 0, alpha,
                                args=(alpha, u, v), limit=100)

    # compute pattern for all a and b values
    pattern = np.zeros((centerPT, centerPT), dtype='complex128')
    for a in range(centerPT):
        pattern[a,a:] = Koi * I0x[a, a:]


    # setup the whole(sensor size) PSF
    patternA = pattern
    patternAt = np.fliplr(patternA)
    pattern3D = np.zeros((xlength, ylength, 4), dtype='complex128')
    pattern3D[0:centerPT, 0:centerPT, 0] = pattern
    pattern3D[0:centerPT, centerPT-1:, 0] = patternAt
    pattern3D[:, :, 1] = np.rot90(pattern3D[:, :, 0], -1)
    pattern3D[:, :, 2] = np.rot90(pattern3D[:,:,0], -2)
    pattern3D[:, :, 3] = np.rot90(pattern3D[:, :, 0], -3)

    # pattern = max(pattern3D,[],3)
    # for the zero plane as there's no phase the result is real,
    # then max grabs the zeros instead of the negative values
    # This for loops are used instead
    # now only two diagonal lines overlap, so only evaluate the diagonals
    pattern = np.sum(pattern3D, axis=2)
    for i in range(pattern3D.shape[0]):
        # first diagonal
        max_index = np.argmax(np.abs(pattern3D[i, i, :]))
        pattern[i, i] = pattern3D[i, i, max_index]
        # second diagonal
        i_mirror = pattern.shape[1]-i-1
        max_index = np.argmax(np.abs(pattern3D[i, i_mirror, :]))
        pattern[i, i_mirror] = pattern3D[i, i_mirror, max_index]

    psf = pattern
    return psf


def LFM_ulensTransmittance(Camera, Resolution):

    # Compute lens transmittance function for one micro-lens (consistent with Advanced Optics Theory book and Broxton's paper)
    ulensPattern = np.zeros((len(Resolution["yMLspace"]), len(Resolution["xMLspace"])),
                            dtype='complex128')
    # for j in range(len(Camera["fm"])):
    for a in range(len(Resolution["yMLspace"])):
        for b in range(len(Resolution["xMLspace"])):
            x1 = Resolution["yMLspace"][a]
            x2 = Resolution["xMLspace"][b]
            xL2norm = x1**2 + x2**2
            ulensPattern[a, b] = np.exp(-1j*Camera["k"]/(2*Camera["fm"])*xL2norm)

    # Mask the pattern, to avoid overlapping when applying it to the whole image
    # for j in range(len(Camera["fm"])):
    patternML_single = np.copy(ulensPattern)
    if (Resolution["maskFlag"] == 1):
        patternML_single[Resolution["sensMask"] == 0] = 0
    else:
        # circ lens shapes with with blocked light between them
        x, y = np.meshgrid(Resolution["yMLspace"], Resolution["xMLspace"])
        patternML_single[(np.sqrt(x*x+y*y) >= Camera["lensPitch"]/2 - 3)] = 0 # TODO: hardcoded 3
    ulensPattern = patternML_single

    return ulensPattern

from scipy.signal import convolve2d

def LFM_mlaTransmittance(Camera, Resolution, ulensPattern):
    # Compute the ML array as a grid of phase/amplitude masks corresponding to mlens
    ylength = len(Resolution["yspace"])
    xlength = len(Resolution["xspace"])

    # build a slightly bigger array to make sure there are no empty borders (due to the hexagonal packing)
    ylength_extended = len(Resolution["yspace"]) + 2*len(Resolution["yMLspace"])
    xlength_extended = len(Resolution["xspace"]) + 2*len(Resolution["xMLspace"])

    # offset centers
    usedLensletCentersOff = np.zeros((Resolution["usedLensletCenters"]["px"].shape[0],
                                      Resolution["usedLensletCenters"]["px"].shape[1], 2),
                                     dtype='int64')
    usedLensletCentersOff[...,0] = np.round(Resolution["usedLensletCenters"]["px"][...,0]) + np.ceil(ylength_extended / 2)
    usedLensletCentersOff[...,1] = np.round(Resolution["usedLensletCenters"]["px"][...,1]) + np.ceil(xlength_extended / 2)

    # in case of multi focus arrays, lens center also have a type
    # if (Resolution["usedLensletCenters"]["px"].shape[2] == 3):
    #     usedLensletCentersOff[:,:,2] = Resolution["usedLensletCenters"]["px"][:,:,2]

    # activate lenslet centers -> set to 1
    MLspace = np.zeros((ylength_extended, xlength_extended), dtype='complex128')
    MLcenters = np.copy(MLspace)
    # if (Resolution["usedLensletCenters"]["px"].shape[2] == 3): # multifocus, ignoring
    #     MLcenters_types = MLspace

    for a in range(usedLensletCentersOff.shape[0]):
        for b in range(usedLensletCentersOff.shape[1]):
            if (usedLensletCentersOff[a, b, 0] < 1 or usedLensletCentersOff[a, b, 0] > ylength_extended or
                usedLensletCentersOff[a, b, 1] < 1 or usedLensletCentersOff[a, b, 1] > xlength_extended):
              continue

            MLcenters[ usedLensletCentersOff[a, b, 0]-1, usedLensletCentersOff[a, b, 1]-1] = 1 + 0.j

            # if (Resolution["usedLensletCenters"]["px"].shape[2] == 3):
            #     MLcenters_types[ usedLensletCentersOff[a, b, 0] - 1, usedLensletCentersOff[a, b, 1]  - 1] = usedLensletCentersOff[a, b, 2]


    # apply the mlens pattern at every ml center
    MLARRAY_extended = np.copy(MLspace)

    # for j in range(len(Camera["fm"])):
    j = 0 # only one fm
    # if (Resolution["usedLensletCenters"]["px"].shape[2] == 3):
    #     tempSlice = np.zeros_like(MLcenters)
    #     tempSlice[MLcenters_types == j] = 1
    # else:
    tempSlice = MLcenters
    MLARRAY_extended += convolve2d(tempSlice, ulensPattern, mode='same')


    # get back the center part of the array (ylength, xlength)
    inner_rows = slice(int(np.ceil(ylength_extended/2) - np.floor(ylength/2))-1,
                       int(np.ceil(ylength_extended/2) + np.floor(ylength/2)))
    inner_cols = slice(int(np.ceil(xlength_extended/2) - np.floor(xlength/2))-1,
                       int(np.ceil(xlength_extended/2) + np.floor(xlength/2)))

    MLARRAY = MLARRAY_extended[inner_rows, inner_cols]

    return MLARRAY

from scipy.ndimage import shift

def imShift2(Image, ShiftX, ShiftY):
    if len(Image.shape) == 2:
        Img = Image[:, :, None]
    else:
        Img = Image

    newImg = np.zeros_like(Img)
    for i in range(Img.shape[-1]):  # loop over color channels
        newImg[..., i] = shift(Img[..., i], (ShiftX, ShiftY), order=1)

    # eqtol = 1e-10

    # xlength, ylength = Img.shape[:2]

    # if abs(np.mod(ShiftX,1)) > eqtol or abs(np.mod(ShiftY,1)) > eqtol:
    #     raise ValueError('SHIFTX and SHIFTY must be integers')

    # ShiftX = round(ShiftX)
    # ShiftY = round(ShiftY)


    # newImg = np.zeros((xlength, ylength, Img.shape[2]), dtype=Img.dtype)

    # if ShiftX >= 0 and ShiftY >= 0:
    #     newImg[ShiftX:, ShiftY:, :] = Img[:-ShiftX, :-ShiftY, :]
    # elif ShiftX >= 0 and ShiftY < 0:
    #     newImg[ShiftX:, :ShiftY, :] = Img[:-ShiftX, -ShiftY:, :]
    # elif ShiftX < 0 and ShiftY >= 0:
    #     newImg[:ShiftX, ShiftY:, :] = Img[-ShiftX:, :-ShiftY, :]
    # else:
    #     newImg[:ShiftX, :ShiftY, :] = Img[-ShiftX:, -ShiftY:, :]

    if len(Image.shape) == 2:
        newImg = newImg[:, :, 0]

    return newImg

from scipy.fft import fft2, ifft2
from scipy.ndimage import zoom

def prop2Sensor(f0, sensorRes, z, wavelength, idealSampling):
    """Computes the final Lightfield PSF"""
    if idealSampling:
        if z == 0:
            f1 = f0
        else:
            # Ideal sampling rate (compute the impulse response h) -> computational Fourier Optics book
            Lx, Ly = f0.shape[0]*sensorRes[0], f0.shape[1]*sensorRes[1]
            k = 2*np.pi/wavelength

            ideal_rate = [wavelength*np.sqrt(z**2 + (Lx/2)**2)/Lx, wavelength*np.sqrt(z**2 + (Ly/2)**2)/Ly]
            ideal_rate = np.array(ideal_rate)
            ideal_samples_no = np.ceil(np.array([Lx, Ly])/ideal_rate).astype(int)
            ideal_samples_no = ideal_samples_no + (1 - ideal_samples_no % 2)
            rate = np.array([Lx, Ly])/ideal_samples_no

            # spacial frequencies in x and y direction
            du = 1./(ideal_samples_no[0]*np.single(rate[0]))
            dv = 1./(ideal_samples_no[1]*np.single(rate[1]))
            u = np.hstack((np.arange(np.ceil(ideal_samples_no[0]/2)),
                           np.arange(-np.floor(ideal_samples_no[0]/2), 0))) * du
            v = np.hstack((np.arange(np.ceil(ideal_samples_no[1]/2)),
                           np.arange(-np.floor(ideal_samples_no[1]/2), 0))) * dv

            # transfer function for Rayleigh-Sommerfeld diffraction integral
            H = np.exp(1j*np.sqrt(1-wavelength**2*(np.tile(u, (len(v), 1)).T**2+np.tile(v, (len(u), 1))**2))*z*k)

            f1 = ifft2(fft2(zoom(f0, ideal_samples_no, order=3), norm='ortho').T*H, norm='ortho').T
            f1 = zoom(f1, f0.shape, order=3)
    else:
        # Original Sampling: compute the Transfer Function H
        Nx, Ny = f0.shape[0], f0.shape[1]
        k = 2*np.pi/wavelength

        # spacial frequencies in x and y direction
        du = 1./(Nx*np.single(sensorRes[0]))  # changed to `np.single` since we are using `ceil`
        dv = 1./(Ny*np.single(sensorRes[1]))
        u = np.hstack((np.arange(np.ceil(Nx/2)),
                       np.arange(-np.floor(Nx/2), 0))) * du
        v = np.hstack((np.arange(np.ceil(Ny/2)),
                       np.arange(-np.floor(Ny/2), 0))) * dv

        # transfer function for Rayleigh diffraction integral
        H = np.exp(1j*np.sqrt(1-wavelength**2 * (np.tile(u, (len(v), 1)).T**2 + np.tile(v, (len(u), 1))**2))*z*k)

        # final Lightfield PSF -> sensor image
        f1 = np.exp(1j*k*z)*(ifft2(fft2(f0, norm='ortho')*H, norm='ortho'))

    return f1

def LFM_computeForwardPatternsWaves(psfWaveStack, MLARRAY, Camera, Resolution):
    # ComputeForwardPatternsWaves: Compute the forward projection for every source point (aa,bb,c) of our square shaped patch around the central microlens
    # Take the PSF incident on the MLA (psfWAVE_STACK), pass it through the
    # microlens array (MLARRAY), and finally propagate it to the sensor

    # for regular grids compute the psf for only one quarter of coordinates (due to symmetry)
    if Camera['range'] == 'quarter':
        coordsRange  = Resolution['TexNnum_half']
    else:
        coordsRange  = Resolution['TexNnum']

    Nnum_half_coord = Resolution['TexNnum_half'] // Resolution['texScaleFactor']
    sensorRes = Resolution['sensorRes']

    # Resolution['texScaleFactor(1/2)'] is actually (Resolution['texRes(1/2)'] * M / Resolution['sensorRes(1/2)'])^-1
    H = np.empty((coordsRange[0], coordsRange[1], len(Resolution['depths'])),
                 dtype='object')
    for c in trange(len(Resolution['depths']), ncols=70, desc=' forward patterns'):
        # print('Forward Patterns, depth:', c+1, '/', len(Resolution['depths']))
        psfREF = psfWaveStack[:,:,c]
        for i in range(coordsRange[0]):
            for j in range(coordsRange[1]):
                aa_tex = i+1
                aa_sensor = aa_tex / Resolution['texScaleFactor'][0]
                bb_tex = j+1
                bb_sensor = bb_tex / Resolution['texScaleFactor'][1]

                # shift the native plane PSF at every (aa, bb) position (native plane PSF is shift invariant)
                psfSHIFT = imShift2(psfREF, (aa_sensor-Nnum_half_coord[0]).round(), (bb_sensor-Nnum_half_coord[1]).round())

                # MLA transmittance
                psfMLA = psfSHIFT*MLARRAY

                # propagate the response to the sensor via Rayleigh-Sommerfeld diffraction
                LFpsfAtSensor = prop2Sensor(psfMLA, sensorRes, Camera['mla2sensor'], Camera['WaveLength'], False)

                # shift the response back to center (we need the patterns centered for convolution)
                LFpsf = imShift2(LFpsfAtSensor, (-(aa_sensor-Nnum_half_coord[0])).round(), (-(bb_sensor-Nnum_half_coord[1])).round())

                # store the response pattern
                H[(i,j,c)] = csr_matrix(np.abs(LFpsf**2))

    return H

from scipy.sparse import csr_matrix

def LFM_computeBackwardPatterns(H, Resolution, crange, lensOrder):
    # ComputeBackwardPatterns: Computes which light points in the
    # object affect every pixel behind a micro-lens.

    # print(crange, lensOrder)

    # H_sizes = np.max(list(H.keys()), axis=0) + 1
    H_sizes = H.shape

    # retrieve sensor image and 3D scene containers sizes
    # if H_sizes[2] == 3: # multifocus case, not used in this code
    #     nDepths = np.shape(H[0][0])[2]  # depths
    #     imgSize = np.shape(H[0][0][Resolution["TexNnum_half"][0], Resolution["TexNnum_half"][1]])  # forward projection size
    # else:
    nDepths = H_sizes[2]  # depths
    imgSize = np.array(H[0, 0, 0].shape)  # forward projection size

    # Compute volume size; it will be different to the image size in case of a
    # superRFactor different than the number of pixels behind a lenslet.
    texSize = np.ceil(imgSize*np.array(Resolution["texScaleFactor"]))
    texSize = texSize + (1 - texSize % 2)
    texSize = texSize.astype(int)

    # offset the lenslet centers to match the image/volume centers
    offsetImg = np.ceil(imgSize/2).astype(int)
    offsetVol = np.ceil(texSize/2).astype(int)
    lensletCenters = {"px": np.zeros_like(Resolution["usedLensletCenters"]["px"]),
                      "vox": np.zeros_like(Resolution["usedLensletCenters"]["vox"])}
    lensletCenters["px"][:,:,0] = Resolution["usedLensletCenters"]["px"][:,:,0] + offsetImg[0]
    lensletCenters["px"][:,:,1] = Resolution["usedLensletCenters"]["px"][:,:,1] + offsetImg[1]
    # if (isinstance(H, list) and len(H[0][0]) == 3): # only for multifocus case
    #     lensletCenters["px"][:,:,2] = Resolution["usedLensletCenters"]["px"][:,:,2]
    lensletCenters["vox"][:,:,0] = Resolution["usedLensletCenters"]["vox"][:,:,0] + offsetVol[0]
    lensletCenters["vox"][:,:,1] = Resolution["usedLensletCenters"]["vox"][:,:,1] + offsetVol[1]

    # for regular grids compute the backprojection patterns only for one quarter of coordinates (due to symmetry)
    if crange == 'quarter':
        coordsRange = Resolution["Nnum_half"]
    else:
        coordsRange = Resolution["Nnum"]

    # compute backprojection patterns for all the pixels in coordsRange
    Ht = np.empty((coordsRange[0], coordsRange[1], nDepths), dtype='object')  # container for back projection patterns

    # Iterate through every pixel behind the central micro-lens and compute
    # which part of the texture (object) affects it.

    for aa_sensor in trange(coordsRange[0], ncols=70, desc='backward patterns'):
        # print('Backward patterns, x: {}/{}'.format(aa_sensor+1, coordsRange[0]))
        aa_tex = np.ceil((1+aa_sensor) * Resolution["texScaleFactor"][0]).astype(int)  # compute the corresponding coord of "aa_sensor" pixel in real world space
        for bb_sensor in range(coordsRange[1]):
            bb_tex = np.ceil((1+bb_sensor) * Resolution["texScaleFactor"][1]).astype(int)  # compute the corresponding coord of "bb_sensor" pixel in real world space

            # backproject the activated sensor pixel
            currentPixel = [aa_sensor + offsetImg[0] - Resolution["Nnum_half"][0],
                            bb_sensor + offsetImg[1] - Resolution["Nnum_half"][1]]  # position of the current active pixel in the image
            tempback = LFM_backwardProjectSinglePoint(H, Resolution, imgSize, texSize, currentPixel, lensletCenters, crange, lensOrder)

            # bring the backprojection to center (the operator assumes the bproj patterns are centered when applying them)
            for cc in range(nDepths):
                backShiftX = np.round(Resolution["TexNnum_half"][0]-aa_tex).astype(int)
                backShiftY = np.round(Resolution["TexNnum_half"][1]-bb_tex).astype(int)
                shifted = imShift2(tempback[:,:,cc], backShiftX, backShiftY)
                # store the pattern
                Ht[(aa_sensor,bb_sensor, cc)] = csr_matrix(shifted)



    return Ht




import numpy as np
from scipy.sparse import coo_matrix

def sconv2singlePointFlip(sizeA, point, B, flipBX, flipBY, shape):
    """
    Adapted from C = sconv2(A, B, shape)
    Author: Bruno Luong <brunoluong@yahoo.com>
    """
    m, n = sizeA

    p, q = B.shape

    i, j = point
    a = 1

    k, l = np.nonzero(B)
    b = B[k, l]

    if flipBX != 0:
        k = p - k - 1
    if flipBY != 0:
        l = q - l - 1

    k += 1
    l += 1

    I, K = np.meshgrid(i, k)
    J, L = np.meshgrid(j, l)
    C = np.outer(a, b)

    shape = shape.lower()
    if shape == 'full':
        C = coo_matrix((C.ravel(), (I.ravel()+K.ravel()-1, J.ravel()+L.ravel()-1)), shape=(m+p-1, n+q-1))
    elif shape == 'valid':
        mnc = np.maximum([m - np.maximum(0, p-1), n - np.maximum(0, q-1)], 0)
        i = I.ravel()+K.ravel()-p
        j = J.ravel()+L.ravel()-q
        b = np.logical_and(np.logical_and(i >= 0, i < mnc[0]), np.logical_and(j >= 0, j < mnc[1]))
        C = coo_matrix((C.ravel()[b], (i[b], j[b])), shape=(mnc[0], mnc[1]))
    elif shape == 'same':
        i = I.ravel()+K.ravel()-np.ceil((p+1)/2).astype(int)
        j = J.ravel()+L.ravel()-np.ceil((q+1)/2).astype(int)
        b = np.logical_and(np.logical_and(i >= 0, i < m), np.logical_and(j >= 0, j < n))
        C = coo_matrix((C.ravel()[b], (i[b], j[b])), shape=(m, n))

    return C.toarray()

# from skimage.transform import rotate

def rotate180(image):
    return np.flip(np.flip(image, axis=0), axis=1)

def LFM_backwardProjectSinglePoint(H, Resolution, imgSize, texSize, currentPixel, lensletCenters, crange, lensOrder):
    # backwardProjectSinglePoint: This function computes the object response to a single senxsor pixel
    # (which voxels from the object affect a single pixel in the sensor).
    # It return a stack of images equal to the number of depths.
    # if ismatrix(H) and H.shape[1] == 3: # for multi focus microscope
    #     multiFocus = 3
    #     nDepths = H[0,0,0].shape[2]
    #     ulensType = lensletCenters["px"][:,:,3]
    # else:
    # print(H, Resolution, imgSize, texSize, currentPixel, lensletCenters, range, lensOrder)

    # import IPython; IPython.embed()

    # H_sizes = np.max(list(H.keys()), axis=0) + 1
    H_sizes = H.shape
    multiFocus = 1
    nDepths = H_sizes[2]

    Backprojection = np.zeros((texSize[0], texSize[1], nDepths))
    lensVoxY = lensletCenters["vox"][:,:,0]
    lensVoxX = lensletCenters["vox"][:,:,1]
    lensSenY = lensletCenters["px"][:,:,0]
    lensSenX = lensletCenters["px"][:,:,1]

    # Iterate different types of micro-lenses
    for i in range(1, multiFocus+1):
        if multiFocus == 1:
            currentLensVoxY = lensVoxY
            currentLensVoxX = lensVoxX
            currentLensSenY = lensSenY
            currentLensSenX = lensSenX
        # else:
        #     currentLensVoxY = lensVoxY[ulensType == i]
        #     currentLensVoxX = lensVoxX[ulensType == i]
        #     currentLensSenY = lensSenY[ulensType == i]
        #     currentLensSenX = lensSenX[ulensType == i]

        # Iterate all depths
        for cc in range(nDepths):
            sliceCurrentDepth = np.zeros(texSize)
            for aa in range(Resolution["TexNnum"][0]):
                for bb in range(Resolution["TexNnum"][1]):
                    # Avoid overlaps for non-regular grids
                    if Resolution["texMask"][aa,bb] == 0:
                        continue

                    aa_new = aa
                    flipX = 0
                    if aa > Resolution["TexNnum_half"][0]-1 and crange == "quarter":
                        aa_new = Resolution["TexNnum"][0] - aa - 1
                        flipX = 1

                    bb_new = bb
                    flipY = 0
                    if bb > Resolution["TexNnum_half"][1]-1 and crange == "quarter":
                        bb_new = Resolution["TexNnum"][1] - bb - 1
                        flipY = 1

                    if multiFocus == 1:
                        # Ht = rotate(H[aa_new,bb_new,cc].todense(), 180, preserve_range=True)
                        Ht = rotate180(H[aa_new,bb_new,cc].toarray())
                    # else: # multifocus, removed
                    #     Ht = rotate(H[0,lensOrder[i-1]][aa_new,bb_new,cc], 180)
                    tempSlice = sconv2singlePointFlip(imgSize, currentPixel, Ht, flipX, flipY, "same")

                    # Grab relevant voxels
                    lensYInsideTex = np.round(currentLensVoxY - Resolution["TexNnum_half"][0] + aa).astype(int)
                    lensXInsideTex = np.round(currentLensVoxX - Resolution["TexNnum_half"][1] + bb).astype(int)

                    validLensTex = (lensXInsideTex < texSize[1]) & (lensXInsideTex >= 0) & \
                        (lensYInsideTex < texSize[0]) & (lensYInsideTex >= 0)

                    # Method 2 without resizing, 2x faster
                    lensYInside = -1 + currentLensSenY + np.round((-Resolution["TexNnum_half"][0] + aa + 1) / Resolution["texScaleFactor"][0])
                    lensXInside = -1 + currentLensSenX + np.round((-Resolution["TexNnum_half"][1] + bb + 1) / Resolution["texScaleFactor"][1])

                    lensYInside = lensYInside.astype(int)
                    lensXInside = lensXInside.astype(int)

                    validLens = (lensXInside < imgSize[1]) & (lensXInside >= 0) & \
                        (lensYInside < imgSize[0]) & (lensYInside >= 0)

                    validLens = validLens & validLensTex
                    validLensTex = validLensTex & validLens


                    # indicesSen = np.ravel_multi_index((lensYInside[validLens]-1, lensXInside[validLens]-1), imgSize)
                    # indicesTex = np.ravel_multi_index((lensYInsideTex[validLensTex]-1, lensXInsideTex[validLensTex]-1), texSize)
                    indicesSen = tuple(np.vstack([lensYInside[validLens], lensXInside[validLens]]))
                    indicesTex = tuple(np.vstack([lensYInsideTex[validLens], lensXInsideTex[validLens]]))

                    if len(indicesSen) != len(indicesTex):
                        print("Error in backProjectSinglePoint: mismatching indices!")

                    if np.sum(validLens) > 0:
                        sliceCurrentDepth[indicesTex] = sliceCurrentDepth[indicesTex] + tempSlice[indicesSen]

            Backprojection[:,:,cc] += sliceCurrentDepth
    return Backprojection




def normalizeHt(Ht):
    # Ht_sizes = np.max(list(Ht.keys()), axis=0) + 1
    Ht_sizes = Ht.shape

    for aa in range(Ht_sizes[0]):
        for bb in range(Ht_sizes[1]):
            sum_temp = 0
            for c in range(Ht_sizes[2]):
                sum_temp += Ht[aa, bb, c].sum()
            # temp = np.concatenate(
            #     [Ht[aa, bb, cc].toarray() for cc in range(Ht_sizes[2])],
            #     axis=1)
            # sum_temp = np.sum(temp)
            if np.isclose(sum_temp, 0):
                continue
            for c in range(Ht_sizes[2]):
                Ht[aa, bb, c] = Ht[aa, bb, c] / sum_temp

    return Ht

def ignoreSmallVals(H, tol):
    for a in range(H.shape[0]):
        for b in range(H.shape[1]):
            for c in range(H.shape[2]):
                k = (a, b, c)
                temp = H[k]
                max_slice = np.max(temp)
                # Clamp values smaller than tol
                temp[temp < max_slice*tol] = 0
                sum_temp = np.sum(temp)
                if sum_temp == 0:
                    continue
                # and normalize per individual PSF such that the sum is 1.
                temp = temp / sum_temp
                H[k] = csr_matrix(temp)
    return H

def LFM_computePatternsSingleWaves(psfWaveStack, Camera, Resolution, tolLFpsf):
    ## Precompute the MLA transmittance funtion
    ulensPattern = LFM_ulensTransmittance(Camera, Resolution)
    MLARRAY = LFM_mlaTransmittance(Camera, Resolution, ulensPattern)

    ## Compute forward light trasport patterns -> patters for all texture points
    print('Computing forward patterns for single focus array')
    H = LFM_computeForwardPatternsWaves(psfWaveStack, MLARRAY, Camera, Resolution)
    H = ignoreSmallVals(H, tolLFpsf)

    ## Compute backward light transport patterns -> patters for all sensor points
    print('Computing backward patterns for single focus array')
    Ht = LFM_computeBackwardPatterns(H, Resolution, Camera["range"], [])
    Ht = normalizeHt(Ht)
    return H, Ht

def LFM_computePatternsMultiWaves(psfWaveStack, Camera, Resolution, tolLFpsf):
    """
    Compute patterns for every position in the ulens range. Hex grid discretization is not symmetric about the
    center of the lens.
    """
    print('Computing forward patterns for multifocal array')

    tic = time.time()

    # Pattern 1
    ulensPattern = LFM_ulensTransmittance(Camera, Resolution)
    MLARRAY = LFM_mlaTransmittance(Camera, Resolution, ulensPattern)
    H1 = LFM_computeForwardPatternsWaves(psfWaveStack, MLARRAY, Camera, Resolution)

    # Pattern 2
    CameraShift = Camera.copy()
    CameraShift['fm'] = np.roll(Camera['fm'], -1)
    ulensPattern = LFM_ulensTransmittance(CameraShift, Resolution)
    MLARRAY = LFM_mlaTransmittance(CameraShift, Resolution, ulensPattern)
    H2 = LFM_computeForwardPatternsWaves(psfWaveStack, MLARRAY, CameraShift, Resolution)

    # Pattern 3
    CameraShift['fm'] = np.roll(Camera['fm'], -2)
    ulensPattern = LFM_ulensTransmittance(CameraShift, Resolution)
    MLARRAY = LFM_mlaTransmittance(CameraShift, Resolution, ulensPattern)
    H3 = LFM_computeForwardPatternsWaves(psfWaveStack, MLARRAY, CameraShift, Resolution)

    toc = time.time()
    print(f"Execution time: {toc - tic:.3f} seconds")

    # Ignore small values
    H = {}
    H1 = ignoreSmallVals(H1, tolLFpsf)
    H2 = ignoreSmallVals(H2, tolLFpsf)
    H3 = ignoreSmallVals(H3, tolLFpsf)
    H[1] = H1
    H[2] = H2
    H[3] = H3

    # Compute backward light transport patterns -> patterns for all texture points
    print('Computing backward patterns for multifocal array')

    tic = time.time()

    lensOrder = [1, 2, 3]
    Ht1 = LFM_computeBackwardPatterns(H, Resolution, Camera['range'], lensOrder)
    Ht2 = LFM_computeBackwardPatterns(H, Resolution, Camera['range'], np.roll(lensOrder, -1))
    Ht3 = LFM_computeBackwardPatterns(H, Resolution, Camera['range'], np.roll(lensOrder, -2))

    Ht = {}
    Ht1 = normalizeHt(Ht1)
    Ht2 = normalizeHt(Ht2)
    Ht3 = normalizeHt(Ht3)
    Ht[1] = Ht1
    Ht[2] = Ht2
    Ht[3] = Ht3

    toc = time.time()
    print(f"Execution time: {toc - tic:.3f} seconds")

    return H, Ht

def LFM_computeLFMatrixOperators(Camera, Resolution, LensletCenters):
    # Maximum spread area on the sensor and corresponding lens centers
    # psf area is maximal at max depth
    # compute psf size in lenslets, and image size in px
    depth_range = np.array(Resolution['depthRange']) + Camera['offsetFobj']
    p = np.argmax(np.abs(depth_range))
    maxDepth = depth_range[p]  # the PSF area is highest at furthest depth
    PSFsize = LFM_computePSFsize(maxDepth, Camera)

    # extract only those lenslets touched by the psf extent
    usedLensletCenters = LFM_getUsedCenters(PSFsize, LensletCenters)
    Resolution['usedLensletCenters'] = usedLensletCenters

    ## Object and ML space

    IMGsizeHalf = max( Resolution['Nnum'][1]*(PSFsize), 2*Resolution['Nnum'][1]) # PSF size in pixels
    IMGsizeHalf = int(IMGsizeHalf)
    print('Size of PSF IMAGE = {}X{} [Pixels]'.format(IMGsizeHalf*2+1, IMGsizeHalf*2+1))

    # yspace/xspace represent sensor coordinates
    # yMLspace/xMLspace represent local ulens coordinates
    Resolution['yspace'] = Resolution['sensorRes'][0]*np.arange(-IMGsizeHalf,IMGsizeHalf+1)
    Resolution['xspace'] = Resolution['sensorRes'][1]*np.arange(-IMGsizeHalf,IMGsizeHalf+1)
    Resolution['yMLspace'] = Resolution['sensorRes'][0]* np.arange(- Resolution['Nnum_half'][0]+1, Resolution['Nnum_half'][0])
    Resolution['xMLspace'] = Resolution['sensorRes'][1]* np.arange(- Resolution['Nnum_half'][1]+1, Resolution['Nnum_half'][1])

    ## Compute Patterns

    # compute native plane PSF for every depth
    psfWaveStack = LFM_calcPSFAllDepths(Camera, Resolution)

    tolLFpsf = 0.005 # clap small values in forward patterns to speed up computations
    if Camera['focus'] == 'single':
        H, Ht = LFM_computePatternsSingleWaves(psfWaveStack, Camera, Resolution, tolLFpsf)
    elif Camera['focus'] == 'multi':
        H, Ht = LFM_computePatternsMultiWaves(psfWaveStack, Camera, Resolution, tolLFpsf)

    return H, Ht
