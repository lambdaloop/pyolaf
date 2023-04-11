#!/usr/bin/env ipython

from scipy.io import loadmat
import numpy as np
from scipy import signal
from tqdm import tqdm, trange
from time import time
from .fftpack import cufftconv, cufftconvsum

try:
    import cupy
    has_cupy = True
except ImportError:
    cupy = np
    has_cupy = False

## Forward projection

def get_indices_forward(lensCenters, Resolution, imgSize, texSize):
    texnum = Resolution['TexNnum'][()]
    texnum_half = Resolution['TexNnum_half'][()]

    offsetImg = np.ceil(imgSize / 2)
    offsetVol = np.ceil(texSize / 2)
    lensYvox = lensCenters['vox'][()][:, :, 0] + offsetVol[0]
    lensXvox = lensCenters['vox'][()][:, :, 1] + offsetVol[1]

    indicesTex = dict()
    indicesImg = dict()
    # indicesTex = np.zeros((texnum[0], texnum[1], lensXvox.size), dtype='int32')
    # indicesImg = np.zeros((texnum[0], texnum[1], lensXvox.size), dtype='int32')
    # masks = dict()

    for aa_tex in range(texnum[0]):
        for bb_tex in range(texnum[1]):
            lensXvoxCurrent = np.round(lensYvox - texnum_half[0] + aa_tex + 1)
            lensYvoxCurrent = np.round(lensXvox - texnum_half[1] + bb_tex + 1)

            texScaleFactor = Resolution['texScaleFactor'][()]
            # convert lenses to img space
            lensXpxCurrent = lensXvoxCurrent - offsetVol[0]
            lensXpxCurrent = lensXpxCurrent/texScaleFactor[0]
            lensXpxCurrent = np.ceil(lensXpxCurrent + offsetImg[0]) - 1

            lensYpxCurrent = lensYvoxCurrent - offsetVol[1]
            lensYpxCurrent = lensYpxCurrent/texScaleFactor[1]
            lensYpxCurrent = np.ceil(lensYpxCurrent + offsetImg[1]) - 1

            # check for out of image and texture
            validLens = (lensXvoxCurrent < texSize[0]) & (lensXvoxCurrent >= 0) & \
                (lensYvoxCurrent < texSize[1]) & (lensYvoxCurrent >= 0) & \
                (lensXpxCurrent < imgSize[0]) & (lensXpxCurrent >= 0) & \
                (lensYpxCurrent < imgSize[1]) & (lensYpxCurrent >= 0)

            lensXvoxCurrent = lensXvoxCurrent[validLens]
            lensYvoxCurrent = lensYvoxCurrent[validLens]
            indices = np.vstack([lensXvoxCurrent, lensYvoxCurrent]).astype('int32')
            indicesTex[(aa_tex, bb_tex)] = tuple(indices)
            # indicesTex[aa_tex, bb_tex] = np.ravel_multi_index(indices, texSize)

            lensXpxCurrent = lensXpxCurrent[validLens]
            lensYpxCurrent = lensYpxCurrent[validLens]
            indices = np.vstack([lensXpxCurrent, lensYpxCurrent]).astype('int32')
            # indicesImg[aa_tex, bb_tex] = np.ravel_multi_index(indices, imgSize)
            indicesImg[(aa_tex, bb_tex)] = tuple(indices)

            # m = cupy.zeros(imgSize)
            # m[tuple(indices)] = 1
            # masks[(aa_tex, bb_tex)] = m

    return indicesTex, indicesImg

def get_filters_forward(H, Resolution, crange):
    texnum = Resolution['TexNnum'][()]
    texnum_half = Resolution['TexNnum_half'][()]
    texMask = Resolution['texMask'][()]
    if not isinstance(texMask, np.ndarray):
        texMask = np.ones(texnum, dtype='int32')

    nDepths = H.shape[2]

    fshape = H[0,0,0].shape
    shape_filters = (texnum[0], texnum[1], nDepths, fshape[0], fshape[1])

    filters = np.zeros(shape_filters)

    # cc = 0
    for cc in range(nDepths):
        for aa_tex in range(texnum[0]):
            aa_new = aa_tex;
            flipX = False
            if crange == 'quarter' and aa_tex >= texnum_half[0]:
                aa_new = texnum[0] - aa_tex - 1
                flipX = True

            # bb_tex = 0
            for bb_tex in range(texnum[1]):
                if texMask[aa_tex, bb_tex] == 0:
                    continue

                bb_new = bb_tex
                flipY = False
                if crange == 'quarter' and bb_tex >= texnum_half[1]:
                    bb_new = texnum[1] - bb_tex - 1
                    flipY = True

                # % Fetch corresponding PSF for given coordinate behind the
                # % lenslet
                Hs = H[aa_new, bb_new, cc].todense()
                if flipX:
                    Hs = np.flipud(Hs)

                if flipY:
                    Hs = np.fliplr(Hs)
                filters[aa_tex, bb_tex, cc] = Hs

    return filters


def LFM_forwardProject(H, realSpace, lensCenters, Resolution, imgSize, crange, step=15):
    tt1 = time()
    texnum = Resolution['TexNnum'][()]
    texnum_half = Resolution['TexNnum_half'][()]
    texMask = Resolution['texMask'][()]
    if not isinstance(texMask, np.ndarray):
        texMask = np.ones(texnum, dtype='int32')

    nDepths = H.shape[2]
    texSize = np.array(realSpace.shape[:2])

    indicesTex, indicesImg = get_indices_forward(lensCenters, Resolution, imgSize, texSize)

    t2 = time()
    filters = get_filters_forward(H, Resolution, crange).astype('float32')
    t3 = time()
    # print('filters: {:.3f}'.format(t3 - t2))

    t2 = time()
    Projection = cupy.zeros(imgSize, dtype='float32')

    # shape_slices = (texnum[0], texnum[1], nDepths, imgSize[0], imgSize[1])
    # slices = np.zeros(shape_slices)

    # filters_cupy = cupy.asarray(filters)
    realSpace_cupy = cupy.asarray(realSpace, dtype='float32')

    filterSize = filters.shape[-2:]

    slices_buffer = cupy.empty((step, imgSize[0], imgSize[1]), dtype='float32')
    filter_buffer = cupy.empty((step, filterSize[0], filterSize[1]), dtype='float32')

    tempspace = cupy.empty(imgSize, dtype='float32')

    t3 = time()
    # print('creation: {:.3f}'.format(t3-t2))

    tconv = 0
    bnum = 0

    # cc = 0
    for cc in trange(nDepths, ncols=70, desc='forward '):
        realspaceCurrentDepth = realSpace_cupy[:, :, cc]

        # aa_tex = 0
        for aa_tex in range(texnum[0]):
            # bb_tex = 0
            for bb_tex in range(texnum[1]):
                if texMask[aa_tex, bb_tex] == 0:
                    continue

                ix_tex = indicesTex[(aa_tex, bb_tex)]
                ix_img = indicesImg[(aa_tex, bb_tex)]
                # subset = realspaceCurrentDepth[ix_tex]
                # if cupy.sum(subset) < 0: continue
                # if cupy.sum(realspaceCurrentDepth[ix_tex]) < 0: continue

                tempspace.fill(0)
                tempspace[ix_img] = realspaceCurrentDepth[ix_tex]

                # Hs = filters_cupy[aa_tex, bb_tex, cc]
                # Projection += cufftconv(tempspace[None], Hs[None], mode='same')[0]
                # Projection += cusignal.fftconvolve(tempspace, Hs, mode='same')

                slices_buffer[bnum] = tempspace
                filter_buffer[bnum] = cupy.asarray(filters[aa_tex, bb_tex, cc])
                bnum += 1


                if bnum >= step:
                    Projection += cufftconvsum(slices_buffer, filter_buffer, mode='same')
                    bnum = 0


    if bnum > 0:
        Projection += cufftconvsum(slices_buffer[:bnum], filter_buffer[:bnum], mode='same')

    tt2 = time()

    return Projection


## Backward projection

def get_indices_backward(lensCenters, Resolution, imgSize, texSize):
    num = Resolution['Nnum'][()]
    num_half = Resolution['Nnum_half'][()]

    offsetImg = np.ceil(imgSize / 2)
    offsetVol = np.ceil(texSize / 2)
    lensYpx = lensCenters['px'][()][:, :, 0] + offsetImg[0]
    lensXpx = lensCenters['px'][()][:, :, 1] + offsetImg[1]

    indicesTex = dict()
    indicesImg = dict()

    for aa_sen in range(num[0]):
        for bb_sen in range(num[1]):
            lensXpxCurrent = np.round(lensYpx - num_half[0] + aa_sen)
            lensYpxCurrent = np.round(lensXpx - num_half[1] + bb_sen)

            texScaleFactor = Resolution['texScaleFactor'][()]
            # convert lenses to img space
            lensXvoxCurrent = lensXpxCurrent - offsetImg[0]
            lensXvoxCurrent = (lensXvoxCurrent+1)*texScaleFactor[0]
            lensXvoxCurrent = np.ceil(lensXvoxCurrent + offsetVol[0]) - 1

            lensYvoxCurrent = lensYpxCurrent - offsetImg[1]
            lensYvoxCurrent = (lensYvoxCurrent+1)*texScaleFactor[1]
            lensYvoxCurrent = np.ceil(lensYvoxCurrent + offsetVol[1]) - 1

            # check for out of image and texture
            validLens = (lensXvoxCurrent < texSize[0]) & (lensXvoxCurrent >= 0) & \
                (lensYvoxCurrent < texSize[1]) & (lensYvoxCurrent >= 0) & \
                (lensXpxCurrent < imgSize[0]) & (lensXpxCurrent >= 0) & \
                (lensYpxCurrent < imgSize[1]) & (lensYpxCurrent >= 0)

            lensXpxCurrent = lensXpxCurrent[validLens]
            lensYpxCurrent = lensYpxCurrent[validLens]
            indices = np.vstack([lensXpxCurrent, lensYpxCurrent]).astype('int32')
            indicesImg[(aa_sen, bb_sen)] = tuple(indices)

            lensXvoxCurrent = lensXvoxCurrent[validLens]
            lensYvoxCurrent = lensYvoxCurrent[validLens]
            indices = np.vstack([lensXvoxCurrent, lensYvoxCurrent]).astype('int32')
            indicesTex[(aa_sen, bb_sen)] = tuple(indices)

    return indicesTex, indicesImg

def get_filters_backward(Ht, Resolution, crange):
    num = Resolution['Nnum'][()]
    num_half = Resolution['Nnum_half'][()]
    sensMask = Resolution['sensMask'][()]
    nDepths = Ht.shape[2]

    fshape = Ht[0,0,0].shape
    shape_filters = (num[0], num[1], nDepths, fshape[0], fshape[1])

    filters = np.zeros(shape_filters)

    # cc = 0
    for cc in range(nDepths):
        # aa_sen = 0
        for aa_sen in range(num[0]):
            aa_new = aa_sen;
            flipX = False
            if crange == 'quarter' and aa_sen >= num_half[0]:
                aa_new = num[0] - aa_sen - 1
                flipX = True

            # bb_sen = 0
            for bb_sen in range(num[1]):
                if sensMask[aa_sen, bb_sen] == 0:
                    continue

                bb_new = bb_sen
                flipY = False
                if crange == 'quarter' and bb_sen >= num_half[1]:
                    bb_new = num[1] - bb_sen - 1
                    flipY = True

                # % Fetch corresponding PSF for given coordinate behind the
                # % lenslet
                Hts = Ht[aa_new, bb_new, cc].todense()
                if flipX:
                    Hts = np.flipud(Hts)

                if flipY:
                    Hts = np.fliplr(Hts)

                filters[aa_sen, bb_sen, cc] = Hts

    return filters


def LFM_backwardProject(Ht, projection, lensCenters, Resolution, texSize, crange, step=15):
    num = Resolution['Nnum'][()]
    num_half = Resolution['Nnum_half'][()]
    nDepths = Ht.shape[2]
    imgSize = np.array(projection.shape)

    sensMask = Resolution['sensMask'][()]

    indicesTex, indicesImg = get_indices_backward(lensCenters, Resolution, imgSize, texSize)
    filters = get_filters_backward(Ht, Resolution, crange).astype('float32')

    backproj = cupy.zeros([texSize[0], texSize[1], nDepths], dtype='float32')

    tempSlice = cupy.empty(texSize, dtype='float32')

    filterSize = filters.shape[-2:]

    # filters_cupy = cupy.asarray(filters)
    projection_cupy = cupy.asarray(projection, dtype='float32')

    slices_buffer = cupy.empty((step, texSize[0], texSize[1]), dtype='float32')
    filter_buffer = cupy.empty((step, filterSize[0], filterSize[1]), dtype='float32')


    for cc in trange(nDepths, ncols=70, desc='backward'):
        bnum = 0

        for aa_sen in range(num[0]):
            for bb_sen in range(num[1]):
                if sensMask[aa_sen, bb_sen] == 0:
                    continue

                ix_tex = indicesTex[(aa_sen, bb_sen)]
                ix_img = indicesImg[(aa_sen, bb_sen)]

                # if cupy.sum(projection_cupy[ix_img]) < 0: continue

                tempSlice.fill(0)
                tempSlice[ix_tex] = projection_cupy[ix_img]

                # Hts = filters_cupy[aa_sen, bb_sen, cc]
                # BackProjection[:, :, cc] += signal.fftconvolve(tempSlice, Hts, mode='same')

                slices_buffer[bnum] = tempSlice
                # filter_buffer[bnum] = filters_cupy[aa_sen, bb_sen, cc]
                filter_buffer[bnum] = cupy.asarray(filters[aa_sen, bb_sen, cc])
                bnum += 1

                if bnum >= step:
                    backproj[:, :, cc] += cufftconvsum(slices_buffer, filter_buffer, mode='same')
                    bnum = 0

        if bnum > 0:
            backproj[:, :, cc] += cufftconvsum(slices_buffer[:bnum], filter_buffer[:bnum], mode='same')


    return backproj
