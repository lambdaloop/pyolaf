#!/usr/bin/env python3

import numpy as np
# from skimage import transform
from scipy.spatial.transform import Rotation

try:
    import cupy
    from cupyx.scipy.ndimage import affine_transform, shift
    has_cupy = True
except ImportError:
    cupy = np
    from scipy.ndimage import affine_transform, shift
    has_cupy = False


def LFM_retrieveTransformation(LensletGridModel, NewLensletGridModel):

    # scale transform
    InputSpacing = np.array([LensletGridModel['HSpacing'], LensletGridModel['VSpacing']])
    NewSpacing = np.array([NewLensletGridModel['HSpacing'], NewLensletGridModel['VSpacing']])
    XformScale = NewSpacing / InputSpacing

    RScale = np.eye(3)
    RScale[0,0] = XformScale[0]
    RScale[1,1] = XformScale[1]

    NewOffset = np.array([LensletGridModel['HOffset'], LensletGridModel['VOffset']]) * XformScale
    RoundedOffset = np.round(NewOffset).astype(int)
    XformTrans = RoundedOffset - NewOffset

    RTrans = np.eye(3)
    RTrans[-1,:2] = XformTrans

    # rotation transform
    RRot = Rotation.from_euler('ZYX', [LensletGridModel['Rot'], 0, 0]).as_matrix()

    # final transformation
    FixAll = RRot @ RScale @ RTrans
    # FixAll = transform.AffineTransform(RRot @ RScale @ RTrans)

    return FixAll


def transform_img(img, ttnew, lens_offset):
    scale = np.diag(ttnew)[:2]
    new_shape = np.floor(np.round(img.shape / scale) / 2) * 2 + 1
    new_shape = new_shape.astype('int32')
    offset = np.ceil(new_shape / 2) - lens_offset
    # this transformation has some discrepancy to matlab code
    # it's the biggest discrepancy across the pipeline
    new = affine_transform(img, cupy.asarray(ttnew[:2, :2]),
                           offset=ttnew[:2,2],
                           output_shape=tuple(new_shape),
                           order=1, prefilter=False)
    # new = affine_transform(new, cupy.eye(2), offset=[-offset[0], -offset[1]], order=1)
    new = shift(new, [offset[0], offset[1]], order=1, prefilter=False)
    new[new < np.mean(new)] = np.mean(new)
    return new

def get_transformed_shape(img_shape, ttnew):
    scale = np.diag(ttnew)[:2]
    new_shape = np.floor(np.round(np.array(img_shape) / scale) / 2) * 2 + 1
    new_shape = new_shape.astype('int32')
    return new_shape

def format_transform(FixAll):
    transform = np.linalg.inv(FixAll).T
    ttnew = np.zeros_like(transform)
    ttnew[:2,:2] = transform[:2,:2].T
    ttnew[:, 2] = transform[[1, 0, 2], 2]
    return ttnew
