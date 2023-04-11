#!/usr/bin/env ipython

import numpy as np
from bisect import bisect_left
from scipy.fft import next_fast_len

try:
    import cupy
    from cupy import fuse
except ImportError:
    cupy = np
    def fuse(f): # empty wrapper
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper

def _centered_numpy(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = np.array(arr.shape[-2:])
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[..., startind[0]:endind[0], startind[1]:endind[1]]

@fuse
def sum_product(sp1, sp2):
    return cupy.sum(sp1 * sp2, axis=0)

def cufftconv(in1, in2, mode="full", sumit=False):
    # in1 = cupy.ascontiguousarray(in1)
    # in2 = cupy.ascontiguousarray(in2)

    # Extract shapes
    s1 = np.array(in1.shape[-2:])
    s2 = np.array(in2.shape[-2:])
    shape = s1 + s2 - 1

    fshape = [next_fast_len(d, True) for d in shape]

    # Compute convolution in fourier space
    sp1 = cupy.fft.rfft2(in1, fshape)
    sp2 = cupy.fft.rfft2(in2, fshape)
    if sumit:
        fslice = tuple([slice(sz) for sz in shape])
        # spsum = cupy.sum(sp1 * sp2, axis=0)
        spsum = sum_product(sp1, sp2)
        ret = cupy.fft.irfft2(spsum[None], fshape)[0][fslice]
    else:
        fslice = tuple([slice(sz) for sz in in1.shape[:-2] + tuple(shape)])
        ret = cupy.fft.irfft2(sp1 * sp2, fshape)[fslice]

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered_numpy(ret, s1)
    elif mode == "valid":
        cropped = _centered_numpy(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    return cropped

def cufftconvsum(in1, in2, mode="full"):
    return cufftconv(in1, in2, mode, sumit=True)
