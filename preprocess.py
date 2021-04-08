import sys
import os
import shutil
import re

from functools import reduce
from skimage import io
from skimage.measure import block_reduce

from image_arrays import *

# TODO: pre-treat stacks
# - QI threshold (muti-trial only) [zero out pixels that fail?]
# - signal/noise threshold


def compose(f, g):
    return lambda a: f(g(a))


def map_tiff(f, pth, label, out_pth=None):
    """Load a tiff from file, process it with the given function `f` and save
    the result."""
    if out_pth is None:
        full_pth = os.path.join(os.path.splitext(pth)[0], label, ".tif")
    else:
        os.makedirs(out_pth, exist_ok=True)
        full_pth = os.path.join(out_pth, os.path.split(pth)[1], label, ".tif")

    mapped = f(io.imread(os.path.join(pth)))
    imsave(full_pth, normalize_uint16(remove_offset(mapped)))


def pipeline_map_tiff(pth, label, *funcs, out_pth=None):
    """Compose one or more functions supplied as the arg list *funcs, then apply
    it with `map_tiff`"""
    f = reduce(compose, funcs, lambda a: a)
    map_tiff(f, pth, label, out_pth=out_pth)


def block_reduce_tiff(pth, reducer, block_size=(1, 2, 2), pad_val=0, **reducer_kwargs):
    """Load a tiff from file, process it with block_reduce and save the result."""
    f = lambda a: block_reduce(a, block_size, reducer, pad_val, reducer_kwargs)
    label = "pooled_%i" % "_".join(map(str, block_size))
    map_tiff(f, pth, label)


def qi_threshold():
    pass


def snr_threshold():
    pass
