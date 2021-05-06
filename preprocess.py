import sys
import os
import shutil
import re

import numpy as np
from functools import reduce
from skimage import io
from skimage.measure import block_reduce

from image_arrays import *


def is_tiff(name):
    return (name.endswith(".tiff") or name.endswith(".tif"))


def compose(f, g):
    return lambda a: g(f(a))


def prepare_full_path(load_path, label, out_path=None):
    load_folder, name = os.path.split(load_path)
    n, ext = os.path.splitext(name)
    new_name = "%s_%s%s" % (n, label, ext)
    if out_path is None:
        return os.path.join(load_folder, new_name)
    else:
        os.makedirs(out_path, exist_ok=True)
        return os.path.join(out_path, new_name)


def map_tiff(f, pth, label, out_pth=None):
    """Load a tiff from file, process it with the given function `f` and save
    the result."""
    full_out_path = prepare_full_path(pth, label, out_pth)
    mapped = f(io.imread(os.path.join(pth)))
    imsave(full_out_path, normalize_uint16(remove_offset(mapped)))


def pipeline_map_tiff(pth, label, *funcs, out_pth=None):
    """Compose one or more functions supplied as the arg list *funcs, then apply
    it with `map_tiff`"""
    f = reduce(compose, funcs, lambda a: a)
    map_tiff(f, pth, label, out_pth=out_pth)


def multi_trial_tiff_pipeline(pth, label, *funcs, out_pth=None):
    """Like `pipeline_map_tiff`, but processes the folder given by path as set
    of trials performed on the same scan field, with the same stimuli. Thus, the
    given `funcs` should all be able to operate on 4-dimensional arrays of shape
    (Trials, Time, X, Y). After processing by the pipeline, the trials are split
    and saved as separate tiffs."""
    names = [f for f in os.listdir(pth) if (f.endswith(".tiff") or f.endswith(".tif"))]
    stacks = np.stack([io.imread(os.path.join(pth, f)) for f in names], axis=0)
    mapped = reduce(compose, funcs, lambda a: a)(stacks)
    for name, s in zip(names, normalize_uint16(remove_offset(mapped))):
        full_out_path = prepare_full_path(os.path.join(pth, name), label, out_pth)
        imsave(full_out_path, s)


def block_reduce_tiff(pth, reducer, block_size=(1, 2, 2), pad_val=0, **reducer_kwargs):
    """Load a tiff from file, process it with block_reduce and save the result."""
    f = lambda a: block_reduce(a, block_size, reducer, pad_val, reducer_kwargs)
    label = "pooled_%i" % "_".join(map(str, block_size))
    map_tiff(f, pth, label)


def crop_sides(arr, x_edge, y_edge):
    y_size = arr.shape[-2]
    x_size = arr.shape[-1]
    if arr.ndim > 3:
        return arr[:, :, y_edge:(y_size - y_edge), x_edge:(x_size - x_edge)]
    else:
        return arr[:, y_edge:(y_size - y_edge), x_edge:(x_size - x_edge)]


def qi_threshold(arr, thresh, mask_val=0):
    """Replace pixels/beams that do not pass the target signal-to-noise ratio."""
    n_trials, t, x, y = arr.shape
    mask = np.stack(
        [
            quality_index(beams)
            for beams in arr.reshape(n_trials, t, -1).transpose(2, 0, 1)
        ],
        axis=0
    ).reshape(x, y) > thresh
    arr[:, :, mask] = mask_val
    return arr


def snr_threshold(arr, bsln_start, bsln_end, stim_start, stim_end, thresh, mask_val=0):
    """Replace pixels/beams that do not pass the target signal-to-noise ratio
    with the given `mask_val`. This is done on a trial by trial basis, to the
    expected input is a three dimensional Time x Space array."""
    bsln_var = np.var(arr[bsln_start:bsln_end], axis=0)
    stim_var = np.var(arr[stim_start:stim_end], axis=0)
    mask = (stim_var / bsln_var) > thresh
    arr[:, mask] = mask_val
    return arr


def process_folders(base_path, new_base, *funcs, copy_dirs={"noise"}, multi_trial=True):
    def loop(child_path):
        pth = os.path.join(base_path, child_path)
        contents = os.listdir(pth)
        names, children = [], []
        for c in contents:
            if is_tiff(c):
                names.append(c)
            elif os.path.isdir(os.path.join(pth, c)):
                if c in copy_dirs:
                    dest = os.path.join(new_base, child_path, c)
                    if not os.path.exists(dest):
                        shutil.copytree(os.path.join(pth, c), dest)
                else:
                    loop(os.path.join(child_path, c))
        if len(names) > 0:
            if multi_trial:
                multi_trial_tiff_pipeline(
                    pth, "", *funcs, out_pth=os.path.join(new_base, child_path)
                )
            else:
                for n in names:
                    pipeline_map_tiff(
                        os.path.join(pth, n),
                        "",
                        *funcs,
                        out_pth=os.path.join(new_base, child_path)
                    )

    loop("")


def settings_to_pipeline(settings):
    multi_trial = settings.pop("multi_trial", False)

    def to_fun(key, arg):
        if key == "crop":
            try:
                x, y = arg.split(",")
                return lambda a: crop_sides(a, x, y)
            except:
                msg = "Expected crop argument to be a comma separated pair of the number"
                msg += " of x and y pixels to cut from each side. (e.g. crop=48,0)"
        elif key == "reduce":
            try:
                x, y, z = [int(d) for d in arg.split(",")]
                dims = (x, y, z) if not multi_trial else (1, x, y, z)
                return lambda a: block_reduce(a, dims, np.mean, 0)
            except:
                msg = "Expected a comma separated list of the number of pixels to reduce"
                msg += " over in each dimension (time, x, y). (e.g. reduce=1,4,4)"
        elif key == "qi" and multi_trial:
            try:
                threshold = float(arg)
                return lambda a: qi_threshold(a, threshold)
            except:
                msg = "Expected a value convertable to float to serve as the quality"
                msg += " index threshold. (e.g. qi=0.4)"
        elif key == "qi" and not multi_trial:
            msg = "The qi option (quality index thresholding) cannot be used unless the"
            msg += " data to be processed is multiple trials (organized accordingly)."
            msg += " If it is, ensure that the argument 'multi_trial=1' is given."
        else:
            return lambda a: a
        print(msg)

    return [to_fun(k, v) for k, v in settings.items()]


if __name__ == "__main__":
    settings = {}
    for arg in sys.argv[1:]:
        try:
            k, v = arg.split("=")
        except:
            msg = "Invalid argument format. Given %s, but expecting " % arg
            msg += "a key value pair delimited by '=' (e.g. diam=8). "
            print(msg)
        settings[k] = v

    cwd = os.getcwd()
    parent_path, target_dir = os.path.split(cwd)
    processed_path = os.path.join(parent_path, "%s_processed" % target_dir)
    process_folders(cwd, processed_path, *settings_to_pipeline(settings))
