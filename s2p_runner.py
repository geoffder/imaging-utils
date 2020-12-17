import sys
import os
import shutil
import re

import h5py as h5
import numpy as np
from skimage import io
from PIL import Image

from suite2p import run_s2p, default_ops


def normalize_uint8(arr, max_val=None):
    max_val = arr.max() if max_val is None else max_val
    return (arr / max_val * 255).clip(0, 255).astype(np.uint8)


def save_frames(fpath, ext, frames, timestep=40):
    frames[0].save(
        fpath + "." + ext,
        save_all=True,
        append_images=frames[1:],
        duration=timestep,
        loop=0,
        optimize=False,
        pallete='I'
    )


def array_to_gif(pth, fname, arr, max_val=None, downsample=1, time_ax=0, timestep=40):
    """Takes desired path and filename (without extension) and a numpy matrix and saves
    it as a GIF using the PIL.Image module. If time_ax indicates that the time dim
    is not first, then it will be moved to make the shape = (T, H, W).
    """
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    arr = normalize_uint8(arr)
    frames = [
        Image.fromarray(arr[i * downsample], mode="P")
        for i in range(int(arr.shape[0] / downsample))
    ]

    os.makedirs(pth, exist_ok=True)
    save_frames(os.path.join(pth, fname), "gif", frames, timestep=timestep)


def bipolar_ops(diam=8, allow_overlap=True):
    ops = default_ops()
    ops["spikedetect"] = False
    ops["sparse_mode"] = False
    ops["diameter"] = diam
    ops["allow_overlap"] = allow_overlap
    return ops


def get_suite2p_data(pth, exclude_non_cells=True):
    """Load extracted recordings (raw fluoresence (F), neuropil (Fneu)) and cell
    information (stat, which includes ROI pixels (xpix, ypix) and their
    weighting (lam)) from the given suite2p output folder.

    recs & recs_neu: (N, T) ndarray
    stats: Numpy object containing of dicts for all N cells.
    """
    recs = np.load(os.path.join(pth, 'F.npy'))
    neu = np.load(os.path.join(pth, 'Fneu.npy'))
    stats = np.load(os.path.join(pth, 'stat.npy'), allow_pickle=True)

    # narrow down selection to those ROIs determined to be "cells" by suite2p
    if exclude_non_cells:
        cell_ids = np.nonzero(np.load(os.path.join(pth, 'iscell.npy'))[:, 0].astype(np.int))[0]
        recs = recs[cell_ids, :]
        neu = neu[cell_ids, :]
        stats = stats[cell_ids]

    return recs, neu, stats


def create_masks(stats, dims):
    """Use ypix and xpix index arrays and the corresponding weights found in lam
    to generate weighted spatial masks for each cell.
    """
    masks = np.zeros((stats.size, *dims))
    for idx in range(stats.size):
        masks[idx, stats[idx]['ypix'], stats[idx]['xpix']] = stats[idx]["lam"]
    return masks


def roi_movie(beams, stats, dims):
    """Create stack from a beams array shape:(N, T) representing signal over time
    for N rois. `stats` is an array of length N containing dicts with the x and y
    pixel indices of the rois (and the weighting of each pixel in field lam)
    corresponding to the signals in beams. Pixel indices correspond to the dimensions
    of the stack originally fed into suite2p.
    """
    arr = np.zeros(dims)
    for i in range(beams.shape[0]):
        arr[:, stats[i]["ypix"], stats[i]["xpix"]] += (
            stats[i]["lam"] * beams[i].reshape(-1, 1))
    return arr


def pack_hdf(pth, data_dict, compression="gzip"):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure."""
    def rec(data, grp):
        for k, v in data.items():
            if type(v) is dict:
                rec(v, grp.create_group(k))
            else:
                grp.create_dataset(k, data=v, compression=compression)

    with h5.File(pth + ".h5", "w") as pckg:
        rec(data_dict, pckg)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }


def analyze_folder(base_path, diam=8, gif_timestep=200):
    names = [
        f for f in os.listdir(base_path)
        if (f.endswith(".tiff") or f.endswith(".tif"))
    ]
    stacks = [io.imread(os.path.join(base_path, f)) for f in names]

    out_path = os.path.join(base_path, "s2p")
    os.makedirs(out_path, exist_ok=True)

    s2p_pth = os.path.join(base_path, "suite2p", "plane0")
    ops = bipolar_ops(diam=diam)
    db = {"data_path": [base_path]}

    for name, stack in zip(names, stacks):
        db["tiff_list"] = [name]
        _out_ops = run_s2p(ops, db)
        recs, neu, stats = get_suite2p_data(s2p_pth, exclude_non_cells=False)

        pixels = {
            str(i): {"y": n["ypix"], "x": n["xpix"], "weights": n["lam"]}
            for i, n in enumerate(stats)
        }

        dims = stack.shape
        masks = create_masks(stats, dims[1:])
        denoised = roi_movie(recs - neu * 0.7, stats, dims)

        data = {
            "Fcell": recs,
            "Fneu": neu,
            "pixels": pixels,
            "masks": masks.transpose(1, 2, 0),        # swap N to last dimension
            "denoised": denoised.transpose(1, 2, 0),  # swap time to last dimension
        }

        no_ext = re.sub("\.tif?[f]", "", name)
        array_to_gif(out_path, no_ext + "_denoised", denoised, timestep=gif_timestep)
        pack_hdf(os.path.join(out_path, no_ext), data)
        del(recs, neu, pixels, masks, denoised)

    shutil.rmtree(os.path.join(base_path, "suite2p"))


if __name__ == "__main__":
    settings = {}
    for arg in sys.argv[1:]:
        try:
            k, v = arg.split("=")
        except:
            msg = "Invalid argument format. Given %s, but expecting " % arg 
            msg += "a key value pair delimited by '=' (e.g. diam=8)."
            print(msg)
        settings[k] = v

    analyze_folder(
        os.getcwd(),
        diam=int(settings.get("diam", 8)),
        gif_timestep=int(settings.get("gif_timestep", 200)),
    )
