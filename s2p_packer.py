import sys
import os
import shutil
import re

import h5py as h5
import numpy as np
from skimage import io
from PIL import Image


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
        cell_ids = np.nonzero(
            np.load(os.path.join(pth, 'iscell.npy'))[:, 0].astype(np.int)
        )[0]
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
        arr[:, stats[i]["ypix"],
            stats[i]["xpix"]] += (stats[i]["lam"] * beams[i].reshape(-1, 1))
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


def pack_suite2p(
    s2p_pth, out_path, out_name, stack_dims, gif_timestep, exclude_non_cells=False
):
    recs, neu, stats = get_suite2p_data(s2p_pth, exclude_non_cells)

    # x and y pix are swapped to coincide with IgorPro conventions
    pixels = {
        str(i): {
            "y": n["xpix"],
            "x": n["ypix"],
            "weights": n["lam"]
        }
        for i, n in enumerate(stats)
    }

    masks = create_masks(stats, stack_dims[1:])
    denoised = roi_movie(recs - neu * 0.7, stats, stack_dims)

    # masks and denoised axes transposed to fit with IgorPro conventions
    data = {
        "Fcell": recs,
        "Fneu": neu,
        "pixels": pixels,
        "masks": masks.transpose(2, 1, 0),  # N to last dim, swap X and Y
        "denoised": denoised.transpose(2, 1, 0),  # time to last dim, swap X and Y
    }

    array_to_gif(out_path, out_name + "_denoised", denoised, timestep=gif_timestep)
    pack_hdf(os.path.join(out_path, out_name), data)
    del (recs, neu, pixels, masks, denoised)


if __name__ == "__main__":
    settings = {}
    for arg in sys.argv[1:]:
        try:
            k, v = arg.split("=")
        except:
            msg = "Invalid argument format. Given %s, but expecting " % arg
            msg += "a key value pair delimited by '=' (e.g. gif_timestep=100)."
            print(msg)
        settings[k] = v

    base_path = os.getcwd()
    names = [
        f for f in os.listdir(base_path) if (f.endswith(".tiff") or f.endswith(".tif"))
    ]
    stack_dims = io.imread(os.path.join(base_path, names[0])).shape

    out_path = os.path.join(base_path, "s2p")
    os.makedirs(out_path, exist_ok=True)
    out_name = re.sub("\.tif?[f]", "", names[0])
    s2p_pth = os.path.join(base_path, "suite2p", "plane0")

    pack_suite2p(
        s2p_pth,
        out_path,
        out_name,
        stack_dims,
        gif_timestep=int(settings.get("gif_timestep", 200)),
        exclude_non_cells=int(settings.get("only_cells", 0)),
    )
