import os
import numpy as np

from image_output import array_to_gif
from hdf_utils import *


def get_suite2p_data(pth, exclude_non_cells=False):
    """Load extracted recordings (raw fluoresence (F), neuropil (Fneu)) and cell
    information (stat, which includes ROI pixels (xpix, ypix) and their
    weighting (lam)) from the given suite2p output folder.

    recs & recs_neu: (N, T) ndarray
    stats: Numpy object containing of dicts for all N cells.
    """
    recs = np.load(os.path.join(pth, "F.npy"))
    neu = np.load(os.path.join(pth, "Fneu.npy"))
    stats = np.load(os.path.join(pth, "stat.npy"), allow_pickle=True)

    # narrow down selection to those ROIs determined to be "cells" by suite2p
    if exclude_non_cells:
        cell_ids = np.nonzero(
            np.load(os.path.join(pth, "iscell.npy"))[:, 0].astype(int)
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
        masks[idx, stats[idx]["ypix"], stats[idx]["xpix"]] = stats[idx]["lam"]
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
        arr[:, stats[i]["ypix"], stats[i]["xpix"]] += stats[i]["lam"] * beams[
            i
        ].reshape(-1, 1)
    return arr


def pack_suite2p(
    s2p_pth,
    out_path,
    out_name,
    space_dims,
    gif_timestep,
    trial_pts=None,
    exclude_non_cells=False,
    denoised_movies=False,
):
    recs, neu, stats = get_suite2p_data(s2p_pth, exclude_non_cells)

    # x and y pix are swapped to coincide with IgorPro conventions
    pixels = {
        str(i): {"y": n["xpix"], "x": n["ypix"], "weights": n["lam"]}
        for i, n in enumerate(stats)
    }

    n_pts = recs.shape[1]
    masks = create_masks(stats, space_dims)

    # masks and denoised axes transposed to fit with IgorPro conventions
    data = {
        "pixels": pixels,
        "masks": masks.transpose(2, 1, 0),  # N to last dim, swap X and Y
        "space_dims": np.array(space_dims),
    }

    if denoised_movies:
        denoised = roi_movie(recs - neu * 0.7, stats, (n_pts, *space_dims))
        data["denoised"] = denoised.transpose(2, 1, 0)  # time to last dim, swap X and Y
        array_to_gif(out_path, out_name + "_denoised", denoised, timestep=gif_timestep)
        # from tifffile import imsave
        # from image_arrays import array_to_tiff

        # array_to_tiff(out_path, out_name + "_denoised", denoised)
        del denoised

    if trial_pts is None:
        data["recs"] = recs
        data["Fneu"] = neu
    else:
        t0 = 0
        for name, pts in trial_pts.items():
            t1 = t0 + pts
            data[name] = {"recs": recs[:, t0:t1], "Fneu": neu[:, t0:t1]}
            t0 = t1

    pack_hdf(os.path.join(out_path, out_name), data)
    del (recs, neu, pixels, masks)


def pixels_to_s2p_stats(pixels):
    return {
        int(i): {"xpix": px["x"], "ypix": px["y"], "lam": px["weights"]}
        for i, px in pixels.items()
    }


def pixels_to_beams(rec, pixels, use_weights=True):
    if use_weights:
        roi_sum = lambda frame, xs, ys, ws: (
            np.sum([frame[x, y] * w for x, y, w in zip(xs, ys, ws)])
        )
    else:
        roi_sum = lambda frame, xs, ys, _: (
            np.sum([frame[x, y] for x, y in zip(xs, ys)])
        )

    return np.array(
        [
            [roi_sum(fr, px["x"], px["y"], px["weights"]) for fr in rec]
            for px in pixels.values()
        ]
    )


def beams_to_movie(beams, pixels, space_dims):
    return roi_movie(beams, pixels_to_s2p_stats(pixels), (beams.shape[1], *space_dims))


def s2p_hdf_to_roi_movie(pth):
    with h5.File(pth, "r") as f:
        data = unpack_hdf(f)

    return beams_to_movie(
        data["recs"] - data["Fneu"] * 0.7, data["pixels"], data["space_dims"]
    )
