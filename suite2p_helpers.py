import os
import shutil
import re
import h5py as h5

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from tifffile import imsave

from suite2p import run_s2p, default_ops

from image_arrays import normalize_uint16, array_to_tiff, array_to_gif


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


def get_raw_scans(datapath, prefix='Scan', start_scan=False, num_scans=False):
    """Load in original 2PLSM scans, stack them and return as (T, Y, X) array.
    If start_scan and num_scans are not specified, all scans in the folder are
    loaded.
    """
    if not (start_scan and num_scans):
        fnames = [
            os.path.join(datapath, f)
            for f in os.listdir(datapath)
            if os.path.isfile(os.path.join(datapath, f)) and '_ch' in f
        ]
    else:
        fnames = [
            "%s%s_%03d_ch1.tif" % (datapath, prefix, num)
            for num in range(start_scan, start_scan+num_scans)
        ]

    return np.concatenate([io.imread(fname) for fname in fnames], axis=0)


def get_beams(movie, stats):
    """Pull out z-projection for each cell using it's generated ROI, but without
    pixel weighting assigned by suite2p. Weighted beam extraction is what is
    represented in suite2p's F.npy output.

    Returns an ndarray of shape (N, T).
    """
    beams = np.array([
        movie[:, cell['ypix'], cell['xpix']].mean(axis=1)
        for cell in stats]
    )
    return beams


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


def store_csvs(pth, fldr, Fcell, Fneu, roi_masks):
    """Store recordings and masks as CSVs to make loading in to IgorPro easy.
    Save floats with limited precision, and masks are integers for readability
    and storage space considerations.
    """
    savepth = os.path.join(pth, fldr)
    if not os.path.isdir(savepth):
        os.mkdir(savepth)

    np.savetxt(os.path.join(savepth, 'Fcell.csv'), Fcell.T, '%1.4f', ',')
    np.savetxt(os.path.join(savepth, 'Fneu.csv'), Fneu.T, '%1.4f', ',')

    # Save masks seperately in to a sub-folder
    maskpth = os.path.join(savepth, 'masks')
    if not os.path.isdir(maskpth):
        os.mkdir(maskpth)

    for i, msk in enumerate(roi_masks):
        np.savetxt(os.path.join(maskpth, 'roi%d.csv' % i), msk, '%d', ',')


def plot_signal(Fcell, Fall, idx, norm=False, plot_Fall=False):
    # pull out cell of interest, normalizing if norm=True
    Fcell_wave = Fcell[idx] / Fcell[idx].max() if norm else Fcell[idx]
    Fall_wave = Fall[idx] / Fall[idx].max() if norm else Fall[idx]

    plt.plot(Fcell_wave, label='F')
    plt.plot(Fall_wave, label='F+Fneu') if plot_Fall else 0
    plt.legend()
    plt.show()


def analyze_folder(base_path, diam=8):
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
            "masks": masks.transpose(1, 2, 0),
            "denoised": denoised.transpose(1, 2, 0),
        }

        no_ext = re.sub("\.tif?[f]", "", name)
        array_to_gif(out_path, no_ext + "_denoised", denoised, timestep=150)
        pack_hdf(os.path.join(out_path, no_ext), data)
        del(recs, neu, pixels, masks, denoised)

    shutil.rmtree(os.path.join(base_path, "suite2p"))


if __name__ == '__main__':
    pass
