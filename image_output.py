# from typing import Any, Callable, Tuple, List
import os
import numpy as np

from skimage import io
from PIL import Image
from tifffile import imsave


def normalize_uint8(arr):
    arr -= np.min(arr)
    arr /= np.max(arr)
    return (arr * 255).astype(np.uint8)


def normalize_uint16(arr):
    arr -= np.min(arr)
    arr /= np.max(arr)
    return (arr * 65535).astype(np.uint16)


def array3d_to_frames(arr, mode="L"):
    return [Image.fromarray(arr[i], mode=mode) for i in range(arr.shape[0])]


def save_frames(fpath, ext, frames, timestep=40):
    """Save list of Image objects as a file with the given ext, e.g. tiff or gif."""
    frames[0].save(
        fpath + "." + ext,
        save_all=True,
        append_images=frames[1:],
        duration=timestep,
        loop=0,
        optimize=False,
        pallete="I",
        # mode="L",
        # palette="L",
        # palette=ImagePalette.ImagePalette(mode="L"),
    )


def array_to_gif(pth, fname, arr, downsample=1, time_ax=0, timestep=40):
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


def array_to_gif_test(pth, fname, arr, downsample=1, time_ax=0, timestep=40):
    """Takes desired path and filename (without extension) and a numpy matrix and saves
    it as a GIF using the PIL.Image module. If time_ax indicates that the time dim
    is not first, then it will be moved to make the shape = (T, H, W).
    """
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    arr = normalize_uint8(arr)
    frames = []
    for i in range(int(arr.shape[0] / downsample)):
        img = Image.fromarray(arr[i * downsample], mode="P")
        # img.putpalette(b"L")
        # img = img.convert("L")
        frames.append(img)

    # frames = [
    #     Image.fromarray(arr[i * downsample], mode="L").putpalette(
    #         ImagePalette.ImagePalette(mode="L")
    #     )
    #     # Image.fromarray(arr[i * downsample]).convert("L")
    #     # Image.fromarray(arr[i * downsample], mode="L")
    #     # Image.fromarray(arr[i * downsample])
    #     for i in range(int(arr.shape[0] / downsample))
    # ]

    os.makedirs(pth, exist_ok=True)
    save_frames(os.path.join(pth, fname), "gif", frames, timestep=timestep)


def array_to_tiff(pth, fname, arr, time_ax=0, hq=True):
    """Use tifffile library to save 16-bit tiff stack. (PIL can only do 8-bit)."""
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    norm = normalize_uint16 if hq else normalize_uint8

    os.makedirs(pth, exist_ok=True)
    imsave(os.path.join(pth, fname) + ".tif", norm(arr))


def re_export_tiff(pth, fname):
    """Load and re-save tiff. Helpful for dropping bioimage meta-data and formatting
    so that they can be treated as 'normal' tiffs."""
    out_pth = os.path.join(pth, "re_export")
    os.makedirs(out_pth, exist_ok=True)
    imsave(os.path.join(out_pth, fname), io.imread(os.path.join(pth, fname)))


def re_export_folder(pth):
    out_pth = os.path.join(pth, "re_export")
    os.makedirs(out_pth, exist_ok=True)
    for fname in os.listdir(pth):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue
        imsave(os.path.join(out_pth, fname), io.imread(os.path.join(pth, fname)))


def load_gif(pth):
    """Load in multiframe gif into numpy array."""
    img = Image.open(pth)
    stack = []
    for i in range(img.n_frames):
        img.seek(i)
        stack.append(np.array(img))
    return np.stack(stack, axis=0)


def save_tiff(pth, name, tiff):
    os.makedirs(pth, exist_ok=True)
    imsave(os.path.join(pth, name), tiff)
