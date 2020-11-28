import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from PIL import Image
from tifffile import imsave
import torch
import torch.nn.functional as F

"""
NOTE: Discoveries in working with suite2p again.
- Can provide data as h5 archives with a stack stored under the key "data"
- h5 array data cannot be float, must be uint16 (like a tiff)
"""

class StackPlotter:
    """
    Returns Object for cycling through frames of a 3D image stack using the
    mouse scroll wheel. Takes the pyplot axis object and data as the first two
    arguments. Additionally, use delta to set the number of frames each step of
    the wheel skips through.
    """
    def __init__(self, ax, stack, delta=10, vmin=None, vmax=None, cmap="gray"):
        self.ax = ax
        self.stack = stack
        self.slices = stack.shape[0]
        self.idx = 0
        self.delta = delta

        self.im = ax.imshow(
            self.stack[self.idx, :, :], cmap=cmap, vmin=vmin, vmax=vmax
        )
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.idx = (self.idx + self.delta) % self.slices
        else:
            self.idx = (self.idx - self.delta) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.stack[self.idx, :, :])
        self.ax.set_ylabel("t = %s" % self.idx)
        self.im.axes.figure.canvas.draw()

    def connect_scroll(self, fig):
        fig.canvas.mpl_connect("scroll_event", self.onscroll)


def plot_stack(arr, delta=10, vmin=None, vmax=None, cmap="gray"):
    fig, ax = plt.subplots(1)
    stack = StackPlotter(ax, arr, delta, vmin, vmax, cmap)
    stack.connect_scroll(fig)
    fig.tight_layout()
    plt.show()


def array3d_to_frames(arr, mode="L"):
    return [Image.fromarray(arr[i], mode=mode) for i in range(arr.shape[0])]


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


def remove_offset(arr):
    return arr - arr.min()


def normalize_uint8(arr, max_val=None):
    max_val = arr.max() if max_val is None else max_val
    return (arr / max_val * 255).clip(0, 255).astype(np.uint8)


def normalize_uint16(arr, max_val=None):
    max_val = arr.max() if max_val is None else max_val
    return (arr / max_val * 65535).clip(0, 65535).astype(np.uint16)


def array_to_gif(pth, fname, arr, max_val=None, downsample=1, time_ax=0, timestep=40):
    """
    Takes desired path and filename (without extension) and a numpy matrix and saves
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


def array_to_tiff(pth, fname, arr, time_ax=0, max_val=None):
    """Use tifffile library to save 16-bit tiff stack. (PIL can only do 8-bit)."""
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    os.makedirs(pth, exist_ok=True)

    imsave(os.path.join(pth, fname) + ".tiff", normalize_uint16(arr, max_val))


def array_to_h5(pth, fname, arr, time_ax=0):
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    os.makedirs(pth, exist_ok=True)

    with h5.File(os.path.join(pth, fname) + ".h5", "w") as f:
        f.create_dataset("data", data=arr)


def re_export_tiff(pth, fname):
    """Load and re-save tiff. Helpful for dropping bioimage meta-data and formatting
    so that they can be treated as 'normal' tiffs."""
    out_pth = os.path.join(pth, "re_export")
    os.makedirs(out_pth, exist_ok=True)
    imsave(os.path.join(out_pth, fname), io.imread(os.path.join(pth, fname)))


def pool_tiff(pth, fname, kernel_size=2, stride=None, padding=0):
    arr    = io.imread(os.path.join(pth, fname))
    t      = torch.from_numpy(arr.astype(float))
    pooled = F.avg_pool2d(t, kernel_size, stride, padding).numpy()
    u16    = normalize_uint16(remove_offset(pooled))
    label  = "pooled_%i" % kernel_size
    array_to_tiff(os.path.join(pth, label), label, u16)
