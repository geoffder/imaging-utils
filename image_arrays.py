import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox
from skimage import io
from scipy import signal
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
    """Returns Object for cycling through frames of a 3D image stack using the
    mouse scroll wheel. Takes the pyplot axis object and data as the first two
    arguments. Additionally, use delta to set the number of frames each step of
    the wheel skips through."""
    def __init__(self, fig, ax, stack, delta=10, vmin=None, vmax=None, cmap="gray"):
        self.fig = fig
        self.ax = ax
        self.stack = stack
        self.slices = stack.shape[0]
        self.idx = 0
        self.delta = delta
        self.im = ax.imshow(self.stack[self.idx, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        self.update()
        self.connect_scroll()

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

    def connect_scroll(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)


class StackExplorer:
    def __init__(
        self,
        stack,
        zaxis=None,
        delta=1,
        roi_sz=1,
        vmin=None,
        vmax=None,
        cmap="gray",
        **plot_kwargs
    ):
        if "gridspec_kw" not in plot_kwargs:
            plot_kwargs["gridspec_kw"] = {"height_ratios": [.7, .3]}
        self.fig, self.ax = plt.subplots(2, **plot_kwargs)
        self.stack, self.delta, self.roi_sz = stack, delta, roi_sz
        self.z_sz, self.y_sz, self.x_sz = stack.shape
        self.z_idx, self.roi_x, self.roi_y = 0, 0, 0
        self.zaxis = np.arange(self.z_sz) if zaxis is None else zaxis
        self.roi_locked = False

        self.build_stack_ax(cmap, vmin, vmax)
        self.build_roi_ax(vmin, vmax)
        self.connect_events()

    def build_stack_ax(self, cmap, vmin, vmax):
        self.im = self.ax[0].imshow(
            self.stack[self.z_idx, :, :], cmap=cmap, vmin=vmin, vmax=vmax
        )
        self.roi_rect = Rectangle(
            (self.roi_x - .5, self.roi_y - .5),
            self.roi_sz,
            self.roi_sz,
            fill=False,
            color="red",
            linewidth=2,
        )
        self.ax[0].add_patch(self.roi_rect)
        self.update_im()

    def build_roi_ax(self, vmin, vmax):
        beam = self.update_beam()
        self.roi_line = self.ax[1].plot(self.zaxis, beam)[0]
        self.z_marker = self.ax[1].plot(self.zaxis[self.z_idx], beam[self.z_idx], "x")[0]
        self.ax[1].set_ylim(vmin, vmax)
        self.update_roi()

    def update_beam(self):
        if self.roi_sz > 1:
            self.beam = np.mean(
                self.stack[:, self.roi_y:self.roi_y + self.roi_sz,
                           self.roi_x:self.roi_x + self.roi_sz],
                axis=(1, 2)
            )
        else:
            self.beam = self.stack[:, self.roi_y, self.roi_x]
        return self.beam

    def on_scroll(self, event):
        if event.button == "up":
            self.z_idx = (self.z_idx + self.delta) % self.z_sz
        else:
            self.z_idx = (self.z_idx - self.delta) % self.z_sz
        self.z_marker.set_data(self.zaxis[self.z_idx], self.beam[self.z_idx])
        self.update_im()

    def on_move(self, event):
        if (
            not self.roi_locked and event.inaxes == self.ax[0] and
            not (event.xdata is None or event.ydata is None)
        ):
            x = np.round(event.xdata).astype(np.int)
            y = np.round(event.ydata).astype(np.int)
            if 0 <= x < self.x_sz and 0 <= y < self.x_sz and (
                self.roi_x != x or self.roi_y != y
            ):
                self.roi_x, self.roi_y = x, y
                self.roi_rect.set_xy((self.roi_x - .5, self.roi_y - .5))
                self.update_roi()

    def on_im_click(self, event):
        if (event.button == 1 and event.inaxes == self.ax[0]):
            self.roi_locked = False if self.roi_locked else True
            self.update_roi()

    def update_im(self):
        self.im.set_data(self.stack[self.z_idx, :, :])
        self.ax[0].set_ylabel("z = %s" % self.z_idx)
        self.im.axes.figure.canvas.draw()

    def update_roi(self):
        msg = "(click to %s)" % ("unlock" if self.roi_locked else "lock")
        self.ax[1].set_title("x = %i; y = %i; %s" % (self.roi_x, self.roi_y, msg))
        beam = self.update_beam()
        self.roi_line.set_ydata(beam)
        self.z_marker.set_data(self.zaxis[self.z_idx], beam[self.z_idx])

    def connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_im_click)


class PeakExplorer:
    def __init__(self, xaxis, recs, defaults={}):
        self.xaxis, self.recs = xaxis, recs
        self.n_rois, self.pts = recs.shape
        self.thresh = defaults.get("thresh", 1)
        self.peak_width = defaults.get("peak_width", 2)
        self.peak_tolerance = defaults.get("peak_tolerance", .5)
        self.peak_interval = defaults.get("peak_interval", 1)

        self.build_fig()
        self.build_inputs()
        self.connect_events()

        self.idx = 0
        self.update_peaks()
        self.rec_line = self.rec_ax.plot(self.xaxis, self.recs[self.idx])[0]
        self.peak_line = self.rec_ax.plot(
            self.xaxis[self.peaks], self.recs[self.idx, self.peaks], "x"
        )[0]

    def build_fig(self):
        self.fig = plt.figure(constrained_layout=True, figsize=(6, 6))
        gs = self.fig.add_gridspec(nrows=3, ncols=2, height_ratios=[.8, .1, .1])
        self.rec_ax = self.fig.add_subplot(gs[0, :])
        self.thresh_ax = self.fig.add_subplot(gs[1, 0])
        self.width_ax = self.fig.add_subplot(gs[1, 1])
        self.toler_ax = self.fig.add_subplot(gs[2, 0])
        self.inter_ax = self.fig.add_subplot(gs[2, 1])
        self.rec_ax.set_title("roi = 0")
        self.rec_ax.set_xlabel("Time (s)")
        self.thresh_ax.set_title("Threshold")
        self.width_ax.set_title("Peak Width")
        self.toler_ax.set_title("Peak Tolerance")
        self.inter_ax.set_title("Peak Interval")

    def build_inputs(self):
        self.thresh_box = TextBox(self.thresh_ax, "", initial=str(self.thresh))

        self.width_box = TextBox(self.width_ax, "", initial=str(self.peak_width))

        self.toler_box = TextBox(self.toler_ax, "", initial=str(self.peak_tolerance))

        self.inter_box = TextBox(self.inter_ax, "", initial=str(self.peak_interval))

    def set_thresh(self, s):
        try:
            self.thresh = float(s)
            self.refresh()
        except:
            self.thresh_box.set_val(str(self.thresh))

    def set_peak_width(self, s):
        try:
            self.peak_width = int(s)
            self.refresh()
        except:
            self.width_box.set_val(str(self.peak_width))

    def set_peak_tolerance(self, s):
        try:
            a = float(s)
            if 0 <= a <= 1:
                self.peak_tolerance = a
                self.refresh()
            else:
                raise (ValueError("tolerance must be between 0 and 1."))
        except ValueError:
            self.toler_box.set_val(str(self.peak_tolerance))

    def set_peak_interval(self, s):
        try:
            a = int(s)
            if a >= 1:
                self.peak_interval = a
                self.refresh()
            else:
                raise (ValueError("peak interval must be >= 1"))
        except ValueError:
            self.inter_box.set_val(str(self.peak_interval))

    def update_peaks(self):
        self.peaks, _ = signal.find_peaks(
            self.recs[self.idx],
            prominence=self.thresh,
            rel_height=self.peak_tolerance,
            width=self.peak_width,
            distance=self.peak_interval
        )

    def update_view(self):
        self.rec_ax.set_title("roi = %i" % self.idx)
        rec = self.recs[self.idx]
        self.rec_line.set_ydata(rec)
        self.peak_line.set_data(self.xaxis[self.peaks], rec[self.peaks])
        self.rec_ax.set_ylim(rec.min(), rec.max())

    def refresh(self):
        self.update_peaks()
        self.update_view()

    def on_scroll(self, event):
        if event.button == "up":
            self.idx = (self.idx + 1) % self.n_rois
        else:
            self.idx = (self.idx - 1) % self.n_rois
        self.refresh()

    def connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.thresh_box.on_submit(self.set_thresh)
        self.width_box.on_submit(self.set_peak_width)
        self.toler_box.on_submit(self.set_peak_tolerance)
        self.inter_box.on_submit(self.set_peak_interval)


def plot_stack(arr, delta=10, vmin=None, vmax=None, cmap="gray"):
    fig, ax = plt.subplots(1)
    stack = StackPlotter(fig, ax, arr, delta, vmin, vmax, cmap)
    fig.tight_layout()
    plt.show()


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


def array_to_tiff(pth, fname, arr, time_ax=0, max_val=None, hq=True):
    """Use tifffile library to save 16-bit tiff stack. (PIL can only do 8-bit)."""
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    norm = normalize_uint16 if hq else normalize_uint8

    os.makedirs(pth, exist_ok=True)
    imsave(os.path.join(pth, fname) + ".tif", norm(arr, max_val))


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


def re_export_folder(pth):
    out_pth = os.path.join(pth, "re_export")
    os.makedirs(out_pth, exist_ok=True)
    for fname in os.listdir(pth):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue
        imsave(os.path.join(out_pth, fname), io.imread(os.path.join(pth, fname)))


def pool_tiff(pth, fname, kernel_size=2, stride=None, padding=0):
    arr = io.imread(os.path.join(pth, fname))
    t = torch.from_numpy(arr.astype(float))
    pooled = F.avg_pool2d(t, kernel_size, stride, padding).numpy()
    u16 = normalize_uint16(remove_offset(pooled))
    label = "pooled_%i" % kernel_size
    array_to_tiff(os.path.join(pth, label), label, u16)


def simple_upsample_2D(arr, y=1, x=1):
    """Assumes `arr` is an ndarray with atleast two dimensions, and that the
    spatial dimensions are the final two."""
    n_dims = len(arr.shape)
    return arr.repeat(x, axis=(n_dims - 1)).repeat(y, axis=(n_dims - 2))


def nearest_index(arr, v):
    """Index of value closest to v in ndarray `arr`"""
    return np.abs(arr - v).argmin()


def lead_window(stim_t, stim, stop, duration):
    """Get slice of stimulus stack preceding the the timestamp `stop`, using the
    time axis stim_t to look up the relevant indices."""
    start_idx = nearest_index(stim_t, stop - duration)
    stop_idx = nearest_index(stim_t, stop)
    return stim[start_idx:stop_idx, :, :]


def soft_max(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)


def avg_trigger_window(
    stim_t,
    stim,
    rec_t,
    rec,
    duration,
    trigger_idxs,
    prominences=None,
    max_prominence=None,
):
    """Rough implementation of threshold triggered averaging of a stimulus."""
    times = rec_t[trigger_idxs]
    legal = (times - duration > np.min(stim_t)) * (times <= np.max(stim_t))
    leads = [lead_window(stim_t, stim, t, duration) for t in times[legal]]

    if prominences is None:
        return np.mean(leads, axis=0)
    else:
        if max_prominence is not None:
            proms = np.clip(prominences[legal], 0, max_prominence)
        else:
            proms = prominences[legal]
        return np.sum(leads * soft_max(proms.reshape(-1, 1, 1, 1)), axis=0)


def butter_bandpass(lowcut, highcut, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=3):
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = signal.lfilter(b, a, data)
    return y


def reduce_chunks(arr, chunk_size, reducer=np.sum, axis=-1):
    og_shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    new_shape = og_shape[:axis] + (-1, chunk_size) + og_shape[axis + 1:]
    arr = arr.reshape(new_shape)
    return reducer(arr, axis=(axis + 1))
