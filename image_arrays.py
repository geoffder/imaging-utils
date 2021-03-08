import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox, Slider
from skimage import io
from scipy import signal
from PIL import Image
from tifffile import imsave
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
        self.ax.set_ylabel("z = %s" % self.idx)
        self.im.axes.figure.canvas.draw()

    def connect_scroll(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)


class MultiStackPlotter:
    """Stacks must all have the same z size, but other dims can be different."""
    def __init__(
        self,
        stacks,
        delta=1,
        vmin=None,
        vmax=None,
        n_cols=2,
        cmap="gray",
        title_fmt_fun=lambda i: "trial = %i" % i,
        idx_fmt_fun=lambda i: "z = %i" % i,
        **plot_kwargs
    ):
        self.stacks, self.n_stacks, self.slices = stacks, len(stacks), stacks[0].shape[0]
        self.delta, self.n_cols = delta, n_cols
        self.idx_fmt_fun = idx_fmt_fun
        self.n_rows = np.ceil(self.n_stacks / self.n_cols).astype(np.int)
        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, **plot_kwargs)
        self.idx = 0
        self.plots = []

        if vmin is not None and type(vmin) != "dict":
            vmin = {"default": vmin}
        elif type(vmin) == "dict" and "default" not in vmin:
            vmin["default"] = None
        if vmax is not None and type(vmax) != "dict":
            vmax = {"default": vmax}
        elif type(vmax) == "dict" and "default" not in vmax:
            vmax["default"] = None

        i = 0
        for row in self.ax:
            for a in row:
                if i < self.n_stacks:
                    self.plots.append(
                        a.imshow(
                            self.stacks[i][self.idx, :, :],
                            cmap=cmap,
                            vmin=vmin.get(i, vmin["default"]),
                            vmax=vmax.get(i, vmax["default"]),
                        )
                    )
                    a.set_title(title_fmt_fun(i))
                else:
                    a.set_visible(False)
                i += 1
        self.update()
        self.connect_scroll()

    def onscroll(self, event):
        if event.button == "up":
            self.idx = (self.idx + self.delta) % self.slices
        else:
            self.idx = (self.idx - self.delta) % self.slices
        self.update()

    def update(self):
        for s, p in zip(self.stacks, self.plots):
            p.set_data(s[self.idx, :, :])
        self.fig.suptitle(self.idx_fmt_fun(self.idx))

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
        if stack.ndim == 4 and stack.shape[0] > 1:
            self.trials = True
            self.avg = np.mean(stack, axis=0)
            stack = np.concatenate([stack, self.avg.reshape(1, *self.avg.shape)], axis=0)
            if "gridspec_kw" not in plot_kwargs:
                plot_kwargs["gridspec_kw"] = {"height_ratios": [.67, .03, .3]}
            self.fig, self.ax = plt.subplots(3, **plot_kwargs)
            self.stack_ax, self.slide_ax, self.beam_ax = self.ax
        else:
            self.trials = False
            stack = stack.reshape(1, *stack.shape)
            if "gridspec_kw" not in plot_kwargs:
                plot_kwargs["gridspec_kw"] = {"height_ratios": [.7, .3]}
            self.fig, self.ax = plt.subplots(2, **plot_kwargs)
            self.stack_ax, self.beam_ax = self.ax

        self.stack, self.delta, self.roi_sz = stack, delta, roi_sz
        self.n_sz, self.z_sz, self.y_sz, self.x_sz = stack.shape
        self.n_idx, self.z_idx, self.roi_x, self.roi_y = 0, 0, 0, 0
        self.zaxis = np.arange(self.z_sz) if zaxis is None else zaxis
        self.roi_locked = False

        self.build_stack_ax(cmap, vmin, vmax)
        self.build_roi_ax(vmin, vmax)
        if self.trials:
            self.build_slide_ax()
        self.fig.tight_layout(h_pad=0)
        self.connect_events()

    def build_slide_ax(self):
        self.slider = Slider(
            self.slide_ax,
            "",
            valmin=0,
            valmax=(self.n_sz - 1),
            valinit=0,
            valstep=1,
            valfmt="%.0f"
        )
        self.slide_ax.set_title("trial 0")

    def build_stack_ax(self, cmap, vmin, vmax):
        self.im = self.stack_ax.imshow(
            self.stack[self.n_idx, self.z_idx, :, :], cmap=cmap, vmin=vmin, vmax=vmax
        )
        self.roi_rect = Rectangle(
            (self.roi_x - .5, self.roi_y - .5),
            self.roi_sz,
            self.roi_sz,
            fill=False,
            color="red",
            linewidth=2,
        )
        self.stack_ax.add_patch(self.roi_rect)
        self.update_im()

    def build_roi_ax(self, vmin, vmax):
        beams = self.update_beams()
        self.roi_lines = [self.beam_ax.plot(self.zaxis, b)[0] for b in beams]
        if self.trials:
            self.roi_lines[-1].set_linewidth(3)  # avg line thicker
        self.z_marker = self.beam_ax.plot(
            self.zaxis[self.z_idx],
            beams[self.n_idx, self.z_idx],
            marker="x",
            c="black",
            markersize=12
        )[0]
        self.beam_ax.set_ylim(vmin, vmax)
        self.update_roi()

    def update_beams(self):
        if self.roi_sz > 1:
            self.beams = np.mean(
                self.stack[:, :, self.roi_y:self.roi_y + self.roi_sz,
                           self.roi_x:self.roi_x + self.roi_sz],
                axis=(2, 3)
            )
        else:
            self.beams = self.stack[:, :, self.roi_y, self.roi_x]
        return self.beams

    def on_slide(self, v):
        self.n_idx = int(v)
        self.slide_ax.set_title(
            "trial %i" % self.n_idx if self.n_idx < self.n_sz - 1 else "average"
        )
        self.update_im()
        self.update_roi()

    def on_scroll(self, event):
        if event.button == "up":
            self.z_idx = (self.z_idx + self.delta) % self.z_sz
        else:
            self.z_idx = (self.z_idx - self.delta) % self.z_sz
        self.z_marker.set_data(self.zaxis[self.z_idx], self.beams[self.n_idx, self.z_idx])
        self.update_im()

    def on_move(self, event):
        if (
            not self.roi_locked and event.inaxes == self.stack_ax and
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
        if (event.button == 1 and event.inaxes == self.stack_ax):
            self.roi_locked = False if self.roi_locked else True
            self.update_roi()

    def update_im(self):
        self.im.set_data(self.stack[self.n_idx, self.z_idx, :, :])
        self.stack_ax.set_ylabel("z = %s" % self.z_idx)
        self.im.axes.figure.canvas.draw()

    def update_roi(self):
        msg = "(click to %s)" % ("unlock" if self.roi_locked else "lock")
        self.beam_ax.set_title("x = %i; y = %i; %s" % (self.roi_x, self.roi_y, msg))
        beams = self.update_beams()
        for i, line in enumerate(self.roi_lines):
            line.set_ydata(beams[i])
            line.set_color("red" if i == self.n_idx else "black")
            line.set_alpha(1 if i == self.n_idx else 0.75)
        self.z_marker.set_data(self.zaxis[self.z_idx], beams[self.n_idx, self.z_idx])

    def connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_im_click)
        if self.trials:
            self.slider.on_changed(self.on_slide)


class PeakExplorer:
    def __init__(self, xaxis, recs, prominence=1, width=2, tolerance=.5, distance=1):
        self.xaxis, self.recs = xaxis, recs
        self.n_rois, self.pts = recs.shape
        self.prominence, self.width = prominence, width
        self.tolerance, self.distance = tolerance, distance

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
        self.prom_ax = self.fig.add_subplot(gs[1, 0])
        self.width_ax = self.fig.add_subplot(gs[1, 1])
        self.toler_ax = self.fig.add_subplot(gs[2, 0])
        self.dist_ax = self.fig.add_subplot(gs[2, 1])
        self.rec_ax.set_title("roi = 0")
        self.rec_ax.set_xlabel("Time (s)")
        self.prom_ax.set_title("Peak Prominence")
        self.width_ax.set_title("Peak Width")
        self.toler_ax.set_title("Peak Tolerance")
        self.dist_ax.set_title("Min Distance Between")

    def build_inputs(self):
        self.prom_box = TextBox(self.prom_ax, "", initial=str(self.prominence))
        self.width_box = TextBox(self.width_ax, "", initial=str(self.width))
        self.toler_box = TextBox(self.toler_ax, "", initial=str(self.tolerance))
        self.dist_box = TextBox(self.dist_ax, "", initial=str(self.distance))

    def set_prominence(self, s):
        try:
            self.prominence = float(s)
            self.refresh()
        except:
            self.prom_box.set_val(str(self.prominence))

    def set_width(self, s):
        try:
            self.width = int(s)
            self.refresh()
        except:
            self.width_box.set_val(str(self.width))

    def set_tolerance(self, s):
        try:
            tolerance = float(s)
            if 0 <= tolerance <= 1:
                self.tolerance = tolerance
                self.refresh()
            else:
                raise (ValueError("Tolerance must be between 0 and 1."))
        except ValueError:
            self.toler_box.set_val(str(self.tolerance))

    def set_distance(self, s):
        try:
            dist = int(s)
            if dist >= 1:
                self.distance = dist
                self.refresh()
            else:
                raise (ValueError("Minimum peak distance must be >= 1"))
        except ValueError:
            self.dist_box.set_val(str(self.distance))

    def update_peaks(self):
        self.peaks, _ = signal.find_peaks(
            self.recs[self.idx],
            prominence=self.prominence,
            rel_height=self.tolerance,
            width=self.width,
            distance=self.distance
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
        self.prom_box.on_submit(self.set_prominence)
        self.width_box.on_submit(self.set_width)
        self.toler_box.on_submit(self.set_tolerance)
        self.dist_box.on_submit(self.set_distance)


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


def soft_min(x):
    ex = np.exp(-x)
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
    nonlinear_weighting=True,
    start_time=None,
    end_time=None,
):
    """Rough implementation of threshold triggered averaging of a stimulus."""
    times = rec_t[trigger_idxs]
    start_time = np.min(stim_t) if start_time is None else start_time
    end_time = np.max(stim_t) if end_time is None else end_time
    legal = (times - duration > start_time) * (times <= end_time)
    leads = [lead_window(stim_t, stim, t, duration) for t in times[legal]]

    if prominences is None:
        window = np.mean(leads, axis=0)
    else:
        if max_prominence is not None:
            proms = np.clip(prominences[legal], 0, max_prominence)
        else:
            proms = prominences[legal]

        if nonlinear_weighting:
            window = np.sum(leads * soft_max(proms.reshape(-1, 1, 1, 1)), axis=0)
        else:
            window = np.sum(leads * (proms / np.sum(proms)).reshape(-1, 1, 1, 1), axis=0)

    return window, legal


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


def find_peaks(arr, *args, **kwargs):
    """Basic wrapper over scipy.signal.find_peaks adding flexibility on the
    shape of arr (could have multiple trials on the 0th axis). Currently only
    taking prominence information from the properties dict. Returns lists since
    the resulting arrays will be ragged between trials."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    peak_idxs, peak_proms = [], []
    for x in arr:
        idxs, props = signal.find_peaks(x, *args, **kwargs)
        peak_idxs.append(idxs)
        peak_proms.append(props["prominences"])

    return peak_idxs, peak_proms


def moving_avg(arr, width):
    return np.convolve(arr, np.ones(width) / width, "same")
