from typing import Any, Callable, List
import os
import h5py as h5

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox, Slider


class StackPlotter:
    """Returns Object for cycling through frames of a 3D image stack using the
    mouse scroll wheel. Takes the pyplot axis object and data as the first two
    arguments. Additionally, use delta to set the number of frames each step of
    the wheel skips through."""

    def __init__(
        self,
        fig,
        ax,
        stack,
        delta=10,
        vmin=None,
        vmax=None,
        cmap="gray",
        z_fmt_fun=lambda i: "z = %i" % i,
        dims=None,
    ):
        self.fig = fig
        self.ax = ax
        self.stack: np.ndarray = stack
        self.slices, self.y_sz, self.x_sz = stack.shape
        self.z_fmt_fun = z_fmt_fun
        self.idx = 0
        self.delta = delta
        self.vmin, self.vmax = vmin, vmax

        self.width, self.height = (self.x_sz, self.y_sz) if dims is None else dims
        self.x_frac = self.width / (self.x_sz)
        self.y_frac = self.height / (self.y_sz)
        extent = (0, self.width, self.height, 0)

        self.im = ax.imshow(
            self.stack[self.idx, :, :], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent
        )
        self.update()
        self.connect_scroll()

    def onscroll(self, event):
        if event.button == "up":
            self.idx = (self.idx + self.delta) % self.slices
        else:
            self.idx = (self.idx - self.delta) % self.slices
        self.update()

    def update(self):
        data = self.stack[self.idx, :, :]
        self.im.set_data(data)
        vmin = np.min(data) if self.vmin is None else self.vmin
        vmax = np.max(data) if self.vmax is None else self.vmax
        self.im.set_clim(vmin, vmax)
        self.ax.set_ylabel(self.z_fmt_fun(self.idx))
        self.im.axes.figure.canvas.draw()

    def connect_scroll(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)


class StackFuncPlotter:
    """Returns Object for cycling through frames of a 3D image stack to be fed
    into the provided func responsible to updating the given axis, using the
    mouse scroll wheel. Takes the pyplot axis object and data as the first two
    arguments. Additionally, use delta to set the number of frames each step of
    the wheel skips through."""

    def __init__(
        self,
        fig,
        ax,
        stack,
        func,
        delta=10,
        z_fmt_fun=lambda i: "z = %i" % i,
        dims=None,
    ):
        self.fig = fig
        self.ax = ax
        self.stack: np.ndarray = stack
        self.func = func
        self.slices, self.y_sz, self.x_sz = stack.shape
        self.z_fmt_fun = z_fmt_fun
        self.idx = 0
        self.delta = delta

        self.width, self.height = (self.x_sz, self.y_sz) if dims is None else dims
        self.x_frac = self.width / (self.x_sz)
        self.y_frac = self.height / (self.y_sz)
        self.extent = (0, self.width, self.height, 0)

        self.update()
        self.connect_scroll()

    def onscroll(self, event):
        if event.button == "up":
            self.idx = (self.idx + self.delta) % self.slices
        else:
            self.idx = (self.idx - self.delta) % self.slices
        self.update()

    def update(self):
        self.func(self.ax, self.stack[self.idx, :, :], extent=self.extent)
        self.ax.set_ylabel(self.z_fmt_fun(self.idx))

    def connect_scroll(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)


class ContourStackPlotter:
    """Returns Object for cycling through frames of a 3D image stack to be fed
    into a contour plot. Takes the pyplot axis object and data as the first two
    arguments. Additionally, use delta to set the number of frames each step of
    the wheel skips through."""

    def __init__(
        self,
        fig,
        ax,
        stack,
        levels=20,
        delta=10,
        z_fmt_fun=lambda i: "z = %i" % i,
        dims=None,
        vmin=None,
        vmax=None,
        colors=None,
        cmap=None,
        fill_mode=True,
    ):
        self.fig = fig
        self.ax = ax
        self.stack: np.ndarray = stack
        self.levels = levels
        self.slices, self.y_sz, self.x_sz = stack.shape
        self.z_fmt_fun = z_fmt_fun
        self.idx = 0
        self.delta = delta
        self.vmin, self.vmax = vmin, vmax
        self.cmap, self.colors, self.fill_mode = cmap, colors, fill_mode

        self.width, self.height = (self.x_sz, self.y_sz) if dims is None else dims
        self.x_frac = self.width / (self.x_sz)
        self.y_frac = self.height / (self.y_sz)
        self.extent = (0, self.width, self.height, 0)
        x = np.linspace(0.0, self.width, self.x_sz)
        y = np.linspace(0.0, self.height, self.y_sz)
        self.xs, self.ys = np.meshgrid(x, y)

        self.update()
        self.connect_scroll()

    def onscroll(self, event):
        if event.button == "up":
            self.idx = (self.idx + self.delta) % self.slices
        else:
            self.idx = (self.idx - self.delta) % self.slices
        self.update()

    def update(self):
        z = self.stack[self.idx, :, :]
        self.ax.clear()
        if self.fill_mode:
            self.ax.contourf(
                self.xs, self.ys, z, levels=self.levels, cmap=self.cmap,
            )
            self.ax.invert_yaxis()
        else:
            self.ax.imshow(
                z, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, extent=self.extent
            )
            self.ax.contour(
                self.xs, self.ys, z, levels=self.levels, colors=self.colors,
            )

        self.ax.set_ylabel(self.z_fmt_fun(self.idx))

    def connect_scroll(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)


class MultiStackPlotter:
    """Stacks must all have the same z size, but other dims can be different."""

    def __init__(
        self,
        stacks: List[np.ndarray],
        delta=1,
        vmin: Any = None,
        vmax: Any = None,
        n_cols=2,
        cmap="gray",
        title_fmt_fun=lambda i: "trial = %i" % i,
        idx_fmt_fun=lambda i: "z = %i" % i,
        dims=None,
        **plot_kwargs
    ):
        self.stacks = stacks
        self.n_stacks = len(stacks)
        self.slices = stacks[0].shape[0]
        self.delta, self.n_cols = delta, n_cols
        self.idx_fmt_fun = idx_fmt_fun
        self.n_rows = np.ceil(self.n_stacks / self.n_cols).astype(int)
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

        if dims is None:
            extents = [None for _ in range(self.n_stacks)]
        elif type(dims) == tuple:
            w, h = dims
            extents = [(0.0, w, h, 0.0) for _ in range(self.n_stacks)]
        elif len(dims) == self.n_stacks:
            extents = [(0.0, w, h, 0.0) for w, h in dims]
        else:
            raise ValueError(
                "dims must be tuple or list of tuples with same lenght as stacks."
            )

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
                            extent=extents[i],
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


class MultiWavePlotter:
    def __init__(
        self,
        waves,
        xaxis=None,
        delta=1,
        ymin=None,
        ymax=None,
        n_cols=1,
        title_fmt_fun=lambda i: "trial = %i" % i,
        idx_fmt_fun=lambda i: "z = %i" % i,
        **plot_kwargs
    ):
        self.waves, self.n_waves, self.slices = waves, len(waves), waves[0].shape[0]
        self.delta, self.n_cols = delta, n_cols
        self.idx_fmt_fun = idx_fmt_fun
        self.n_rows = np.ceil(self.n_waves / self.n_cols).astype(int)
        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, **plot_kwargs)
        self.idx = 0
        self.lines = {i: [] for i in range(self.n_waves)}

        if ymin is not None and type(ymin) != "dict":
            ymin = {"default": ymin}
        elif type(ymin) == "dict" and "default" not in ymin:
            ymin["default"] = None
        if ymax is not None and type(ymax) != "dict":
            ymax = {"default": ymax}
        elif type(ymax) == "dict" and "default" not in ymax:
            ymax["default"] = None

        i = 0
        for row in self.ax:
            row = row if type(row) == list else [row]
            for a in row:
                if i < self.n_waves:
                    for w in self.waves[i][self.idx]:
                        if xaxis is None:
                            self.lines[i].append(a.plot(w)[0])
                        else:
                            self.lines[i].append(a.plot(xaxis, w)[0])
                    a.set_title(title_fmt_fun(i))
                    a.set_ylim(
                        ymin.get(i, ymin["default"]), ymax.get(i, ymax["default"])
                    )
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
        for ws, ls in zip(self.waves, self.lines.values()):
            for w, l in zip(ws[self.idx], ls):
                l.set_ydata(w)
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
        auto_roi_scale=False,
        cmap="gray",
        z_fmt_fun=lambda i: "z = %i" % i,
        n_fmt_fun=None,
        trial_fmt_fun=None,
        dims=None,
        **plot_kwargs
    ):
        self.z_fmt_fun = z_fmt_fun
        self.auto_roi_scale = auto_roi_scale

        vmin = stack.min() if vmin is None else vmin
        vmax = stack.max() if vmax is None else vmax

        if stack.ndim > 3:
            if stack.ndim == 5 and stack.shape[0] > 1:
                self.ns = True
                self.trials = stack.shape[1] > 1
            elif stack.shape[0] > 1:
                self.trials = True
                self.ns = False
            else:
                self.ns, self.trials = False, False

            if stack.ndim < 5:
                stack = stack.reshape(1, *stack.shape)

            if self.trials:
                self.avg = np.mean(stack, axis=1, keepdims=True)
                stack = np.concatenate([stack, self.avg], axis=1)
                self.trials = True

            if self.ns or self.trials:
                if "gridspec_kw" not in plot_kwargs:
                    if self.ns and self.trials:
                        plot_kwargs["gridspec_kw"] = {
                            "height_ratios": [0.64, 0.03, 0.03, 0.3]
                        }
                    else:
                        plot_kwargs["gridspec_kw"] = {
                            "height_ratios": [0.67, 0.03, 0.3]
                        }
                self.fig, self.ax = plt.subplots(
                    2 + self.ns + self.trials, **plot_kwargs
                )

                if self.ns and self.trials:
                    (
                        self.stack_ax,
                        self.n_slide_ax,
                        self.trial_slide_ax,
                        self.beam_ax,
                    ) = self.ax
                elif self.ns:
                    self.stack_ax, self.n_slide_ax, self.beam_ax = self.ax
                else:
                    self.stack_ax, self.trial_slide_ax, self.beam_ax = self.ax

                self.n_fmt_fun = (
                    (lambda i: "N %i" % i) if n_fmt_fun is None else n_fmt_fun
                )
                self.trial_fmt_fun = (
                    (lambda i: "trial %i" % i)
                    if trial_fmt_fun is None
                    else trial_fmt_fun
                )
            else:
                if "gridspec_kw" not in plot_kwargs:
                    plot_kwargs["gridspec_kw"] = {"height_ratios": [0.7, 0.3]}
                self.fig, self.ax = plt.subplots(2, **plot_kwargs)
                self.stack_ax, self.beam_ax = self.ax
        else:
            self.ns, self.trials = False, False
            stack = stack.reshape(1, 1, *stack.shape)
            if "gridspec_kw" not in plot_kwargs:
                plot_kwargs["gridspec_kw"] = {"height_ratios": [0.7, 0.3]}
            self.fig, self.ax = plt.subplots(2, **plot_kwargs)
            self.stack_ax, self.beam_ax = self.ax

        self.stack, self.delta = stack, delta
        self.n_sz, self.tr_sz, self.z_sz, self.y_sz, self.x_sz = stack.shape
        self.n_idx, self.tr_idx, self.z_idx, self.roi_x, self.roi_y = 0, 0, 0, 0, 0
        self.zaxis = np.arange(self.z_sz) if zaxis is None else zaxis
        self.roi_locked = False

        self.width, self.height = (self.x_sz, self.y_sz) if dims is None else dims
        self.x_frac = self.width / (self.x_sz)
        self.y_frac = self.height / (self.y_sz)
        extent = (0, self.width, self.height, 0)
        self.roi_x_sz, self.roi_y_sz = (
            roi_sz if type(roi_sz) == tuple else (roi_sz, roi_sz)
        )

        self.build_stack_ax(cmap, vmin, vmax, extent)
        self.build_roi_ax(vmin, vmax)
        if self.ns:
            self.build_n_slide_ax()
        if self.trials:
            self.build_trial_slide_ax()
        self.fig.tight_layout(h_pad=0)
        self.connect_events()

    def build_n_slide_ax(self):
        self.n_slider = Slider(
            self.n_slide_ax,
            "",
            valmin=0,
            valmax=(self.n_sz - 1),
            valinit=0,
            valstep=1,
            valfmt="%.0f",
        )
        self.n_slide_ax.set_title(self.n_fmt_fun(0))

    def build_trial_slide_ax(self):
        self.trial_slider = Slider(
            self.trial_slide_ax,
            "",
            valmin=0,
            valmax=(self.tr_sz - 1),
            valinit=0,
            valstep=1,
            valfmt="%.0f",
        )
        self.trial_slide_ax.set_title(self.trial_fmt_fun(0))

    def build_stack_ax(self, cmap, vmin, vmax, extent):
        self.im = self.stack_ax.imshow(
            self.stack[self.n_idx, self.tr_idx, self.z_idx, :, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        self.roi_rect = Rectangle(
            (0.0, 0.0),
            self.roi_x_sz * self.x_frac,
            self.roi_y_sz * self.y_frac,
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
            beams[self.tr_idx, self.z_idx],
            marker="x",
            c="black",
            markersize=12,
        )[0]
        self.beam_ax.set_ylim(vmin, vmax)
        self.update_roi()

    def update_beams(self):
        self.beams = np.mean(
            self.stack[
                self.n_idx,
                :,
                :,
                self.roi_y : self.roi_y + self.roi_y_sz,
                self.roi_x : self.roi_x + self.roi_x_sz,
            ],
            axis=(2, 3),
        )
        return self.beams

    def on_n_slide(self, v):
        self.n_idx = int(v)
        self.n_slide_ax.set_title(self.n_fmt_fun(self.n_idx))
        self.update_im()
        self.update_roi()

    def on_trial_slide(self, v):
        self.tr_idx = int(v)
        self.trial_slide_ax.set_title(
            self.trial_fmt_fun(self.tr_idx)
            if self.tr_idx < self.tr_sz - 1
            else "average"
        )
        self.update_im()
        self.update_roi()

    def on_scroll(self, event):
        if event.button == "up":
            self.z_idx = (self.z_idx + self.delta) % self.z_sz
        else:
            self.z_idx = (self.z_idx - self.delta) % self.z_sz
        self.z_marker.set_data(
            self.zaxis[self.z_idx], self.beams[self.tr_idx, self.z_idx]
        )
        self.update_im()

    def on_move(self, event):
        if (
            not self.roi_locked
            and event.inaxes == self.stack_ax
            and not (event.xdata is None or event.ydata is None)
        ):
            x = np.floor(event.xdata / self.x_frac).astype(int)
            y = np.floor(event.ydata / self.y_frac).astype(int)
            if (
                0 <= x <= (self.x_sz - self.roi_x_sz)
                and 0 <= y <= (self.y_sz - self.roi_y_sz)
                and (self.roi_x != x or self.roi_y != y)
            ):
                self.roi_x, self.roi_y = x, y
                self.roi_rect.set_xy((x * self.x_frac, y * self.y_frac))
                self.update_roi()

    def on_im_click(self, event):
        if event.button == 1 and event.inaxes == self.stack_ax:
            self.roi_locked = False if self.roi_locked else True
            self.update_roi()

    def update_im(self):
        self.im.set_data(self.stack[self.n_idx, self.tr_idx, self.z_idx, :, :])
        # self.stack_ax.set_ylabel("z = %s" % self.z_idx)
        self.stack_ax.set_ylabel(self.z_fmt_fun(self.z_idx))
        self.im.axes.figure.canvas.draw()

    def update_roi(self):
        msg = "(click to %s)" % ("unlock" if self.roi_locked else "lock")
        self.beam_ax.set_title("x = %i; y = %i; %s" % (self.roi_x, self.roi_y, msg))
        beams = self.update_beams()
        if self.auto_roi_scale:
            self.beam_ax.set_ylim(beams.min(), beams.max())
        for i, line in enumerate(self.roi_lines):
            line.set_ydata(beams[i])
            line.set_color("red" if i == self.tr_idx else "black")
            line.set_alpha(1 if i == self.tr_idx else 0.75)
        self.z_marker.set_data(self.zaxis[self.z_idx], beams[self.tr_idx, self.z_idx])

    def connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_im_click)
        if self.ns:
            self.n_slider.on_changed(self.on_n_slide)
        if self.trials:
            self.trial_slider.on_changed(self.on_trial_slide)


class PeakExplorer:
    def __init__(
        self,
        xaxis,
        recs,
        threshold=0,
        prominence=1,
        width=2,
        tolerance=0.5,
        wlen=None,
        distance=1,
        title_fmt_fun=lambda i: "roi = %i" % i,
        auto_y_scale=True,
    ):
        self.xaxis, self.recs = xaxis, recs
        self.n_rois, self.pts = recs.shape
        self.threshold, self.prominence, self.width = threshold, prominence, width
        self.tolerance, self.wlen, self.distance = tolerance, wlen, distance
        self.auto_y_scale = auto_y_scale
        self.title_fmt_fun = title_fmt_fun

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
        gs = self.fig.add_gridspec(nrows=3, ncols=3, height_ratios=[0.8, 0.1, 0.1])
        self.rec_ax = self.fig.add_subplot(gs[0, :])
        self.thresh_ax = self.fig.add_subplot(gs[1, 0])
        self.prom_ax = self.fig.add_subplot(gs[1, 1])
        self.width_ax = self.fig.add_subplot(gs[1, 2])
        self.toler_ax = self.fig.add_subplot(gs[2, 0])
        self.wlen_ax = self.fig.add_subplot(gs[2, 1])
        self.dist_ax = self.fig.add_subplot(gs[2, 2])
        self.rec_ax.set_title(self.title_fmt_fun(0))
        self.rec_ax.set_xlabel("Time (s)")
        self.thresh_ax.set_title("Threshold")
        self.prom_ax.set_title("Prominence")
        self.width_ax.set_title("Width")
        self.toler_ax.set_title("Tolerance")
        self.wlen_ax.set_title("Window Size")
        self.dist_ax.set_title("Min Interval")

    def build_inputs(self):
        self.thresh_box = TextBox(self.thresh_ax, "", initial=str(self.threshold))
        self.prom_box = TextBox(self.prom_ax, "", initial=str(self.prominence))
        self.width_box = TextBox(self.width_ax, "", initial=str(self.width))
        self.toler_box = TextBox(self.toler_ax, "", initial=str(self.tolerance))
        self.wlen_box = TextBox(self.wlen_ax, "", initial=str(self.wlen))
        self.dist_box = TextBox(self.dist_ax, "", initial=str(self.distance))

    def set_threshold(self, s):
        try:
            self.threshold = float(s)
            self.refresh()
        except:
            self.thresh_box.set_val(str(self.threshold))

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

    def set_wlen(self, s):
        try:
            self.wlen = int(s)
            self.refresh()
        except:
            self.wlen = None
            self.wlen_box.set_val(str(None))

    def update_peaks(self):
        self.peaks, _ = signal.find_peaks(
            self.recs[self.idx],
            threshold=self.threshold,
            prominence=self.prominence,
            rel_height=self.tolerance,
            width=self.width,
            wlen=self.wlen,
            distance=self.distance,
        )

    def update_view(self):
        self.rec_ax.set_title(self.title_fmt_fun(self.idx))
        rec = self.recs[self.idx]
        self.rec_line.set_ydata(rec)
        self.peak_line.set_data(self.xaxis[self.peaks], rec[self.peaks])
        if self.auto_y_scale:
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
        self.thresh_box.on_submit(self.set_threshold)
        self.prom_box.on_submit(self.set_prominence)
        self.width_box.on_submit(self.set_width)
        self.toler_box.on_submit(self.set_tolerance)
        self.wlen_box.on_submit(self.set_wlen)
        self.dist_box.on_submit(self.set_distance)


def plot_stack(arr, delta=10, vmin=None, vmax=None, cmap="gray"):
    fig, ax = plt.subplots(1)
    stack = StackPlotter(fig, ax, arr, delta, vmin, vmax, cmap)
    fig.tight_layout()
    plt.show()
    return stack


def remove_offset(arr):
    return arr - arr.min()


def array_to_h5(pth, fname, arr, time_ax=0):
    if time_ax != 0:
        arr = np.moveaxis(arr, time_ax, 0)

    os.makedirs(pth, exist_ok=True)
    with h5.File(os.path.join(pth, fname) + ".h5", "w") as f:
        f.create_dataset("data", data=arr)


def simple_upsample_2D(arr, y=1, x=1):
    """Assumes `arr` is an ndarray with atleast two dimensions, and that the
    spatial dimensions are the final two."""
    n_dims = len(arr.shape)
    return arr.repeat(x, axis=(n_dims - 1)).repeat(y, axis=(n_dims - 2))


def nearest_index(arr, v):
    """Index of value closest to v in ndarray `arr`"""
    return np.abs(arr - v).argmin()


def lead_window(stim_t, stim, stop, n_frames):
    """Get slice of length `n_frames` from the stimulus stack preceding the the
    timestamp `stop`, using the time axis `stim_t` to look up the relevant index."""
    stop_idx = nearest_index(stim_t, stop)
    return stim[(stop_idx - n_frames) : stop_idx, :, :]


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
    lead_time,
    post_time,
    trigger_idxs,
    prominences=None,
    max_prominence=None,
    nonlinear_weighting=True,
    start_time=None,
    end_time=None,
):
    """Rough implementation of threshold triggered averaging of a stimulus."""
    duration = lead_time + post_time
    n_frames = nearest_index(stim_t, np.min(stim_t) + duration)
    times = rec_t[np.array(trigger_idxs)]
    post_shift = times + post_time
    start_time = np.min(stim_t) if start_time is None else start_time
    end_time = np.max(stim_t) if end_time is None else end_time
    legal = ((post_shift - duration > start_time) * (post_shift <= end_time)).astype(
        bool
    )
    post_shift = post_shift[legal]
    window = np.zeros((n_frames, stim.shape[1], stim.shape[2]))

    if len(post_shift) == 0:
        return window, []

    if prominences is None:
        weights = np.ones(len(post_shift)) / len(post_shift)
    else:
        if max_prominence is not None:
            weights = np.clip(prominences[legal], 0, max_prominence)
        else:
            weights = prominences[legal]

        if nonlinear_weighting:
            weights = soft_max(weights / np.max(weights))  # prevent overflow
        else:
            weights = weights / np.sum(weights)

    for t, w in zip(post_shift, weights):
        window += lead_window(stim_t, stim, t, n_frames) * w

    return window, times[legal]


def trigger_xaxis(stim_t, lead_time, post_time):
    duration = lead_time + post_time
    stim_dt = (np.max(stim_t) - np.min(stim_t)) / (len(stim_t - 1))
    n_frames = nearest_index(stim_t, np.min(stim_t) + duration)
    return np.arange(n_frames) * stim_dt - lead_time


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
    new_shape = og_shape[:axis] + (-1, chunk_size) + og_shape[axis + 1 :]
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


def quality_index(arr):
    """Variance of the mean of trials over the mean of the variance of each
    individual trial. This gives an index of the quality of the signal by
    measuring how much of the variance in each trial explained by signals
    present across all trials."""
    return np.var(np.mean(arr, axis=0)) / np.mean(np.var(arr, axis=1))


def upscale(arr, factor, axes=[0, 1]):
    return arr.repeat(factor, axis=axes[0]).repeat(factor, axis=axes[1])


def map_axis(f: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, axis=-1):
    """Map the specified axis (defaults to -1) with the function `f`. `f` should
     thus expect an ndarray with the shape created by `axis` and all of its
     'sub-axes'. The shape returned by `f` may differ from the original shape of
    its input.

    For example:
    - `arr` of shape (2, 5) with `axis` = -1. f operates on a 1d array of length 5.
    - `arr` of shape (2, 5, 5) with `axis` = 1. f operates on an array of shape (5, 5).
    """
    in_shape = arr.shape
    if len(in_shape) > 1:
        reshaped = arr.reshape(np.prod(in_shape[:axis]).astype(int), *in_shape[axis:])
        mapped: np.ndarray = np.stack([f(a) for a in reshaped], axis=0)

        if len(mapped.shape) == 1:
            return mapped.reshape(in_shape[:axis])
        else:
            return mapped.reshape(*in_shape[:axis], *mapped.shape[1:])
    else:
        return f(arr)


def moving_average(arr: np.ndarray, n=3, axis=-1) -> np.ndarray:
    """No padding moving average."""

    def f(a):
        cs = np.cumsum(a)
        cs[n:] = cs[n:] - cs[:-n]
        return cs[n - 1 :] / n

    return map_axis(f, arr, axis=axis)


def rolling_average(arr: np.ndarray, n=3, axis=-1) -> np.ndarray:
    """Same as moving average, but with convolve and same padding."""
    ones = np.ones(n)
    return map_axis(lambda a: np.convolve(a, ones, "same"), arr, axis=axis) / n


def butter_lowpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return map_axis(lambda d: signal.lfilter(b, a, d), data)
