from typing import Any, Callable, List
import os
import h5py as h5
from hdf_utils import pack_dataset

import numpy as np
import bottleneck as bn
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox, Slider
from matplotlib.backend_bases import MouseButton


class ROIExplorer:
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
                self.avg = np.expand_dims(bn.nanmean(stack, axis=1), 1)
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
        self.rois = {}
        self.add_roi()

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

    def add_roi(
        self,
    ):
        roi = {"x": 0, "y": 0}
        roi["rect"] = Rectangle(  # type: ignore
            (0.0, 0.0),
            self.roi_x_sz * self.x_frac,
            self.roi_y_sz * self.y_frac,
            fill=False,
            color="red",
            linewidth=2,
        )
        self.stack_ax.add_patch(roi["rect"])
        beams = self.update_beams(roi)
        roi["lines"] = [self.beam_ax.plot(self.zaxis, b)[0] for b in beams]  # type: ignore
        if self.trials:
            roi["lines"][-1].set_linewidth(3)  # avg line thicker
        if len(self.rois):
            self.rois[len(self.rois)] = self.rois[0]
        self.rois[0] = roi  # new roi is the active one
        return roi

    def remove_roi(self, idx):
        roi = self.rois[idx]
        roi["rect"].remove()
        for l in roi["lines"]:
            self.beam_ax.lines.remove(l)

    def build_stack_ax(self, cmap, vmin, vmax, extent):
        self.im = self.stack_ax.imshow(
            self.stack[self.n_idx, self.tr_idx, self.z_idx, :, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        self.update_im()

    def build_roi_ax(self, vmin, vmax):
        self.z_marker = self.beam_ax.plot(
            self.zaxis[self.z_idx],
            self.rois[0]["beams"][self.tr_idx, self.z_idx],
            marker="x",
            c="black",
            markersize=12,
        )[0]
        self.beam_ax.set_ylim(vmin, vmax)
        self.update_roi()

    def update_beams(self, roi):
        roi["beams"] = bn.nanmean(
            bn.nanmean(
                self.stack[
                    self.n_idx,
                    :,
                    :,
                    roi["y"] : roi["y"] + self.roi_y_sz,
                    roi["x"] : roi["x"] + self.roi_x_sz,
                ],
                axis=2,
            ),
            axis=2,
        )
        return roi["beams"]

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
            self.zaxis[self.z_idx], self.rois[0]["beams"][self.tr_idx, self.z_idx]
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
                and (self.rois[0]["x"] != x or self.rois[0]["y"] != y)
            ):
                self.rois[0]["x"] = x
                self.rois[0]["y"] = y
                self.rois[0]["rect"].set_xy((x * self.x_frac, y * self.y_frac))
                self.update_roi()

    def in_roi(self, roi, x, y):
        return (
            x > roi["x"]
            and x < roi["x"] + self.roi_x_sz
            and y > roi["y"]
            and y < roi["y"] + self.roi_y_sz
        )

    def on_im_click(self, event):
        if event.inaxes == self.stack_ax:
            if event.button == MouseButton.LEFT:
                self.roi_locked = False if self.roi_locked else True
                self.update_roi()
            elif event.button == MouseButton.RIGHT:
                x = np.floor(event.xdata / self.x_frac).astype(int)
                y = np.floor(event.ydata / self.y_frac).astype(int)
                new = True
                for i in range(1, len(self.rois)):
                    if self.in_roi(self.rois[i], x, y):
                        self.remove_roi(i)
                        new = False
                if new:
                    self.add_roi()

    def on_beam_click(self, event):
        if event.button == MouseButton.LEFT and event.inaxes == self.beam_ax:
            self.z_idx = nearest_index(self.zaxis, event.xdata)
            self.z_marker.set_data(
                self.zaxis[self.z_idx], self.rois[0]["beams"][self.tr_idx, self.z_idx]
            )
            self.update_im()

    def update_im(self):
        self.im.set_data(self.stack[self.n_idx, self.tr_idx, self.z_idx, :, :])
        self.stack_ax.set_ylabel(self.z_fmt_fun(self.z_idx))
        self.im.axes.figure.canvas.draw()

    def update_roi(self):
        msg = "(click to %s)" % ("unlock" if self.roi_locked else "lock")
        self.beam_ax.set_title("x = %i; y = %i; %s" % (self.roi_x, self.roi_y, msg))
        beams = [self.update_beams(r) for r in self.rois.values()]
        if self.auto_roi_scale:
            mn = min(*[b.min() for b in beams])
            mx = max(*[b.max() for b in beams])
            self.beam_ax.set_ylim(mn, mx)
        for roi in self.rois.values():
            for i, line in enumerate(roi["lines"]):
                line.set_ydata(roi["beams"][i])
                line.set_color("red" if i == self.tr_idx else "black")
                line.set_alpha(1 if i == self.tr_idx else 0.75)
        self.z_marker.set_data(
            self.zaxis[self.z_idx], beams[0][self.tr_idx, self.z_idx]
        )

    def connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_im_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_beam_click)
        if self.ns:
            self.n_slider.on_changed(self.on_n_slide)
        if self.trials:
            self.trial_slider.on_changed(self.on_trial_slide)
