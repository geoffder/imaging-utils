from typing import Callable
import os
import h5py as h5
from hdf_utils import pack_dataset

import numpy as np
import bottleneck as bn
from scipy import signal


def nearest_index(arr, v):
    """Index of value closest to v in ndarray `arr`"""
    return bn.nanargmin(np.abs(arr - v))


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


def avg_lead_from_times(
    stim_t,
    stim,
    times,
    n_frames,
    prominences=None,
    clip_prominence=None,
    nonlinear_weighting=True,
):
    window = np.zeros((n_frames, stim.shape[1], stim.shape[2]))

    if len(times) == 0:
        return window

    if prominences is None:
        weights = np.ones(len(times)) / len(times)
    else:
        if clip_prominence is not None:
            weights = np.clip(prominences, 0, clip_prominence)
        else:
            weights = prominences

        if nonlinear_weighting:
            weights = soft_max(weights / np.max(weights))  # prevent overflow
        else:
            weights = weights / np.sum(weights)

    for t, w in zip(times, weights):
        window += lead_window(stim_t, stim, t, n_frames) * w

    return window


def avg_trigger_window(
    stim_t,
    stim,
    rec_t,
    lead_time,
    post_time,
    trigger_idxs,
    prominences=None,
    clip_prominence=None,
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
        return window, [], None

    if prominences is None:
        weights = np.ones(len(post_shift)) / len(post_shift)
    else:
        prominences = prominences[legal]
        if clip_prominence is not None:
            weights = np.clip(prominences, 0, clip_prominence)
        else:
            weights = prominences

        if nonlinear_weighting:
            weights = soft_max(weights / np.max(weights))  # prevent overflow
        else:
            weights = weights / np.sum(weights)

    for t, w in zip(post_shift, weights):
        window += lead_window(stim_t, stim, t, n_frames) * w

    return window, times[legal], prominences


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
    return bn.nanvar(bn.nanmean(arr, axis=0)) / bn.nanmean(bn.nanvar(arr, axis=1))


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


def triggered_leads(
    noise_xaxis,
    raw_noise,
    recs_xaxis,
    recs,
    grid_idxs,
    lead_time=5.0,  # length of triggered average movie (seconds before peak)
    post_time=2.0,
    peak_threshold=0,  # difference between peaks and their adjacent points
    prominence=0.75,  # difference between peaks and their surrounding troughs
    peak_width=2,  # minimum number of points (within tolerance)
    peak_tolerance=0.4,  # ratio value can drop from peak within width
    window_size=60,  # length of the window used to calculate prominence
    min_peak_interval=1,  # number of points required between peaks
    max_prominence=None,  # drop prominences above the max
    clip_prominence=None,  # clip to avoid dominance by errant peaks
    weighting="non-linear",
    start_time=10,  # time to begin using peaks for triggered average
    end_time=None,  # cutoff time for considering peaks
    min_peak_count=20,  # ROIs with fewer peaks are thrown out
    workspace=None,
):
    legal_times, legal_proms = [], []
    count, pos_to_roi, roi_to_pos = 0, [], {}
    pos_to_grid_idx = []

    start_time = bn.nanmin(noise_xaxis) if start_time is None else start_time
    end_time = bn.nanmax(noise_xaxis) if end_time is None else end_time
    duration = lead_time + post_time
    n_frames = nearest_index(noise_xaxis, np.min(noise_xaxis) + duration)

    for i in range(recs.shape[1]):  # roi
        trial_times, trial_proms = [], []
        for j in range(recs.shape[0]):  # trial
            peak_idxs, peak_proms = find_peaks(
                recs[j, i],
                threshold=peak_threshold,
                prominence=prominence,
                width=peak_width,
                wlen=window_size,
                rel_height=peak_tolerance,
                distance=min_peak_interval,
            )
            if max_prominence is not None and len(peak_idxs[0]) > 0:
                peak_idxs = np.array(
                    [
                        i
                        for i, p in zip(peak_idxs[0], peak_proms[0])
                        if p < max_prominence
                    ]
                )
                peak_proms = np.array([p for p in peak_proms[0] if p < max_prominence])

            else:
                peak_idxs = peak_idxs[0]
                peak_proms = peak_proms[0]

            times = recs_xaxis[np.array(peak_idxs)]
            post_shift = times + post_time

            legals = (post_shift - duration > start_time) * (
                post_shift <= end_time
            ).astype(bool)
            trial_times.append(times[legals])
            trial_proms.append(peak_proms[legals])

        # rois with trials without triggers are dropped (lookups track the gaps)
        if all(map(lambda l: len(l) > min_peak_count, trial_times)):
            legal_times.append(trial_times)
            legal_proms.append(trial_proms)
            pos_to_roi.append(i)
            roi_to_pos[i] = count
            pos_to_grid_idx.append(grid_idxs[i])
            count += 1

    # if an hdf workspace is given, store the result within, otherwise simply
    # create and return a dict
    if workspace is None:
        grp = {}
    else:
        if type(workspace) == tuple:
            ws, k = workspace
        else:
            ws = workspace
            k = "lead"

        if k in ws:
            del ws[k]

        grp = ws.create_group(k)

    shape = (len(legal_times), recs.shape[0], n_frames, *raw_noise.shape[-2:])
    grp["stack"] = np.zeros(shape)

    for i in range(len(pos_to_roi)):
        for j, (ts, ps) in enumerate(zip(legal_times[i], legal_proms[i])):
            grp["stack"][i, j] = avg_lead_from_times(
                noise_xaxis,
                raw_noise,
                ts + post_time,
                n_frames,
                prominences=ps if weighting is not None else None,
                clip_prominence=clip_prominence,
                nonlinear_weighting=(weighting == "non-linear"),
            )

    n_legals = np.stack([np.array([len(l) for l in ts]) for ts in legal_times], axis=0)

    grp["xaxis"] = trigger_xaxis(noise_xaxis, lead_time, post_time)
    grp["n_events"] = n_legals
    grp["pos_to_roi"] = np.array(pos_to_roi)
    grp["pos_to_grid_idx"] = np.array(pos_to_grid_idx)
    if workspace is None or type(grp) is dict:
        grp["roi_to_pos"] = roi_to_pos
    else:
        pack_dataset(grp, {"roi_to_pos": roi_to_pos})

    return grp, {"times": legal_times, "proms": legal_proms}
