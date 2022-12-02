import numpy as np
from scipy import optimize
from scipy import signal
from image_arrays import nearest_index


def linear_prediction(
    noise_xaxis,
    noise,
    resp_hz,
    triggered,
    triggered_pivot_idx,
    centre_idxs,
    cols=8,
    rows=8,
    time=2.0,
    mean_sub=True,
):
    noise_frames, _, noise_rows = noise.shape
    noise_dt = noise_xaxis[1] - noise_xaxis[0]
    flt_frames = int(time / noise_dt)
    pred_frames = int(resp_hz * noise_dt * noise_frames) - flt_frames
    pred = np.zeros((triggered.shape[0], pred_frames))
    pred_xaxis = (
        np.arange(pred_frames) / resp_hz + noise_xaxis[0] + (flt_frames / resp_hz)
    )

    for roi in range(pred.shape[0]):
        centre_col = centre_idxs[roi] // noise_rows
        centre_row = centre_idxs[roi] % noise_rows
        flt = (
            triggered[
                roi,
                triggered_pivot_idx - flt_frames - 1 : triggered_pivot_idx,
                centre_col - cols : centre_col + cols,
                centre_row - rows : centre_row + rows,
            ]
            - 0.5
        )
        res = np.zeros(noise.shape[0] - flt_frames)
        for c in range(cols * 2):
            for r in range(rows * 2):
                res += np.convolve(
                    noise[:, centre_col - cols + c, centre_row - rows + r],
                    np.flip(flt[:, c, r]),
                    mode="valid",
                )
        pred[roi] = signal.resample(res, pred.shape[1]) / (cols * rows * 2)

    if mean_sub:
        pred -= np.mean(pred, axis=-1, keepdims=True)

    return {"xaxis": pred_xaxis, "pred": pred}


def output_function(prediction, resp_xaxis, responses, n_bins=100, poly=False):
    bin_sz = len(prediction["xaxis"]) // n_bins
    n_rois = responses.shape[0]
    pred_bins = np.zeros((n_rois, n_bins))
    resp_bins = np.zeros((n_rois, n_bins))
    idx_offset = nearest_index(resp_xaxis, prediction["xaxis"][0])

    for roi in range(n_rois):
        sorted_idxs = np.argsort(prediction["pred"][roi])
        for i in range(n_bins):
            idxs = sorted_idxs[i * bin_sz : (i + 1) * bin_sz]
            pred_bins[roi][i] = np.mean(prediction["pred"][roi][idxs])
            resp_bins[roi][i] = np.mean(responses[roi][idxs + idx_offset])

    expf = lambda t, a, b, c: a * np.exp(t / -b) + c

    fit_coefs, fit_fun, fit_x, fit_y = [[] for _ in range(4)]
    for roi in range(n_rois):
        if poly:
            fit_coefs.append(np.polyfit(pred_bins[roi], resp_bins[roi], 3))
            fit_fun.append(np.poly1d(fit_coefs[-1]))
        else:
            fit_coefs.append(
                optimize.curve_fit(expf, pred_bins[roi], resp_bins[roi])[0]
            )
            fit_fun.append(lambda x: expf(x, *fit_coefs[-1]))
        fit_x.append(np.linspace(np.min(pred_bins[roi]), np.max(pred_bins[roi]), 1000))
        fit_y.append(np.array([fit_fun[-1](x) for x in fit_x[-1]]))

    return {
        "pred_idx_offset": idx_offset,
        "resp_bins": resp_bins,
        "pred_bins": pred_bins,
        "fit_coefs": fit_coefs,
        "fit_fun": fit_fun,
        "fit_x": np.stack(fit_x, axis=0),
        "fit_y": np.stack(fit_y, axis=0),
        "nonlinear": np.array(
            [
                [fit_fun[roi](x) for x in prediction["pred"][roi]]
                for roi in range(n_rois)
            ]
        ),
    }
