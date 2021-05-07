# Suite2p helper scripts
The script
[s2p_runner.py](https://github.com/geoffder/imaging-utils/blob/main/s2p_runner.py)
is designed to provide a simple way to run the
[suite2p](https://github.com/MouseLand/suite2p) toolkit on a folder of tiffs and
repackaging the results of the analysis in an IgorPro friendly format (`hdf5`)
from the numpy binaries (`.npy`) produced by suite2p. Currently the script is
geared towards usage on bipolar data, so the configurability is limited at the
moment for the sake of simplicity. Functionality and configurability will be
extended over time as needed.

# Setup
1. Follow suite2p installation instructions found
  [here](https://github.com/MouseLand/suite2p#installation)
2. While in the suite2p conda environment you created while setting up suite2p,
  ensure the additional libraries required by `s2p_runner.py` are installed,
  like so: `conda install scikit-image pillow`

# Usage
## s2p_runner.py
* Run the script `s2p_runner.py` from a directory containing tiffs which you
  would like to extract ROIs from, which might look something like this:
  * **Windows:** `C:\path\to\tiffs> python C:\path\to\script\s2p_runner.py`
  * **Linux:** `user@host /path/to/tiffs $ python /path/to/script/s2p_runner.py`
* This will run suite2p on all of the tiffs contained in the current directory
  independantly with the same configuration, outputting a separate `.h5` file
  for each of the tiffs.
* Tiffs grouped into sub-folders (one level deep only) will be treated as
  multiple scans/trials of the same scanfield. This should result in more
  accurate/reliable ROI detection, as the algorithms in suite2p will have more
  data to work with. A single `.h5` (and `.gif`) will be output to the `s2p`
  directory for each sub-folder. Within the hdf5, the extracted recordings will
  be separated into sub-groups with labels corresponding to the names of the
  originating tiffs. Shared data (e.g. ROI definitions) across trials will be
  found at the root level as usual.
* Configuration options are added as additional (optional) arguments with the form
  `arg=val`, like so:
  * `python /path/to/script/s2p_runner.py diameter=8 gen_movies=1 gif_timestep=100`
* Currently available options incude all those in the [suite2p opts
  dict](https://github.com/MouseLand/suite2p/blob/main/suite2p/run_s2p.py) by
  the same names as detailed in their
  [docs](https://suite2p.readthedocs.io/en/latest/settings.html#main-settings).
  Note that the following defaults are changed from the values in suite2p's
  `default_ops` for the bipolar GluSnfr use case, though they can be overwritten
  with arguments provided to the script as with the others:
    * spikedetect = False
    * sparse_mode = False
    * diameter = 8
    * allow_overlap = False
    * connected = False
* In addition the options listed below for `s2p_packer.py` can also be given to
  this script, as the same code is responsible for exporting the results into
  hdf5 archives after the analysis by suite2p is complete.

## s2p_packer.py
* This script will re-pack an existing **suite2p** folder into and hdf5, and
  generate a denoised gif, placing them in an adjacent directory named `s2p`.
* Designed to be ran inside of a folder containing a tiff (or tiffs, assuming
  the whole folder was used) that has already been analysed using the `suite2p` GUI.
* Usage is otherwise similar to `s2p_runner.py`.
* Currently available options are:
  * `gen_movies`: (0 or 1) whether to generate denoised movies (memory
    intensive). Off (0) by default.
  * `gif_timestep`: milliseconds per frame for the generated gif
  * `only_cells`: (0 or 1) If 1, ROIs predicted as non-cells by suite2p will
    not be included in the hdf5 data, otherwise all are included (default).

### Anatomy of an output h5
* `Fcell`: raw fluorescence of each ROI (N x T matrix)
* `Fneu`: neuropil (bg) fluorescence assigned to each ROI (N x T matrix)
* `pixels`: folder with triplet of arrays for each ROI: xpix, ypix, and lam.
* `xpix` + `ypix`: coordinates (indices) of each pixel that make up the ROI
* `lam` (naming from suite2p): weighting of each pixel
* `masks`: stack of 2D masks (constructed from the data in “pixels”) each
  containing the weighted footprint of a single ROI (X x Y x N matrix)
* `denoised`: movie constructed using a denoised signal (default: Fcell – Fneu *
  0.7) of each ROI

## preprocess.py
* This script runs the current folder (including sub-folders) of `tiffs` through a
  processing pipeline, and saves the output in an adjacent folder (in the parent
  directory) with the name postpended with `_processed`.
* The pipeline is described by the arguments given to the script when it is run.
  With the exception of the `multi_trial` flag, these are translated into
  functions to be applied to the input imaging data **in order**.
### Options:
* `multi_trial`: (0 or 1) indicates that `tiffs` found grouped in a folder represent
  multiple recordings of the same field responding to the same stimulus. This
  will be used to adapt the processing functions to the correct shape of the
  data. The placement of this argument does not matter, first makes the most
  sense though. Off (0) by default.
* `crop`: Comma separated pair of integer values indicating the number of pixels
  to trim from both sides in x, and y.
  * For example, `crop=48,0`, would result in 48 pixels removed from both sides
  of the recordings in x and zero pixels in y.
* `reduce`: Comma separated triplet of integer values indicating the 3d shape
  (time, x, y) of a kernel with which to apply a strided mean reduction/pooling to
  the data. Particularly useful for improving signal to noise while reducing the
  memory required for analysis.
  * For example, `reduce=1,4,4` will pool over the spatial dimensions with a 4x4
    kernel, quarering the number of pixels in x and y.
* `qi`: Float value defining a quality index threshold. A quality index will be
  calculated for each pixel (beam over time), and any pixels that do not meet
  the threshold will be zeroed out. As these calculations rely on multi-trial
  data, this option requires that the `multi_trial=1` option be given (and that
  data is multi trial, with `tiffs` grouped in subfolders).
  * For example, `qi=0.4`, would set to zero any pixels that did not meet a quality
    index of 0.4 over multiple stimulus presentations.
* `snr`: List of 5 comma delimited values defining the baseline time index window (2
  integers), the response window (2 integers), and a signal-to-noise ratio
  threshold. For each pixel, the snr is calculated as the ratio of the variance
  between the response window indices over the baseline window variance. If the
  snr does not pass the given threshold, the pixel is zeroed out.
  * For example, `snr=50,500,600,1200,2.0`, would calculate snr as the ratio of
  the variance between time points 600:1200 over the variance between 50:500,
  and check if it passed a threshold of 2.0.
**Example usage:**
`python /path/to/script/preprocess.py multi_trial=1 crop=48,0 reduce=1,4,4 qi=0.4`
