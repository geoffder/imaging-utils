# imaging-utils

## Suite2p helper scripts
The script
[s2p_runner.py](https://github.com/geoffder/imaging-utils/blob/main/s2p_runner.py)
is designed to provide a simple way to run the
[suite2p](https://github.com/MouseLand/suite2p) toolkit on a folder of tiffs and
repackaging the results of the analysis in an IgorPro friendly format (`hdf5`)
from the numpy binaries (`.npy`) produced by suite2p. Currently the script is
geared towards usage on bipolar data, so the configurability is limited at the
moment for the sake of simplicity. Functionality and configurability will be
extended over time as needed.

### Setup
1. Follow suite2p installation instructions found
  [here](https://github.com/MouseLand/suite2p#installation)
2. While in the suite2p conda environment you created while setting up suite2p,
  ensure the additional libraries required by `s2p_runner.py` are installed,
  like so: `conda install scikit-image pillow`

### Usage ( s2p_runner.py )
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
  * `s2p_runner.py diameter=8 gen_movies=1 gif_timestep=100`
* Currently available options incude all those in the [suite2p opts
  dict](https://github.com/MouseLand/suite2p/blob/main/suite2p/run_s2p.py) by
  the same names as detailed in their
  [docs](https://suite2p.readthedocs.io/en/latest/settings.html#main-settings).
  In addition the options listed below for `s2p_packer.py` can also be given to
  this script, as the same code is responsible for exporting the results into
  hdf5 archives after the analysis by suite2p is complete.

### s2p_packer.py
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
