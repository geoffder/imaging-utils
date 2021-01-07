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

### Usage
* Run the script `s2p_runner.py` from a directory containing tiffs which you
  would like to extract ROIs from, which might look something like this:
  * **Windows:** `C:\path\to\tiffs> python C:\path\to\script\s2p_runner.py`
  * **Linux:** `user@host /path/to/tiffs $ python /path/to/script/s2p_runner.py`
* This will run suite2p on all of the tiffs contained in the current directory
  independantly with the same configuration, outputting a separate `.h5` file
  for each of the tiffs. (The ability to group tiffs into sub-folders (one deep)
  to be treated as multiple trials from the same scan field for the purposes of
  improved ROI detection is forthcoming.)
* Configuration options are added as additional (optional) arguments with the form
  `arg=val`, like so:
  * `s2p_runner.py diam=8 gif_timestep=100`
* Currently available options are:
  * `diam`: number of pixels diameter of "cell" ROIs to look for
  * `gif_timestep`: milliseconds per frame for the generated gifs

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
