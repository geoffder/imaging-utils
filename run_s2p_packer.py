import sys
import os
import re
from skimage import io
from s2p_packer import *

if __name__ == "__main__":
    settings = {}
    for arg in sys.argv[1:]:
        try:
            k, v = arg.split("=")
            settings[k] = v
        except:
            msg = "Invalid argument format. Given %s, but expecting " % arg
            msg += "a key value pair delimited by '=' (e.g. gif_timestep=100)."
            raise (ValueError(msg))

    base_path = os.getcwd()
    names = [
        f for f in os.listdir(base_path) if (f.endswith(".tiff") or f.endswith(".tif"))
    ]
    stacks = [io.imread(os.path.join(base_path, f)) for f in names]

    trial_pts = (
        {re.sub("\.tif?[f]", "", f): s.shape[0] for f, s in zip(names, stacks)}
        if len(names) > 0
        else None
    )

    space_dims = stacks[0].shape[1:]
    out_path = os.path.join(base_path, "s2p")
    os.makedirs(out_path, exist_ok=True)
    out_name = re.sub("\.tif?[f]", "", names[0])
    s2p_pth = os.path.join(base_path, "suite2p", "plane0")

    pack_suite2p(
        s2p_pth,
        out_path,
        out_name,
        space_dims,
        trial_pts=trial_pts,
        gif_timestep=int(settings.get("gif_timestep", 200)),
        exclude_non_cells=bool(settings.get("only_cells", 0)),
        denoised_movies=bool(int(settings.get("gen_movies", 0))),
    )
