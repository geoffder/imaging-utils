import sys
import os
import shutil
import re

from skimage import io

from suite2p import run_s2p, default_ops
from s2p_packer import pack_suite2p


def bipolar_ops(diam=8, allow_overlap=True):
    ops = default_ops()
    ops["spikedetect"] = False
    ops["sparse_mode"] = False
    ops["diameter"] = diam
    ops["allow_overlap"] = allow_overlap
    return ops


def analyze_folder(base_path, diam=8, gif_timestep=200):
    names = [
        f for f in os.listdir(base_path) if (f.endswith(".tiff") or f.endswith(".tif"))
    ]
    stacks = [io.imread(os.path.join(base_path, f)) for f in names]

    out_path = os.path.join(base_path, "s2p")
    os.makedirs(out_path, exist_ok=True)

    s2p_pth = os.path.join(base_path, "suite2p", "plane0")
    ops = bipolar_ops(diam=diam)
    db = {"data_path": [base_path]}

    for name, stack in zip(names, stacks):
        shutil.rmtree(os.path.join(base_path, "suite2p"), ignore_errors=True)
        db["tiff_list"] = [name]
        _out_ops = run_s2p(ops, db)
        out_name = re.sub("\.tif?[f]", "", name)
        pack_suite2p(s2p_pth, out_path, out_name, stack.shape, gif_timestep)

    shutil.rmtree(os.path.join(base_path, "suite2p"))


if __name__ == "__main__":
    settings = {}
    for arg in sys.argv[1:]:
        try:
            k, v = arg.split("=")
        except:
            msg = "Invalid argument format. Given %s, but expecting " % arg
            msg += "a key value pair delimited by '=' (e.g. diam=8)."
            print(msg)
        settings[k] = v

    analyze_folder(
        os.getcwd(),
        diam=int(settings.get("diam", 8)),
        gif_timestep=int(settings.get("gif_timestep", 200)),
    )
