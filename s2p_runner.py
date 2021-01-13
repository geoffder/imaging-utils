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


def is_tiff(name):
    return (name.endswith(".tiff") or name.endswith(".tif"))


def analyze_folder(base_path, diam=8, gif_timestep=200):
    contents = os.listdir(base_path)
    names = [f for f in contents if is_tiff(f)]
    stacks = [io.imread(os.path.join(base_path, f)) for f in names]

    # for running sub-folders of tiffs as multiple trials of a single field
    sub_groups = {}
    for d in contents:
        if os.path.isdir(d):
            ns = [f for f in os.listdir(d) if is_tiff(f)]
            if len(ns) > 0:
                sub_groups[d] = ns

    out_path = os.path.join(base_path, "s2p")
    os.makedirs(out_path, exist_ok=True)

    s2p_path = os.path.join(base_path, "suite2p", "plane0")
    ops = bipolar_ops(diam=diam)
    db = {"data_path": [base_path]}

    for name, stack in zip(names, stacks):
        shutil.rmtree(os.path.join(base_path, "suite2p"), ignore_errors=True)
        db["tiff_list"] = [name]
        _out_ops = run_s2p(ops, db)
        out_name = re.sub("\.tif?[f]", "", name)
        pack_suite2p(s2p_path, out_path, out_name, stack.shape[1:], gif_timestep)

    for grp, tiff_list in sub_groups.items():
        sub_path = os.path.join(base_path, grp)
        sub_s2p_path = os.path.join(sub_path, "suite2p", "plane0")
        sub_stacks = [io.imread(os.path.join(sub_path, f)) for f in tiff_list]
        pts = {
            re.sub("\.tif?[f]", "", f): s.shape[0]
            for f, s in zip(tiff_list, sub_stacks)
        }
        space_dims = sub_stacks[0].shape[1:]
        shutil.rmtree(os.path.join(sub_path, "suite2p"), ignore_errors=True)
        db = {"data_path": [sub_path], "tiff_list": tiff_list}
        _out_ops = run_s2p(ops, db)
        pack_suite2p(sub_s2p_path, out_path, grp, space_dims, gif_timestep, trial_pts=pts)
        shutil.rmtree(os.path.join(sub_path, "suite2p"), ignore_errors=True)

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
