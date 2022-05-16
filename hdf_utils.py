import h5py as h5


def pack_hdf(pth, data_dict, compression="gzip"):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure."""

    def rec(data, grp):
        for k, v in data.items():
            typ = type(v)
            if typ is dict:
                rec(v, grp.create_group(k))
            elif typ == float or typ == int:
                grp.create_dataset(k, data=v)  # can't compress scalars
            else:
                grp.create_dataset(k, data=v, compression=compression)

    with h5.File(pth + ".h5", "w") as pckg:
        rec(data_dict, pckg)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5.Dataset else unpack_hdf(v) for k, v in group.items()
    }
