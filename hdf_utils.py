import h5py as h5

INT_PREFIX = "packed_int_"


def pack_dataset(h5_file, data_dict, compression="gzip"):
    """Takes data organized in a python dict, and stores it in the given hdf5
    with the same structure. Keys are converted to strings to comply to hdf5
    group naming convention. In `unpack_hdf`, if the key is all digits, it will
    be converted back from string."""

    def rec(data, grp):
        for k, v in data.items():
            k = "%s%i" % (INT_PREFIX, k) if isinstance(k, int) else k
            if type(v) is dict:
                rec(v, grp.create_group(k))
            elif isinstance(v, float) or isinstance(v, int):
                grp.create_dataset(k, data=v)  # can't compress scalars
            else:
                grp.create_dataset(k, data=v, compression=compression)

    rec(data_dict, h5_file)


def pack_hdf(pth, data_dict):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure. Keys are converted to strings to comply to hdf5 group naming
    convention. In `unpack_hdf`, if the key is all digits, it will be converted
    back from string."""
    with h5.File(pth + ".h5", "w") as pckg:
        pack_dataset(pckg, data_dict)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""

    def fix_key(k):
        if isinstance(k, str) and k.startswith(INT_PREFIX):
            return int(k[len(INT_PREFIX) :])
        else:
            return k

    return {
        fix_key(k): v[()] if type(v) is h5.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }
