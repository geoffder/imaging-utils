import h5py as h5
from typing import Union

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


class Workspace:
    def __init__(self, path=None):
        if path is None:
            self.is_hdf = False
            self._data: Union[dict, h5.File] = {}
        else:
            self.is_hdf = True
            self._data: Union[dict, h5.File] = h5.File(path, mode="w")

    def __setitem__(self, key, item):
        if self.is_hdf and key in self._data:
            try:
                self._data[key][...] = item
            except TypeError:  # if new item has a different shape
                del self._data[key]
                self._data[key] = item
        elif self.is_hdf and type(item) is dict:
            if key in self._data:
                del self._data[key]
            pack_dataset(self._data, {key: item})
        else:
            self._data[key] = item

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)

    def has_key(self, k):
        return k in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def create_group(self, key):
        if self.is_hdf:
            if key in self._data:
                del self._data[key]
            return self._data.create_group(key)
        else:
            self._data[key] = {}
            return self._data[key]

    def close(self):
        if self.is_hdf:
            self._data.close()
