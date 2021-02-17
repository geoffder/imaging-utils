import numpy as np
import torch
import torch.nn.functional as F

from image_arrays import *

def pool_tiff(pth, fname, kernel_size=2, stride=None, padding=0):
    arr = io.imread(os.path.join(pth, fname))
    t = torch.from_numpy(arr.astype(float))
    pooled = F.avg_pool2d(t, kernel_size, stride, padding).numpy()
    u16 = normalize_uint16(remove_offset(pooled))
    label = "pooled_%i" % kernel_size
    array_to_tiff(os.path.join(pth, label), label, u16)
