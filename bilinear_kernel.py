from mxnet import nd
import numpy as np

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size),
        dtype='float32')

    # weight[range(in_channels), range(out_channels), :, :] = filt
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt

    return nd.array(weight)