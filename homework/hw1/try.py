import numpy as np
import math
from numba import cuda
TPB = 2


@cuda.jit(device=True)
def sFunc(x0):
    return (1. - 2. * math.sin(np.pi * x0) ** 2.)


@cuda.jit(' void ( float32 [:] , float32 [:]) ')
def sKernel(d_f, d_x):
    i = cuda.grid(1)
    n = d_x.size
    if i < n:
        d_f[i] = sFunc(d_x[i])


def sArray(x):
    n = x.size
    d_x = cuda.to_device(x)
    d_f = cuda.device_array(n, dtype=np. float32)
    gridDim = (n + TPB - 1) / TPB
    blockDim = TPB
    sKernel[gridDim, blockDim](d_f, d_x)
    return d_f.copy_to_host()


def main():
    x = np.array([1., 2., 3.])
    print(sArray(x))


if __name__ == '__main__':
    main()
