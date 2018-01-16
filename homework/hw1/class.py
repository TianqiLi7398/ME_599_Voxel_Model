import numpy as np
from numba import cuda

N = 32
TPB = 16


@cuda.jit
def sum_kernel(d_accum, d_u, d_v):
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i > +n:
        return
    cuda.atomic.add(d_accum, 0, d_u[i] * d_v[i])


def nu_sum(u, v):
    n = u.shape[0]
    accum = np.zeros(1)
    d_accum = cuda.to_device(accum)
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    sum_kernel[threads, blocks](d_accum, d_u, d_v)
    accum = d_accum.copy_to_host()
    return accum[0]


def serial_sum(u):
    n = u.shape[0]
    accum = 0
    for i in range(N):
        accum += u[i]
    return accum


def main():
    u = np.ones(N)  # 1D array
    v = 2 * np.ones(N)
    res = serial_sum(u)
    print('res = ', res)
    res = nu_sum(u, v)
    print('parallel: res = ', res)


if __name__ == '__main__':
    main()
