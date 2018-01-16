import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
TPB = 8


# Pronlem 2

# 2a

def save_array(n):
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.pi * (i / (n - 1))
    return y

# Problem 3
# 3a


def scalar_mult(u, c):
    # this function calculates scalar multiplication of a scalar and
    # an array
    # inputs:
        # u: the scalar to multiple
        # c: the array to multiple
    # output:
        # prod: scalar multiplication product
    prod = u * c
    return prod

# 3b


def component_add(u, v):
    # this function calcuates the component-wise addition of 2 arrays
    # inputs:
        # u: the first addend
        # v: the second addend
    # output:
        # summ: sum of component-wise addition
    summ = np.add(u, v)
    return summ

# 3c


def linear_function(u, c, d):
    # this function evaluates the linear function y = c*x + d
    # inputs:
        # u: the scalar
        # c: the first addend array
        # d: the second addend array
    # output:
        # y: sum of the linear function
    y = component_add(scalar_mult(u, c), d)
    return y

# 3d


def component_mult(u, v):
    # this function calcuartes the component-wise mutiplication of 2 arrays
    # inputs:
        # u: the first multiplicand array
        # v: the second multiplicand array
    # output:
        # prod: product of component-wise multiplication
    prod = np.multiply(u, v)
    return prod

# 3e


def inner(u, v):
    # this function calcuates the inner product of 2 arrays
    # inputs:
        # u: the first multiplicand array
        # v: the second multiplicand array
    # output:
        # prod: inn product of component-wise multiplication
    prod = np.dot(u, v)
    return prod

# 3f


def norm_1(u):
    # this function calcuates the norm of 1 arrays
    # inputs:
        # u: the array to get norm
    # output:
        # prod: norm
    prod = np.linalg.norm(u)
    return prod


# Problem 4
# 4a


@cuda.jit
def scalar_mult_kernel(d_out, d_u, c):
    i = cuda.grid(1)
    n = d_u.size
    if i < n:
        d_out[i] = d_u[i] * c


def nu_scalar_mult(u, c):
    n = u.size

    d_u = cuda.to_device(u)
    d_out = cuda.device_array(n)

    gridDim = (n + TPB - 1) // TPB
    blockDim = TPB

    scalar_mult_kernel[gridDim, blockDim](d_out, d_u, c)
    return d_out.copy_to_host()

# 4b


def nu_component_add(u, v):
    n = u.size
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    gridDim = (n + TPB - 1) // TPB
    blockDim = TPB
    component_add_kernel[gridDim, blockDim](d_out, d_u, d_v)
    return d_out.copy_to_host()


@cuda.jit
def component_add_kernel(d_out, d_u, d_v):
    i = cuda.grid(1)
    n = d_u.size
    if i < n:
        d_out[i] = d_u[i] + d_v[i]

    # 4c


@cuda.jit
def linear_function_kernel(d_out, d_u, c, d_d):
    i = cuda.grid(1)
    n = d_u.size
    if i < n:
        d_out[i] = c * d_u[i] + d_d[i]


def nu_linear_function(u, c, d):
    n = u.size
    d_u = cuda.to_device(u)
    d_d = cuda.to_device(d)
    d_out = cuda.device_array(n)
    gridDim = (n + TPB - 1) // TPB
    blockDim = TPB
    linear_function_kernel[gridDim, blockDim](d_out, d_u, c, d_d)
    return d_out.copy_to_host()

# 4d


@cuda.jit
def component_mult_kernel(d_out, d_u, d_v):
    i = cuda.grid(1)
    n = d_u.shape[0]
    if i > n - 1:
        return
    d_out[i] = d_u[i] * d_v[i]


def nu_component_mult(u, v):
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    component_mult_kernel[blocks, threads](d_out, d_u, d_v)
    return d_out.copy_to_host()

# 4e


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

# 4f


def norm(u):
    pass

# Problem 5

# 5a


def prob5_a(n):
    '''
    This function is defined to create a function in P5.a'''
    # i. Set each entry in v equal to 1.
    v = np.ones(n)

    # ii. Set each entry in u to 1/(n-1), then reset the first
    # entry to 1 (remember that means u[0]=1).

    u = np.ones(n)
    u.fill(1 / (n - 1))
    u[0] = 1

    # iii. Compute z=-u and the norm of u+z. Inspect and verify
    # your results.
    z = -u
    norm = np.linalg.norm(u + z)

    # iv. Compute the dot product (inner product) of u and v. Inspect
    # and verify your results.
    dot_pro = np.dot(u, v)

    # v. Create a “reversed dot” product in which you sum the
    # contributions in reverse order. Inspect and verify your results
    # when computing the dot product of u and v using this new function.
    sum_ = 0
    for i in range(len(u)):
        sum_ += u[- (i + 1)] * v[- (i + 1)]

    return dot_pro, sum_


def main():
    # Problem 1
    # ADD YOUR OWN CODE
    print('#1. I ran through the examples in the text introducing python and numpy.')
    print('Things worked as expected (or not).')

    # Problem 2

    n = 11
    # 2a.

    print('#2. a, set n = 10, and the Numpy array is ', save_array(n))

    # 2b.
    y = np.linspace(0, np.pi, n)
    print('b. using np.linspace we can get the array ', y,
          ' which is the same as Part a.')

    # 2c.
    x = np.linspace(1, n, n)
    plt.plot(x, y, 'b*')
#    plt.show()

    # Problem 3

    v = np.array([1, 2, 3.])
    u = np.array([0, 1, 2.])
    d = np.array([2, 3, 4.])
    c = 3

    print('#3. The following answers are run with the parameters')
    print('u = [0, 1, 2]')
    print('v = [1, 2, 3]')
    print('d = [2, 3, 4]')
    print('c = 3')

    # 3a.

    print('a. Scalar multiplication of u and c is ', scalar_mult(u, c))

    # 3b.

    print('b. Component-wise addition of u and v is ', component_add(u, v))

    # 3c.

    print('c. Evaluate the linear function y = c*u + d: ', linear_function(u, c, d))

    # 3d.

    print('d. Component-wise multiplication of u and v is ', component_mult(u, v))

    # 3e.

    print('e. Inner product of u and v is ', inner(u, v))

    # 3f.

    print('f. Euclidean norm of u is ', norm_1(d))

    # Problem 4

    # 4a.

    print('#4a. The parallel version of scalar multiplication is ',
          nu_scalar_mult(u, c))
    # 4b.

    print('b. The parallel version of component_wise addition is ',
          nu_component_add(u, v))
    # 4c.

    print('c. The parallel version of linear function is ',
          nu_linear_function(u, c, d))

    # 4d.

    print('d. The parallel version of component_wise multiplication is ',
          nu_component_mult(u, v))
    # 4e.

    print('e. The parallel version of linear function is ',
          nu_sum(np.ones(32), np.ones(32)))


# Problem 5
    print('#5.a Create input arrays u and v of length n')

    dot, rev_dot = prob5_a(5)
    print('for n=5, I found the dot product = ', dot,
          'and reverse dot product = ', rev_dot)

    dot, rev_dot = prob5_a(10000000)
    print('b. for n=10,000,000, I found the dot product = ', dot,
          'and reverse dot product = ', rev_dot)

    print('My results lead me to the following commments/conclusions...')


if __name__ == '__main__':
    main()
