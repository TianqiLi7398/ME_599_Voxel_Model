'''
ME 599 Voxel Model, homework 2
Tianqi Li
Jan 29, 2018
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math
import numba
from numba import jit, cuda, float32
seaborn.set()
import time
from mpl_toolkits.mplot3d import Axes3D

NX = 64
NY = 64

@cuda.jit(device = True)
def f2D(x0, y0):
    return math.sin(np.pi * x0) * math.sinh(np.pi * y0) / math.sinh(np.pi)

def f2D_ser(x0, y0):
    return math.sin(np.pi * x0) * math.sinh(np.pi * y0) / math.sinh(np.pi)

def fArray2D_ser(x, y):
    m = x.shape[0]
    n = y.shape[0]
    f = np.empty(shape=[m, n], dtype=np.float32)
    for i in range(m):
        for j in range(n):
            f[i, j] = f2D_ser(x[i], y[j])
    return f


@cuda.jit
def fKernel2D(d_f, d_x, d_y):
    i, j = cuda.grid(2)
    m, n = d_f.shape
    if i < m and j < n:
        d_f[i, j] = f2D(d_x[i], d_y[j])


def fArray2D_par(x, y):
    TPBX, TPBY = 16, 16
    m = x.shape[0]
    n = y.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array(shape=[m, n], dtype=np.float32)
    gridDims = (m + TPBX - 1) // TPBX, (n + TPBY - 1) // TPBY
    blockDims = TPBX, TPBY
    fKernel2D[gridDims, blockDims](d_f, d_x, d_y)

    return d_f.copy_to_host()

def fArray2D_p(x, y, TPBX, TPBY):
    m = x.shape[0]
    n = y.shape[0]
    # TPBX, TPBY = 16, 16
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array(shape = [m, n], dtype = np.float32)
    gridDims = (m + TPBX - 1) // TPBX, (n + TPBY - 1) // TPBY
    blockDims = TPBX, TPBY
    fKernel2D[gridDims, blockDims](d_f, d_x, d_y)
    

    return d_f.copy_to_host()

# @cuda.jit(device = True)
# def f2D_p(x0, y0):
#     return math.sin(np.pi * x0) * math.sinh(np.pi * y0) / math.sinh(np.pi)




def problem_1(xarray_len, yarray_len):
    x = np.linspace(0, 1, xarray_len, dtype=np.float32)
    y = np.linspace(0, 1, yarray_len, dtype=np.float32)
    a = time.time()
    fser = fArray2D_ser(x, y)
    b = time.time()
    fpar = fArray2D_par(x, y)
    c = time.time()
    # calculate acceleration: CPU time / GPU Time
    return b - a, c - b, (b - a) / (c - b)


def problem2(x, y, TPBX,TPBY):
    
    m = x.shape[0]
    n = y.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array(shape=[m, n], dtype=np.float32)
    gridDims = (m + TPBX - 1) // TPBX, (n + TPBY - 1) // TPBY
    blockDims = TPBX, TPBY
    fKernel2D[gridDims, blockDims](d_f, d_x, d_y)


    return d_f.copy_to_host()

# Problem 4

def fArray2D_p_timed(x, y, TPBX, TPBY):
    m = x.shape[0]
    n = y.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array(shape = [m, n], dtype = np.float32)
    gridDims = (m + TPBX - 1) // TPBX, (n + TPBY - 1) // TPBY
    blockDims = TPBX, TPBY
    start = time.time()
    fKernel2D[gridDims, blockDims](d_f, d_x, d_y)
    elapsed = time.time() - start

    return d_f.copy_to_host(), elapsed

# Problem 5


@cuda.jit
def norm_kernel(d_f, d_ap, p):
    nx, ny = d_ap.shape
    i, j = cuda.grid(2)
    if i >= nx or j >= ny:
        return
    d_f[i, j] = d_ap[i, j] ** p

def p_norm_helper(ap, p, TPBX, TPBY):
    nx,ny = ap.shape
    d_ap = cuda.to_device(ap)
    d_f = cuda.device_array(shape = [nx, ny], dtype = np.float32)
    threads = (TPBX, TPBY)
    BX = (nx + TPBX - 1) // TPBX
    BY = (ny + TPBY - 1) // TPBY
    blocks = (BX,BY)
    norm_kernel[blocks, threads](d_f, d_ap, p)
    return d_ap.copy_to_host()

def p_norm(ap, p, TPBX, TPBY):
    d_ap_out = p_norm_helper(ap, p, TPBX, TPBY)
    d_ap_out_flatten = d_ap_out.flatten()
    reduced_sum = cuda.reduce(lambda a, b: a + b)
    reduced_ap = reduced_sum(d_ap_out_flatten)
    rst = reduced_ap ** (1 / p)
    return rst

# Linf norm kernel
def norm_inf(ap):
    max_val = np.amax(ap)
    return max_val



# Problem 6

@cuda.jit
def kernel_derivativeArray(d_x, d_y, d_ratio, x_pre, x_next, num):
    n = 300
    h = 0.01
    i= cuda.grid(1)
    if i >= d_x.shape[0]:
        return
    x_pre_0 = d_x[i]
    x_pre_1 = d_y[i]

    if num == 1:
        for m in range(n):
            x_next_0 = x_pre_0 + h*1*x_pre_1
            x_next_1 = x_pre_1 + h*(-1)*x_pre_0
            x_pre_0 = x_next_0
            x_pre_1 = x_next_1
    elif num ==2:
        for m in range(n):
            x_next_0 = x_pre_0 + h*1*x_pre_1
            x_next_1 = x_pre_1 + h*(-x_pre_0 - 0.1*x_pre_1)
            x_pre_0 = x_next_0
            x_pre_1 = x_next_1
    elif num ==3:
        for m in range(n):
            x_next_0 = x_pre_0 + h*1*x_pre_1
            x_next_1 = x_pre_1 + h*(-x_pre_0 + 0.1*(1-x_pre_0**2)*x_pre_1)
            x_pre_0 = x_next_0
            x_pre_1 = x_next_1
    # d_out[0,i] = x_next_0
    # d_out[1,i] = x_next_1
    dis_pre = math.sqrt(d_x[i]**2 + d_y[i]**2)
    d_ratio[i] = math.sqrt(x_next_0**2+x_next_1**2)/dis_pre


def derivativeArray(x,y, num):
    N = x.shape[0]
    TPBX = 16 
    TPBY = 16
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_ratio = cuda.device_array(shape =N)
    x_pre = cuda.device_array(shape = [1,2])
    x_next = cuda.device_array(shape = [1,2])
    gridDims = (N+TPBX-1)//TPBX
    blockDims = TPBX
    kernel_derivativeArray[gridDims, blockDims](d_x, d_y, d_ratio, x_pre, x_next, num)
    out = d_ratio.copy_to_host()
    # dist = np.sqrt(np.multiply(out,out)[0] + np.multiply(out,out)[1])
    return out

def main():

    print("ME 599 Voxel Model, homework 2")
    print("Tianqi Li")
    print("Jan 29, 2018")
    print("____________________________________________________")

    # problem 2

    print('Problem 2, plot the execution times and acceleration V.S. array size')
    xarray_len, yarray_len = 10, 10
    ser_time, par_time, ratio = np.empty((xarray_len, yarray_len)), np.empty(
        (xarray_len, yarray_len)), np.empty((xarray_len, yarray_len))
    ser_plot, par_plot, ratio_plot, dx, dy = [], [], [], [], []
    
    for i in range(xarray_len):
        for j in range(yarray_len):
            ser_time[i][j], par_time[i][j], ratio[i][j] = problem_1(i+1, j+1)
            ser_plot.append(ser_time[i][j])
            par_plot.append(par_time[i][j])
            ratio_plot.append(ratio[i][j])
            dx.append(i + 1)
            dy.append(j + 1)
            # print(i, j)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax = plt.axes(projection='3d')
    
    # ax.plot3D(dx, dy, ratio_plot, 'b*')
    ax.set_title('Serial Execution Time')
    ax.set_xlabel('Size of xArray')
    ax.set_ylabel('Size of yArray')
    
    ax.plot3D(dx, dy, ser_plot, 'b*')
    
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('Parallelize Execution Time')
    ax.set_xlabel('Size of xArray')
    ax.set_ylabel('Size of yArray')
    
    ax.plot3D(dx, dy, par_plot, 'b*')
    
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('acceleration')
    ax.set_xlabel('Size of xArray')
    ax.set_ylabel('Size of yArray')
    
    ax.plot3D(dx, dy, ratio_plot, 'b*')
    
    plt.show()
    
    # Problem 3
    print("____________________________________________________")

    print('Problem 3, find out the largest square block size:')
    
    N = 100
    
    TPBX, TPBY = 32,32 # try this out --- get largest blockDim --- problem3
    x = np.linspace(0, 1, N, dtype = np.float32)
    y = np.linspace(0, 1, N, dtype = np.float32)
    # val = problem2(x, y, TPBX, TPBY)

    try:
        val = problem2(x, y, TPBX, TPBY)
    except numba.cuda.cudadrv.driver.CudaAPIError:

        print("I use Linux command 'lspci | grep VGA', my GPU is NVIDIA Corporation GF100 [GeForce GTX 480] (rev a3)",
            ", and I found two errors when I reach the limit. When I run")
        print("TPBX, TPBY; (32,32), I get the error: numba.cuda.cudadrv.driver.CudaAPIError: [701] Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;")
        print("TPBX, TPBY: (64,64) or larger, I get the error: numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE;")

    print("____________________________________________________")
    # Problem 4
    NX_try = 100
    NY_try = 100
    TPBX, TPBY = 16, 16 # try this out --- different aspect ratio --- problem4a
    # square vs rectangular
    x = np.linspace(0, 1, NX_try, dtype = np.float32)
    y = np.linspace(0, 1, NY_try, dtype = np.float32)

    N=5

    print("Problem 4")
    tplus16, tplus4, ktplus16, ktplus4 = 0,0, 0,0

    for i in range(N):
        a = time.time()
        farray2D_parallel = fArray2D_p(x, y, 16, 16)
        tplus16 += time.time() - a
        # tplus += dt
        f, t = fArray2D_p_timed(x, y, 16, 16)
        ktplus16 += t

    for i in range(N):
        a = time.time()
        farray2D_parallel = fArray2D_p(x, y, 16, 16)
        tplus4 += time.time() - a
        # tplus += dt
        f, t = fArray2D_p_timed(x, y, 16, 4)
        ktplus4 += t

    print("a. Outside cell, I tried 16*16 and 16*4, the execution time for 16*16 is %f, the execution time for 16*4 is %f"%(tplus16/N,tplus4/N) )

    print("b. Warp waround, I tried 16*16 and 16*4, the execution time for 16*16 is %f, the execution time for 16*4 is %f"%(ktplus16/N,ktplus4/N) )


    print("____________________________________________________")
    # Problem 5

    print("Problem 5")

    # constract a 256*256 array

    N = 256
    TPBX, TPBY = 16, 16

    a = np.linspace(0,2*math.pi,256, dtype = np.float32)
    array_built = np.empty((256,256))
    for i in range(256):
        for j in range(256):
            array_built[i,j] = math.sin(a[i])*math.sin(a[j])


    # plot3d(array_built, a, a, vars=['x', 'y', 'f(x,y)'], 
    # titlestring='Full cycle of a sin wave')

   
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    X,Y = np.meshgrid(a,a)
    ax.plot_wireframe(X,Y ,array_built, color='red')
    ax.set_title('Full cycle of a sin wave')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show() 




    print("L^2 Norm: ",p_norm(array_built, 2, TPBX, TPBY))
    print("L^4 Norm: ",p_norm(array_built, 4, TPBX, TPBY))
    print("L^6 Norm: ",p_norm(array_built, 6, TPBX, TPBY))
    print("As p ->INFINITE, L^p Norm: ",norm_inf(array_built))
    print("____________________________________________________")

    # problem 6

    print("Problem 6")
    print("a. We cannot directly parallelize computation over a grid of time, because for parallelize all threads need an initial value to calculate forward difference derivative approximation.")

    print("")
    print("b. Yes we can. We need to get a grid (grid = block*thread) of IPV, and calculate the derivative in each grid separately.")
    print("")
    print("c. Please see the plot")

    N = 100
    rad = np.linspace(0,3,N)
    theta = np.linspace(0,2*np.pi,N)
    x = np.empty(shape = [N,N])
    v = np.empty(shape = [N,N])
    for i in range(100):
        x[i] = rad[i]*np.sin(theta)
        v[i] = rad[i]*np.cos(theta)
    x = x.flatten()
    v = v.flatten()


    ratio1 = derivativeArray(x,v,1)
 
    fig = plt.figure(4)
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot(x, v, ratio1)
    ax.set_title('problem 6.c')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    plt.show()

    print("d. Please see the plot")
    ratio2 = derivativeArray(x,v,2)
 
    fig = plt.figure(4)
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot(x, v, ratio2)
    ax.set_title('problem 6.d')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    plt.show()

    print("e. Please see the plot")
    ratio3 = derivativeArray(x,v,3)

    fig = plt.figure(4)
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot(x, v, ratio3)
    ax.set_title('problem 6.e')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    plt.show()



if __name__ == '__main__':
    main()
