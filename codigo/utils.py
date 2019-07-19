import numba
from numba import cuda
import numpy as np

"""
Implementación de scan inclusivo
"""
@cuda.jit
def scan_block(d_data, d_scan, aux):
    temp = cuda.shared.array(shape=32, dtype=numba.float32)
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    temp1 = d_data[tid+cuda.blockIdx.x*cuda.blockDim.x]
    
    d = 1
    while d < 32:
        temp2 = cuda.shfl_up_sync(-1, temp1, d)
        if tid % 32 >= d:
            temp1 += temp2
        d <<=1
    if tid % 32 == 31:
        temp[tid//32] = temp1
    cuda.syncthreads()
    
    if tid < 32:
        temp2 = 0.0
        if tid < cuda.blockDim.x/32:
            temp2 = temp[tid]
            
        d = 1
        while d < 32:
            temp3 = cuda.shfl_up_sync(-1, temp2, d)
            if tid % 32 >= d:
                temp2 += temp3
            d<<=1
            
        if tid < cuda.blockDim.x//32:
            temp[tid] = temp2
    cuda.syncthreads()
    
    if tid >= 32:
        temp1 += temp[tid//32-1]
    cuda.syncthreads()
    
    if idx < d_scan.size:
        d_scan[idx] = temp1
    if tid == cuda.blockDim.x-1:
        aux[cuda.blockIdx.x] = temp1
    
@cuda.jit
def scan_sum(data, aux):
    # Índices
    bidx = cuda.blockIdx.x 
    idx = cuda.grid(1)
    if cuda.blockDim.x <= idx < data.size:
        data[idx] += aux[cuda.blockIdx.x-1]

def scan(d_x, d_scan, tpb=1024):
    aux = np.empty(d_x.size//tpb, dtype=d_x.dtype)
    scan_block[d_x.size//tpb+1, tpb](d_x, d_scan, aux)
    scan_sum[d_x.size//tpb+1, tpb](d_scan, np.cumsum(aux))


@cuda.jit
def multi_scan_block(d_data, d_scan, aux):
    temp = cuda.shared.array(shape=32, dtype=numba.float32)
    tid = cuda.threadIdx.x
    idx = tid+cuda.blockIdx.x*cuda.blockDim.x
    if idx < d_data.shape[1]:
        temp1 = d_data[cuda.blockIdx.y, idx]        
    d = 1
    while d < 32:
        temp2 = cuda.shfl_up_sync(-1, temp1, d)
        if tid % 32 >= d:
            temp1 += temp2
        d <<=1
    if tid % 32 == 31:
        temp[tid//32] = temp1
    cuda.syncthreads()
    
    if tid < 32:
        temp2 = 0.0
        if tid < cuda.blockDim.x/32:
            temp2 = temp[tid]
            
        d = 1
        while d < 32:
            temp3 = cuda.shfl_up_sync(-1, temp2, d)
            if tid % 32 >= d:
                temp2 += temp3
            d<<=1
            
        if tid < cuda.blockDim.x//32:
            temp[tid] = temp2
    cuda.syncthreads()
    
    if tid >= 32:
        temp1 += temp[tid//32-1]
    cuda.syncthreads()
    
    if idx < d_data.shape[1]:
        d_scan[cuda.blockIdx.y, idx] = temp1

    if tid == cuda.blockDim.x-1:
        for i in range(cuda.blockIdx.x, cuda.gridDim.x):
            cuda.atomic.add(aux, (cuda.blockIdx.y, i), temp1)


@cuda.jit
def multi_scan_sum(data, aux):
    col = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    if cuda.blockDim.x <= col < data.shape[1]:
        data[cuda.blockIdx.y, col] += aux[cuda.blockIdx.y, cuda.blockIdx.x-1]

@cuda.jit
def multi_scan_sum_with_gini(data, values, aux, my_min):
    row = cuda.blockIdx.y
    col = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x

    n = data.shape[1]
    total_true = aux[0,aux.shape[1] - 1]

    if cuda.grid(1) == 0:
        my_min[0] = np.inf

    if cuda.blockDim.x <= col < data.shape[1]:
        data[row, col] += aux[row, cuda.blockIdx.x-1]

    if col < data.shape[1]:
        n_i = col % n + 1
        n_d = n - n_i
        if n_d != 0 and values[row, col] != values[row, col + 1]:
            t_i = data[row, col]
            t_d = total_true - t_i
            data[row, col] = (t_i * (n_i - t_i)/(n_i)) + (t_d * (n_d - t_d)/(n_d))
        else:
            data[row, col] = n

@cuda.jit
def multi_scan_sum_with_addresses(data, aux, address, my_flag, my_offset):
    row = cuda.blockIdx.y
    col = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x

    total_false = aux[0, aux.shape[1] - 1]

    if cuda.blockDim.x <= col < data.shape[1]:
        data[row, col] += aux[row, cuda.blockIdx.x-1]

    if col < my_flag.shape[1]:
        address[row, col] = col + total_false - data[row, col] if my_flag[row, col] else data[row, col] - 1
        address[row, col] += my_offset

def multi_scan(d_x, d_scan, tpb=1024, stream=0):
    aux = cuda.device_array((d_x.shape[0], d_x.shape[1]//tpb+1), dtype=d_x.dtype, stream=stream)
    aux[:,:] = 0
    blocks = (d_x.shape[1]//tpb+1, d_x.shape[0])
    multi_scan_block[blocks, tpb, stream](d_x, d_scan, aux)
    multi_scan_sum[blocks, tpb, stream](d_scan, aux)


def multi_scan_with_gini(d_x, d_scan, d_values, my_min, aux, tpb=1024, stream=0):
    blocks = (d_x.shape[1]//tpb+1, d_x.shape[0])
    multi_scan_block[blocks, tpb, stream](d_x, d_scan, aux)
    multi_scan_sum_with_gini[blocks, tpb, stream](d_scan, d_values, aux, my_min)
    return aux[0].copy_to_host(stream=stream)[aux.shape[1]-1]

def multi_scan_with_address(d_x, d_scan, address, my_flag, aux, my_offset, tpb=1024, stream=0):
    blocks = (d_x.shape[1]//tpb+1, d_x.shape[0])
    multi_scan_block[blocks, tpb, stream](d_x, d_scan, aux)
    multi_scan_sum_with_addresses[blocks, tpb, stream](d_scan, aux, address, my_flag, my_offset)

"""
La implementación se corresponde a
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
pero adaptada a encontrar el mínimo en vez de la suma.
"""

@cuda.jit
def cuda_min_element(my_array, my_mins, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=128, dtype=numba.float32)
    shared_idx = cuda.shared.array(shape=128, dtype=numba.int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    idx = cuda.blockDim.x  * cuda.blockIdx.x + tidx
    
    # 3. Inicialiamos a infinito
    shared_mem[tidx] = np.inf
    
    # 4. Cada thread comprueba un stride del grid
    while idx < my_array.size:
        if my_array[idx] < shared_mem[tidx]:
            shared_mem[tidx] = my_array[idx]
            shared_idx[tidx] = idx
                      
        idx += cuda.blockDim.x * cuda.gridDim.x
    cuda.syncthreads()
    
    idx = cuda.blockDim.x * cuda.blockIdx.x + tidx
        
    # 5. Unroll de bloque
    # Consideramos que estamos usando 128 hebras por bloque.
    if tidx < 64:
        if shared_mem[tidx + 64] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 64]
            shared_idx[tidx] = shared_idx[tidx + 64]
    
    cuda.syncthreads()
    
    
    # 6. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
            
    # El primer thread de cada bloque indica su minimo
    # Si da para más de un bloque, luego hay que reaplicar el kernel
    if tidx == 0:
        my_mins[cuda.blockIdx.x] = shared_mem[tidx]
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]


def reduce_min_index(a):
    list_indexes = []
       
    def set_kernel_params(a):
        threads = 128
        blocks = a.size // threads + 1
        sm_size = threads * 2 * 4
        mins = np.empty(blocks, np.float32)
        indexes = np.empty(blocks, np.int32)
        return blocks, threads, 0, sm_size, a, mins, indexes
        
    def run_kernel(p):
        cuda_min_element[p[0], p[1]](p[4], p[5], p[6])
        
    p = set_kernel_params(a)
    if p[0] == 1:
        run_kernel(p)
        return p[6][0]
    else:
        while p[0] > 2:
            run_kernel(p)
            list_indexes.append(p[6])
            a = p[5]
            p = set_kernel_params(a)
        run_kernel(p)
        list_indexes.append(p[6])
        

        output_index = 0
        for indexes in list_indexes[::-1]:
            output_index = indexes[output_index]
            
        return output_index

"""
Implementación de reducción para suma, similar a la anterior
pero para realizar sumas
"""

@cuda.jit
def cuda_sum(my_array, my_sums):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=128, dtype=numba.float32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    idx = cuda.blockDim.x  * cuda.blockIdx.x + tidx
    
    # 3. Inicialiamos a cero
    shared_mem[tidx] = 0
    
    # 4. Cada thread comprueba un stride doble del grid
    while idx < my_array.size:
        shared_mem[tidx] += my_array[idx] 
                      
        idx += cuda.blockDim.x * cuda.gridDim.x
    cuda.syncthreads()
    
           
    # 5. Unroll de bloque
    # Consideramos que estamos usando 128 hebras por bloque.
    if tidx < 64:
        shared_mem[tidx] += shared_mem[tidx+64]
    
    cuda.syncthreads()
    
    # 6. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        shared_mem[tidx] += shared_mem[tidx+32]
        shared_mem[tidx] += shared_mem[tidx+16]
        shared_mem[tidx] += shared_mem[tidx+8]
        shared_mem[tidx] += shared_mem[tidx+4]
        shared_mem[tidx] += shared_mem[tidx+2]
        shared_mem[tidx] += shared_mem[tidx+1]
                     
    # El primer thread de cada bloque indica su suma
    # Si da para más de un bloque, luego hay que reaplicar el kernel
    if tidx == 0:
        my_sums[cuda.blockIdx.x] = shared_mem[tidx]

def reduce_sum(a):  
    def set_kernel_params(a):
        threads = 128
        blocks = a.size // threads + 1
        sm_size = threads * 4
        sums = np.empty(blocks, np.float32)
        return blocks, threads, 0, sm_size, a, sums
        
    def run_kernel(p):
        cuda_sum[p[0], p[1]](p[4], p[5])
        
    p = set_kernel_params(a)
    if p[0] == 1:
        run_kernel(p)
        return p[5][0]
    else:
        while p[0] > 2:
            run_kernel(p)
            a = p[5]
            p = set_kernel_params(a)
        run_kernel(p)

        return p[5][0]


"""
Multi reducción por filas. Una fila no puede tener más de 1024 elementos.
"""
@cuda.jit
def cuda_multi_min_element_64(my_array, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=64, dtype=numba.float32)
    shared_idx = cuda.shared.array(shape=64, dtype=numba.int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
  
    # 3. Inicialiamos el bloque y rellenamos con infinito
    if tidx < my_array.shape[1]:
        shared_mem[tidx] = my_array[bidx, tidx]
    else:
        shared_mem[tidx] = np.inf

    shared_idx[tidx] = tidx
        
    cuda.syncthreads()    
       
    # 4. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
    # El primer thread de cada bloque indica su minimo
    if tidx == 0:
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]

@cuda.jit
def cuda_multi_min_element_128(my_array, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=128, dtype=numba.float32)
    shared_idx = cuda.shared.array(shape=128, dtype=numba.int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
  
    # 3. Inicialiamos el bloque y rellenamos con infinito
    if tidx < my_array.shape[1]:
        shared_mem[tidx] = my_array[bidx, tidx]
    else:
        shared_mem[tidx] = np.inf

    shared_idx[tidx] = tidx
        
    cuda.syncthreads()    
    # 4. Unroll de bloque
    # Consideramos que estamos usando 128 hebras por bloque.
    if tidx < 64:
        if shared_mem[tidx + 64] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 64]
            shared_idx[tidx] = shared_idx[tidx + 64]
    
    cuda.syncthreads()
    
    # 5. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
    # El primer thread de cada bloque indica su minimo
    if tidx == 0:
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]


@cuda.jit
def cuda_multi_min_element_256(my_array, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=256, dtype=numba.float32)
    shared_idx = cuda.shared.array(shape=256, dtype=numba.int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
  
    # 3. Inicialiamos el bloque y rellenamos con infinito
    if tidx < my_array.shape[1]:
        shared_mem[tidx] = my_array[bidx, tidx]
    else:
        shared_mem[tidx] = np.inf

    shared_idx[tidx] = tidx
        
    cuda.syncthreads()    
    # 4. Unroll de bloque
    if tidx < 128:
        if shared_mem[tidx + 128] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 128]
            shared_idx[tidx] = shared_idx[tidx + 128]

    cuda.syncthreads()
    if tidx < 64:
        if shared_mem[tidx + 64] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 64]
            shared_idx[tidx] = shared_idx[tidx + 64]
    
    cuda.syncthreads()
    
    # 5. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
    # El primer thread de cada bloque indica su minimo
    if tidx == 0:
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]

@cuda.jit
def cuda_multi_min_element_512(my_array, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=512, dtype=numba.float32)
    shared_idx = cuda.shared.array(shape=512, dtype=numba.int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
  
    # 3. Inicialiamos el bloque y rellenamos con infinito
    if tidx < my_array.shape[1]:
        shared_mem[tidx] = my_array[bidx, tidx]
    else:
        shared_mem[tidx] = np.inf

    shared_idx[tidx] = tidx
        
    cuda.syncthreads()    

    # 4. Unroll de bloque
    if tidx < 256:
        if shared_mem[tidx + 256] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 256]
            shared_idx[tidx] = shared_idx[tidx + 256]
    cuda.syncthreads()

    if tidx < 128:
        if shared_mem[tidx + 128] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 128]
            shared_idx[tidx] = shared_idx[tidx + 128]

    cuda.syncthreads()
    if tidx < 64:
        if shared_mem[tidx + 64] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 64]
            shared_idx[tidx] = shared_idx[tidx + 64]
    
    cuda.syncthreads()
    
    # 5. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
    # El primer thread de cada bloque indica su minimo
    if tidx == 0:
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]

@cuda.jit
def cuda_multi_min_element_1024(my_array, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=1024, dtype=numba.float32)
    shared_idx = cuda.shared.array(shape=1024, dtype=numba.int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
  
    # 3. Inicialiamos el bloque y rellenamos con infinito
    if tidx < my_array.shape[1]:
        shared_mem[tidx] = my_array[bidx, tidx]
    else:
        shared_mem[tidx] = np.inf

    shared_idx[tidx] = tidx
        
    cuda.syncthreads()    

    # 4. Unroll de bloque
    if tidx < 512:
        if shared_mem[tidx + 512] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 512]
            shared_idx[tidx] = shared_idx[tidx + 512]
    cuda.syncthreads()

    if tidx < 256:
        if shared_mem[tidx + 256] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 256]
            shared_idx[tidx] = shared_idx[tidx + 256]
    cuda.syncthreads()

    if tidx < 128:
        if shared_mem[tidx + 128] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 128]
            shared_idx[tidx] = shared_idx[tidx + 128]

    cuda.syncthreads()
    if tidx < 64:
        if shared_mem[tidx + 64] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 64]
            shared_idx[tidx] = shared_idx[tidx + 64]
    
    cuda.syncthreads()
    
    # 5. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
    # El primer thread de cada bloque indica su minimo
    if tidx == 0:
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]

def multi_reduce_min_index(device_array):
    blocks, row_size = device_array.shape
    device_indexes = cuda.device_array(blocks)
    if 0 < row_size <= 64:
        cuda_multi_min_element_64[blocks,64](device_array, device_indexes)
    elif 64 < row_size <= 128:
        cuda_multi_min_element_128[blocks,128](device_array, device_indexes)
    elif 128 < row_size <= 256:
        cuda_multi_min_element_256[blocks, 256](device_array, device_indexes)
    elif 256 < row_size <= 512:
        cuda_multi_min_element_512[blocks, 512](device_array, device_indexes)
    elif 512 < row_size <= 1024:
        cuda_multi_min_element_1024[blocks, 1024](device_array, device_indexes)
    else:
        return np.array([reduce_min_index(device_array[i]) for i in range(blocks)])
    
    return device_indexes


@cuda.jit
def min_index_reduction(my_data, my_min, my_idx, aux, lock):
    block_mins = cuda.shared.array(shape=32, dtype=numba.float32)
    block_mins_idx = cuda.shared.array(shape=32, dtype=numba.int32)
    row = cuda.blockIdx.y
    col = cuda.threadIdx.x +cuda.blockIdx.x*cuda.blockDim.x
    if col < my_data.shape[1]:
        data1 = my_data[row, col]
    else:
        data1 = np.inf
    idx1 = col
    
    # 1. Reducción min_index en cada warp
    data2 = cuda.shfl_up_sync(-1, data1, 1)
    idx2 = cuda.shfl_up_sync(-1, idx1, 1)
    if data2 < data1:
        data1 = data2
        idx1 = idx2
            
    data2 = cuda.shfl_up_sync(-1, data1, 2)
    idx2 = cuda.shfl_up_sync(-1, idx1, 2)
    if data2 < data1:
        data1 = data2
        idx1 = idx2
            
    data2 = cuda.shfl_up_sync(-1, data1, 4)
    idx2 = cuda.shfl_up_sync(-1, idx1, 4)
    if data2 < data1:
        data1 = data2
        idx1 = idx2
            
    data2 = cuda.shfl_up_sync(-1, data1, 8)
    idx2 = cuda.shfl_up_sync(-1, idx1, 8)
    if data2 < data1:
        data1 = data2
        idx1 = idx2
            
    data2 = cuda.shfl_up_sync(-1, data1, 16)
    idx2 = cuda.shfl_up_sync(-1, idx1, 16)
    if data2 < data1:
        data1 = data2
        idx1 = idx2
    if cuda.threadIdx.x%32 == 31:
        block_mins[cuda.threadIdx.x//32] = data1
        block_mins_idx[cuda.threadIdx.x//32] = idx1
        
    cuda.syncthreads()
    
    # 2. Reducción del bloque completo en el primer warp
    if cuda.threadIdx.x < 32:
        if cuda.threadIdx.x < cuda.blockDim.x // 32:
            data2 = block_mins[cuda.threadIdx.x]
            idx2 = block_mins_idx[cuda.threadIdx.x]
        else:
            data2 = np.inf
            idx2 = col
        
        data3 = cuda.shfl_up_sync(-1, data2, 1)
        idx3 = cuda.shfl_up_sync(-1, idx2, 1)
        if data3 < data2:
            data2 = data3
            idx2 = idx3
                
        data3 = cuda.shfl_up_sync(-1, data2, 2)
        idx3 = cuda.shfl_up_sync(-1, idx2, 2)
        if data3 < data2:
            data2 = data3
            idx2 = idx3
                
        data3 = cuda.shfl_up_sync(-1, data2, 4)
        idx3 = cuda.shfl_up_sync(-1, idx2, 4)
        if data3 < data2:
            data2 = data3
            idx2 = idx3
                
        data3 = cuda.shfl_up_sync(-1, data2, 8)
        idx3 = cuda.shfl_up_sync(-1, idx2, 8)
        if data3 < data2:
            data2 = data3
            idx2 = idx3
                
        data3 = cuda.shfl_up_sync(-1, data2, 16)
        idx3 = cuda.shfl_up_sync(-1, idx2, 16)
        if data3 < data2:
            data2 = data3
            idx2 = idx3

    if cuda.threadIdx.x == 31:
        aux[cuda.blockIdx.y, cuda.blockIdx.x] = 0
        while cuda.atomic.compare_and_swap(lock, 0, 1) == 1:
            continue
        if data2 < my_min[0]:
            my_min[0] = data2
            my_idx[0] = cuda.blockIdx.y
            my_idx[1] = idx2
        cuda.threadfence()
        lock[0] = 0

@cuda.jit
def warp_based_reduce_sum(my_data, my_sum):
    block_sums = cuda.shared.array(shape=32, dtype=numba.int32)
    col = cuda.grid(1)
    if col < my_data.shape[0]:
        data1 = my_data[col]
    else:
        data1 = 0

    
    # 1. Reducción min_index en cada warp
    data2 = cuda.shfl_up_sync(-1, data1, 1)
    data1 += data2
            
    data2 = cuda.shfl_up_sync(-1, data1, 2)
    data1 += data2
            
    data2 = cuda.shfl_up_sync(-1, data1, 4)
    data1 += data2
            
    data2 = cuda.shfl_up_sync(-1, data1, 8)
    data1 += data2
            
    data2 = cuda.shfl_up_sync(-1, data1, 16)
    data1 += data2

    if cuda.threadIdx.x%32 == 31:
        block_sums[cuda.threadIdx.x//32] = data1
        
    cuda.syncthreads()
    
    # 2. Reducción del bloque completo en el primer warp
    if cuda.threadIdx.x < 32:
        if cuda.threadIdx.x < cuda.blockDim.x // 32:
            data2 = block_sums[cuda.threadIdx.x]
        else:
            data2 = 0
        
        data3 = cuda.shfl_up_sync(-1, data2, 1)
        data2 += data3
                
        data3 = cuda.shfl_up_sync(-1, data2, 2)
        data2 += data3
                
        data3 = cuda.shfl_up_sync(-1, data2, 4)
        data2 += data3
                
        data3 = cuda.shfl_up_sync(-1, data2, 8)
        data2 += data3
                
        data3 = cuda.shfl_up_sync(-1, data2, 16)
        data2 += data3

    if cuda.threadIdx.x == 31:
        cuda.atomic.add(my_sum, 0, data2)
