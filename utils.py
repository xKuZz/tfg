import numba
from numba import cuda
import numpy as np
"""
Implementación de scan para Numba CUDA.
Basado en https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
"""

def scan(x, out, MAX_TPB=128):
  
    n = x.size
    tpb = MAX_TPB 
    # Elementos por bloque
    epb = tpb * 2 
    # Número de bloques completos
    blocks = n // epb
    # Elementos en el último bloque
    elb = n % epb

    # Si sólo tenemos un bloque
    if blocks == 0 or (blocks == 1 and elb == 0):
        # Mínima potencia de 2 que incluye todos los elementos
        exponente2 = np.int(np.ceil(np.log2(elb)))
        # Total de elementos en el último bloque, teniendo en cuenta ceros extra de relleno
        total_elb = 2 ** exponente2
        # Total de hebras por bloque
        my_tpb = total_elb // 2
        # Memoria compartida
        sm_size = total_elb * x.dtype.itemsize
        
        aux = cuda.device_array(1, x.dtype)
        not_full_block_scan[1, my_tpb, 0, sm_size](x, out, aux, 0, elb, 0)
        
        return aux

    else:
        n_scans = blocks if elb == 0 else blocks + 1
        aux = cuda.device_array(n_scans, x.dtype)

        # Memoria compartida
        sm_size = epb * x.dtype.itemsize

        # Prescan de todos los bloques
        prescan[blocks, tpb, 0, sm_size](x, out, aux)

        # Prescan del último bloque, si procede
        if elb != 0:
            # Mínima potencia de 2 que incluye todos los elementos
            exponente2 = np.int(np.ceil(np.log2(elb)))
            # Total de elementos en el último bloque, teniendo en cuenta ceros extra de relleno
            total_elb = 2 ** exponente2
            # Total de hebras por bloque
            my_tpb = total_elb // 2
            # Memoria compartida
            sm_size = total_elb * x.dtype.itemsize
            
            not_full_block_scan[1, my_tpb, 0, sm_size](x, out, aux, n_scans - 1, elb, n - elb)

        d_out2 = cuda.device_array(aux.shape, aux.dtype)
        total = scan(aux, d_out2, MAX_TPB)

        scan_sum[n_scans, tpb](out, d_out2)

        return total

@cuda.jit
def scan_sum(data, aux):
    temp = cuda.shared.array(1, numba.int32)

    # Índices
    bidx = cuda.blockIdx.x 
    tidx = cuda.threadIdx.x
    eidx = cuda.grid(1) * 2
    
    # Si estamos en la primera hebra guardamos el
    # auxiliar lo acumulado del bloque anteior
    if tidx == 0:
        temp[0] = aux[bidx]

    cuda.syncthreads()
    
    # Sumamos lo acumulado del bloque anterior
    if eidx <= data.size:
        data[eidx] += aux[bidx] 

        if eidx + 1 < data.size:
            data[eidx + 1] += aux[bidx]

@cuda.jit
def prescan(data_in, data_out, aux):
    
    # Índices y memoria compartida
    shared_aux = cuda.shared.array(0, numba.int32)
    tidx = cuda.threadIdx.x 
    idx = cuda.grid(1) 
    bidx = cuda.blockIdx.x 
    blockSize = cuda.blockDim.x
    
    # Cargamos datos en memoria compartida
    shared_aux[2 * tidx] = data_in[2 * idx]
    shared_aux[2 * tidx + 1] = data_in[2 * idx + 1]
    
    offset = 1

    # Construcción del árbol up-sweepe
    d = blockSize
    while d > 0:
        cuda.syncthreads()
        
        if tidx < d:
            ai = offset * (2 * tidx + 1) - 1
            bi = offset * (2 * tidx + 2) - 1

            shared_aux[bi] += shared_aux[ai]
        offset <<= 1 
        d >>= 1 
    
    # Ponemos a cero el último elemento
    if tidx == 0:
        aux[bidx] = shared_aux[2 * blockSize - 1]
        shared_aux[2 * blockSize - 1] = 0
        
    # Contrucción del árbol down-sweepe
    b = blockSize << 1
    d = 1
    while d < b:
        offset >>= 1
        cuda.syncthreads()
        
        if tidx < d:
            ai = offset * (2 * tidx + 1) - 1
            bi = offset * (2 * tidx + 2) - 1
            
            t = shared_aux[ai]
            shared_aux[ai] = shared_aux[bi]
            shared_aux[bi] += t
            
        d <<= 1
        
    cuda.syncthreads()
    
    # Guardamos los resultados obtenidos
    data_out[2 * idx] = shared_aux[2 * tidx]
    data_out[2 * idx + 1] = shared_aux[2 * tidx + 1]


@cuda.jit
def not_full_block_scan(data_in, data_out, aux, auxidx, elb, start_idx):
    # Índices y memoria compartida
    shared_aux = cuda.shared.array(0, numba.int32)

    tidx = cuda.threadIdx.x 
    blockSize =  cuda.blockDim.x

    # Cargamos en memoria compartida si procede, si no ponemos 0.
    a = 2 * tidx
    b = 2 * tidx + 1

    shared_aux[a] = data_in[start_idx + a] if a < elb else 0
    shared_aux[b] = data_in[start_idx + b] if b < elb else 0
    
    offset = 1

    d = blockSize
    while d > 0:
        cuda.syncthreads()
        
        if tidx < d:
            ai = offset * (2 * tidx + 1) - 1
            bi = offset * (2 * tidx + 2) - 1

            shared_aux[bi] += shared_aux[ai]
        offset <<= 1 
        d >>= 1

    # Poner último elemento a 0
    if tidx == 0:
        if auxidx != -1:
            aux[auxidx] = shared_aux[2 * blockSize - 1]

        shared_aux[2 * blockSize - 1] = 0
        
   
    d = 1
    c = blockSize << 1
    while d < c: 
        offset >>= 1
        cuda.syncthreads()
        
        if tidx < d:
            ai = offset * (2 * tidx + 1) - 1
            bi = offset * (2 * tidx + 2) - 1
            
            t = shared_aux[ai]
            shared_aux[ai] = shared_aux[bi]
            shared_aux[bi] += t
            
        d <<= 1
        
    cuda.syncthreads()
    
    # Escribimos salida
    if a < elb:
        data_out[start_idx + a] = shared_aux[a]
    if b < elb:
        data_out[start_idx + b] = shared_aux[b]

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
        threads = min(a.size, 128)
        blocks = a.size // threads
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
        while p[0] > 1:
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
  