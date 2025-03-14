# Reshape (final) + no Padding + read File [UI]
import cupy as cp
import numpy as np
from numba import cuda, int32
import math
import time
import csv


def read_csv_data(file_path):
    """Đọc dữ liệu từ file CSV và trả về danh sách các key."""
    keys = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0].strip().isdigit():
                    keys.append(np.int32(row[0]))
    except FileNotFoundError:
        print(f"Error: File không tồn tại tại đường dẫn {file_path}")
    except ValueError as e:
        print(f"Error: Không thể chuyển đổi giá trị sang số nguyên - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return keys

# @cuda.jit
# def reshape_1d_to_2d_kernel(input_array, output_array, n):
#     idx = cuda.grid(1)
#     if idx < input_array.size:
#         row = idx // n
#         col = idx % n
#         if row < output_array.shape[0] and col < output_array.shape[1]:
#             output_array[row, col] = input_array[idx]

@cuda.jit
def insert_parallel_3d_kernel(output_array, sorted_keys, m, level_leaf):
    l = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if l < output_array.shape[0] and i < output_array.shape[1] and k < output_array.shape[2]:
        if l < level_leaf:  # Các node internal
            position = (i * m + k + 1) * (m - 1) * m**(level_leaf - l - 1)
            if 0 < position <= len(sorted_keys):
                output_array[l, i, k] = sorted_keys[position - 1]
            else:
                output_array[l, i, k] = -1
        else:  # Level cuối (node lá)
            idx = i * (m-1) + k  # Vị trí trong sorted_keys
            if idx < len(sorted_keys):
                output_array[l, i, k] = sorted_keys[idx]
            else:
                output_array[l, i, k] = -1

# @cuda.jit
# def copy_leaf_nodes_kernel(sorted_keys, array, level, m):
#     # Tính vị trí node và key trong node
#     idx = cuda.grid(1)
#     if idx < len(sorted_keys):
#         node_idx = idx // (m-1)  # Số node lá
#         key_idx = idx % (m-1)    # Vị trí key trong node
#         if node_idx < array.shape[1]:  # Kiểm tra không vượt quá số node
#             array[level, node_idx, key_idx] = sorted_keys[idx]

@cuda.jit
def convert_3d_to_2d_kernel(array_3d, array_2d, m, h, num_keys):
    l = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if l < array_3d.shape[0] and i < array_3d.shape[1] and k < array_3d.shape[2]:
        node_index = i + int(math.ceil(((math.ceil(num_keys / ((m - 1) * (m ** (h - 1 - l))))) - 1) / (m - 1)))
        if node_index < array_2d.shape[0] and array_3d[l, i, k] != -1:
            array_2d[node_index, k] = array_3d[l, i, k]


def main():
    num_keys = int(num_keys_str)
    file_path = "DataSet/" + num_keys_str + "_unique_positive_integers.csv"

    start_time_read = time.time()
    internal_keys = read_csv_data(file_path)
    end_time_read = time.time()
    print(f"Thời gian đọc file {num_keys} keys: {end_time_read - start_time_read:.6f} giây")

    # internal_keys = np.arange(1, num_keys + 1, dtype=np.int32)
    
    # Chiều cao của cây
    # h = math.ceil(math.log(num_keys / (m - 1), m)) + 1
    h = math.log(len(internal_keys) / (m - 1), m)
    if (h - math.floor(h)) > 0 and (h - math.floor(h)) < 0.0000001:
      h = math.ceil(h)
    else:
      h = math.ceil(h) +1
    # Số keys mỗi Node
    num_keys_per_node = m - 1

    # Khởi tạo mảng 3 chiều cho B+ Tree TRONG GPU: (cấp độ, Node trong cấp độ, keys trong Node)
    bptree_array = cp.zeros((h, m**(h - 1), num_keys_per_node), dtype=np.int32)

    # Khởi tạo mảng 2 chiều cho B+ Tree: (Leaf Node + Internal Node, keys trong Node)
    bptree_array_2d = cp.zeros((bptree_array.shape[1] + math.ceil(((num_keys) / (m - 1) - 1) / (m - 1)), num_keys_per_node), dtype=np.int32)

    # Bắt đầu tính thời gian toàn bộ chương trình
    start_time_total = time.time()

    # Sắp xếp mảng keys bằng Cupy
    start_time_sort = time.time()
    internal_keys_gpu = cp.asarray(internal_keys)  # Chuyển keys sang mảng Cupy
    internal_keys_gpu.sort()  # Sắp xếp trên GPU
    end_time_sort = time.time()
    print(f"Thời gian sắp xếp {num_keys} keys: {end_time_sort - start_time_sort:.4f} giây, m = {m}")

    # Reshape 1D to 2D on GPU
    # start_time_reshape = time.time()
    # n = num_keys_per_node
    # m_rows = (len(internal_keys) + n - 1) // n
    # output_gpu = cuda.device_array((m_rows, n), dtype=np.int32)
    # threads_per_block = 256
    # blocks_per_grid = (len(internal_keys) + threads_per_block - 1) // threads_per_block
    # reshape_1d_to_2d_kernel[blocks_per_grid, threads_per_block](internal_keys_gpu, output_gpu, n)
    # cuda.synchronize()
    # leaf_keys = output_gpu.copy_to_host()
    # end_time_reshape = time.time()
    # print(f"Thời gian reshape 1D sang 2D: {end_time_reshape - start_time_reshape:.4f} giây")

    # Insert parallel 3D on GPU
    start_time_insert = time.time()
    threadsperblock = (8, 8, 8)
    blockspergrid_x = max(1, int(np.ceil(bptree_array.shape[1] / threadsperblock[0])))
    blockspergrid_y = max(1, int(np.ceil(bptree_array.shape[2] / threadsperblock[1])))
    blockspergrid_z = max(1, int(np.ceil(bptree_array.shape[0] / threadsperblock[2])))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    insert_parallel_3d_kernel[blockspergrid, threadsperblock](bptree_array, internal_keys_gpu, m, h - 1)
    cuda.synchronize()
    end_time_insert = time.time()
    print(f"Thời gian chèn song song vào mảng 3D: {end_time_insert - start_time_insert:.4f} giây")

    # start_time_copy = time.time()
    # threads_per_block = 256
    # blocks_per_grid = (len(internal_keys_gpu) + threads_per_block - 1) // threads_per_block
    # copy_leaf_nodes_kernel[blocks_per_grid, threads_per_block](internal_keys_gpu, bptree_array, h-1, m)
    # cuda.synchronize()
    # end_time_copy = time.time()
    # print(f"Thời gian sao chép Node lá: {end_time_copy - start_time_copy:.4f} giây")

    # Convert 3D to 2D on GPU
    start_time_convert = time.time()
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        int(math.ceil(bptree_array.shape[1] / threads_per_block[0])),
        int(math.ceil(bptree_array.shape[2] / threads_per_block[1])),
        int(math.ceil(bptree_array.shape[0] / threads_per_block[2]))
    )
    convert_3d_to_2d_kernel[blocks_per_grid, threads_per_block](bptree_array, bptree_array_2d, m, h, num_keys)
    cuda.synchronize()
    result_array_2d = cp.asnumpy(bptree_array_2d)
    end_time_convert = time.time()
    print(f"Thời gian chuyển đổi mảng 3D sang 2D: {end_time_convert - start_time_convert:.4f} giây")

    # Tổng thời gian xây dựng cây
    end_time_total = time.time()
    print(f"Thời gian toàn bộ chương trình không tính đọc file: {end_time_total - start_time_total:.4f} giây")

    # Hiển thị kết quả
    # print(result_array_2d)


if __name__ == "__main__":
    num_keys_str = '17715610'
    m = 11
    # num_keys = 9000000
    main()