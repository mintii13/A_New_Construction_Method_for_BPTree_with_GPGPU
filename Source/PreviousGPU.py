import math
import time
from numba import cuda
import numpy as np
import csv
import cupy as cp

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

@cuda.jit
def create_leaf_nodes(K, nodeNum, m, nodes):
    #node[tid, 1] = kNum
    #node[tid, 2] = ptrs
    #node[tid, 3] = type
    #node[tid, 4] = parentNode
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < nodeNum:
        # Lưu trữ order và kNum trong mảng nodes
        nodes[tid, 0] = order  # Giá trị order
        if tid < nodeNum - 1:
            nodes[tid, 1] = m  # Các node bình thường có m keys
        else:
            remaining_keys = len(K) - (tid * m)
            nodes[tid, 1] = min(m, remaining_keys)  # Các node cuối cùng có số keys nhỏ hơn
        nodes[tid, 2] = -1      #ptrs
        nodes[tid, 3] = 0       #type
        nodes[tid, 4] = -1      #parent node

# #Step 2
# def bulk_insert_keys(keys, nodeNum, m, Tdm):
# # Số lượng threads per block và số blocks
#     threads_per_block = 256
#     blocks_per_grid = (nodeNum + (threads_per_block - 1)) // threads_per_block

#     # Chuyển dữ liệu lên device
#     d_K = cuda.to_device(K)
#     d_nodes = cuda.device_array((nodeNum, 5), dtype=np.int32)

#     # Tính thời gian thực thi trên GPU
#     start_gpu = time.time()

#     # Gọi kernel với cấu hình blocks_per_grid và threads_per_block
#     create_leaf_nodes[blocks_per_grid, threads_per_block](d_K, nodeNum, m, d_nodes)

#     # Đảm bảo GPU hoàn thành công việc trước khi lấy kết quả
#     cuda.synchronize()

#     # Lấy kết quả về từ device
#     d_nodes.copy_to_host(nodes)

#     # Tính thời gian GPU
#     print(f"GPU Time: {time.time() - start_gpu}")

@cuda.jit
def bulk_insert_keys(K, nodeNum, m, Tdm):    
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if tid < len(K):
        # Tính node_index cho key hiện tại
        node_index = tid // m
        
        # Tính node_loc: vị trí trong node để insert key
        node_loc = tid % m
        
        # Insert key vào vị trí tính được
        if node_index < nodeNum:
            Tdm[node_index][node_loc] = K[tid]

def bulk_insert_keys_kernel(nodeNum, m, K):        
    # Khởi tạo Tdm array với mỗi node có thể chứa tối đa m keys
    Tdm = np.zeros((nodeNum, m), dtype=np.int32)
    
    # Setup và launch kernel
    threads_per_block = 256
    blocks_per_grid = (len(K) + threads_per_block - 1) // threads_per_block
    
    # Chuyển dữ liệu lên GPU
    d_keys = cuda.to_device(K)
    d_Tdm = cuda.to_device(Tdm)
    
    # Launch kernel
    start = time.time()
    bulk_insert_keys[blocks_per_grid, threads_per_block](d_keys, nodeNum, m, d_Tdm)
    # Copy kết quả về
    Tdm = d_Tdm.copy_to_host()
    return Tdm

@cuda.jit
def compute_max_keys(nodes, Tdm, max_keys, level_start, level_size, m):
    """
    Computes maximum keys for internal nodes
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if tid < level_size:
        node_idx = level_start + tid
        if nodes[node_idx, 3] == 1:  # if internal node
            children_start = nodes[node_idx, 2]
            num_children = nodes[node_idx, 1] + 1
            
            # For each key position
            for i in range(nodes[node_idx, 1]):
                max_key = 0
                child_idx = children_start + i
                
                # Find maximum key in child node
                if nodes[child_idx, 3] == 0:  # leaf node
                    for j in range(nodes[child_idx, 1]):
                        max_key = max(max_key, Tdm[child_idx, j])
                else:  # internal node
                    max_key = max_keys[child_idx]
                
                max_keys[node_idx] = max_key

@cuda.jit
def initialize_leaf_nodes(nodes, Tdm, nodeNum, m):
    """Khởi tạo các leaf nodes"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < nodeNum:
        # Set basic properties for leaf nodes
        nodes[tid, 0] = m + 1  # order
        nodes[tid, 1] = m if tid < nodeNum - 1 else cuda.gridDim.x  # number of keys
        nodes[tid, 2] = -1     # no children for leaf nodes
        nodes[tid, 3] = 0      # type = leaf
        nodes[tid, 4] = -1     # parent (will be set later)

@cuda.jit
def build_internal_nodes(nodes, nodeNum, m, cur_level_start, next_level_start, level_size):
    """Xây dựng internal nodes cho một level"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if tid < level_size:
        node_idx = next_level_start + tid
        first_child = cur_level_start + tid * (m + 1)
        num_children = min(m + 1, cur_level_start + level_size - first_child)
        
        # Set properties for internal node
        nodes[node_idx, 0] = m + 1  # order
        nodes[node_idx, 1] = num_children - 1  # number of keys = number of children - 1
        nodes[node_idx, 2] = first_child  # pointer to first child
        nodes[node_idx, 3] = 1  # type = internal
        nodes[node_idx, 4] = -1  # parent (will be set in next level)
        
        # Set parent pointers for children
        for i in range(num_children):
            child_idx = first_child + i
            if child_idx < nodeNum:
                nodes[child_idx, 4] = node_idx

def calculate_tree_parameters(nodeNum, m):
    """Tính toán các thông số của cây"""
    height = 1
    total_nodes = nodeNum
    level_size = nodeNum
    level_starts = [0]
    level_sizes = [nodeNum]
    
    while level_size > 1:
        level_size = (level_size + m) // (m + 1)
        level_starts.append(total_nodes)
        level_sizes.append(level_size)
        total_nodes += level_size
        height += 1
        
    return height, total_nodes, level_starts, level_sizes

def build_tree_structure(nodes, Tdm, nodeNum, m):
    """Xây dựng cấu trúc cây B+ hoàn chỉnh"""
    # Tính toán thông số cây
    height, total_nodes, level_starts, level_sizes = calculate_tree_parameters(nodeNum, m)
    
    # Tạo mảng nodes mới với kích thước đủ cho tất cả nodes
    new_nodes = np.zeros((total_nodes, 5), dtype=np.int32)
    
    # Copy mảng nodes cũ vào mảng mới
    new_nodes[:len(nodes)] = nodes
    
    # Chuyển dữ liệu lên GPU
    d_nodes = cuda.to_device(new_nodes)
    
    # Khởi tạo leaf nodes
    threads_per_block = 256
    blocks_per_grid = (nodeNum + threads_per_block - 1) // threads_per_block
    initialize_leaf_nodes[blocks_per_grid, threads_per_block](d_nodes, Tdm, nodeNum, m)
    
    # Xây dựng các internal levels
    for level in range(height - 1):
        cur_level_start = level_starts[level]
        cur_level_size = level_sizes[level]
        next_level_start = level_starts[level + 1]
        next_level_size = level_sizes[level + 1]
        
        blocks_per_grid = (next_level_size + threads_per_block - 1) // threads_per_block
        
        build_internal_nodes[blocks_per_grid, threads_per_block](
            d_nodes, total_nodes, m, cur_level_start, next_level_start, cur_level_size)
    
    # Copy kết quả về host
    new_nodes = d_nodes.copy_to_host()
    return new_nodes

def complete_tree_construction(K, m, nodeNum, nodes):
    """Hàm chính để xây dựng cây B+"""
    nodeNum = math.ceil((len(K) + m - 1) // m)
    
    # Khởi tạo nodes và Tdm
    nodes = np.zeros((nodeNum, 5), dtype=np.int32)
    Tdm = bulk_insert_keys_kernel(nodeNum, m, K)  # Hàm của bạn để insert keys
    # Xây dựng cấu trúc cây hoàn chỉnh
    final_nodes = build_tree_structure(nodes, Tdm, nodeNum, m)
    
    return final_nodes


def main():
    key_size = int(key_size_str)
    file_path = "DataSet/" + key_size_str + "_unique_positive_integers.csv"
    start_time_read = time.time()
    keys = read_csv_data(file_path)
    end_time_read = time.time()
    print(f"Thời gian đọc file {key_size} Keys: {end_time_read - start_time_read:.6f} giây")
    
    # Chiều cao của cây
    # h = math.ceil(math.log(num_keys / (m - 1), m)) + 1
    # Bắt đầu tính thời gian toàn bộ chương trình
    start_time_total = time.time()

    start_time_copy = time.time()
    keys_gpu = cp.asarray(keys)  # Chuyển keys sang mảng Cupy
    time_copy = time.time() - start_time_copy
    print(f"Thời gian copy sang GPU: {time.time() - start_time_copy :.4f}")
    start_time_sort = time.time()
    keys_gpu.sort()  # Sắp xếp trên GPU
    end_time_sort = time.time()
    time_sort = end_time_sort - start_time_sort
    print(f"Thời gian sắp xếp: {end_time_sort - start_time_sort:.4f} giây, order = {m + 1}")

    nodeNum = math.ceil((len(keys) + m - 1) // m)
    start_time_build = time.time()
    # Mảng chứa các node, mỗi node là một mảng 2 giá trị: order và kNum
    nodes = np.zeros((nodeNum, 5), dtype=np.int32)
    complete_tree_construction(keys, m, nodeNum, nodes)
    print(f"Thời gian copy to GPU and sort: {time_copy + time_sort :.4f}")
    print(f"Thời gian level-build: {time.time() - start_time_build :.4f}")
    print(f"Thời gian chạy cả chương trình không tính đọc file: {time.time() - start_time_total :.4f}")
    
if __name__ == "__main__":
    key_size_str = '194871710'
    order = 11
    m = order - 1
    # num_keys = 9000000
    main()