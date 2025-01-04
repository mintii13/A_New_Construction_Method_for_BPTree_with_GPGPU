import time
from typing import Optional, List, Tuple
import csv

def read_csv_data(file_path):
    """Đọc dữ liệu từ file CSV và trả về danh sách các key"""
    keys = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Giả sử mỗi dòng chứa một số duy nhất, hãy chuyển nó thành int
            keys.append(int(row[0]))
    return keys

class Node:
    def __init__(self, is_leaf=True):
        self.keys = []
        self.children = []
        self.is_leaf = is_leaf
        self.next = None

class BPlusTree:
    def __init__(self, order=3):
        self.root = None
        self.order = order
    
    def get_max_key_from_subtree(self, node):
        if node.is_leaf:
            return node.keys[-1]
        return self.get_max_key_from_subtree(node.children[-1])
    
    def batch_construct(self, initial_keys):
        start_time = time.time()
        
        # Step 1: Sort keys
        start_time_sort = time.time()
        keys = sorted(initial_keys)
        time_sort = time.time() - start_time_sort
        print(f"Sorting time:                        {time_sort:.4f} seconds")
        # print("\nSorted keys:", keys)
        
        # Step 2: Create leaf nodes
        leaf_nodes = []
        i = 0
        while i < len(keys):
            node = Node(is_leaf=True)
            node.keys = keys[i:i+self.order]
            leaf_nodes.append(node)
            # print(f"Created leaf node: {node.keys}")
            i += self.order
            
        # Link leaf nodes
        for i in range(len(leaf_nodes)-1):
            leaf_nodes[i].next = leaf_nodes[i+1]
        
        def build_internal_level(current_level_nodes):
            if len(current_level_nodes) <= 1:
                return current_level_nodes

            next_level = []
            i = 0
            
            while i < len(current_level_nodes):
                node = Node(is_leaf=False)
                end = min(i + self.order + 1, len(current_level_nodes))
                node.children = current_level_nodes[i:end]
                
                for j in range(i, min(i + self.order, len(current_level_nodes))):
                    max_key = self.get_max_key_from_subtree(current_level_nodes[j])
                    node.keys.append(max_key)
                
                next_level.append(node)
                i += self.order + 1
            
            return build_internal_level(next_level)
        
        # Build all levels
        self.root = build_internal_level(leaf_nodes)[0]
        
        construction_time = time.time() - start_time
        print(f"Construction time (include sorting): {construction_time:.4f} seconds")
    
    def search(self, key: int) -> Tuple[bool, List[Node]]:
        """
        Search for a key in the B+ tree
        Returns: (found: bool, path: List[Node])
        """
        if not self.root:
            return False, []
        
        path = []
        current = self.root
        
        # Traverse down the tree
        while not current.is_leaf:
            path.append(current)
            
            # Find the appropriate child
            for i, k in enumerate(current.keys):
                if key <= k:
                    current = current.children[i]
                    break
            else:
                current = current.children[-1]
        
        path.append(current)
        return key in current.keys, path

    def print_tree(self):
        def print_node(node, level=0):
            indent = "  " * level
            if node.is_leaf:
                print(f"{indent}Leaf: {node.keys}")
            else:
                print(f"{indent}Internal: {node.keys}")
                for child in node.children:
                    print_node(child, level + 1)
                    
        print("\nFull B+ Tree structure:")
        print_node(self.root)

def verify_tree(tree: BPlusTree, all_keys: List[int]):
    """Verify the correctness of the B+ tree"""
    start_time = time.time()
    errors = []
    
    # Test 1: Search for all keys that should be in the tree
    for key in all_keys:
        found, path = tree.search(key)
        if not found:
            errors.append(f"Key {key} should be in tree but wasn't found")
        # Verify path
        if not path[-1].is_leaf:
            errors.append(f"Search for key {key} didn't end at a leaf node")
    
    # Test 2: Search for some keys that shouldn't be in the tree
    not_in_tree = [-1, max(all_keys) + 1]
    for key in not_in_tree:
        found, _ = tree.search(key)
        if found:
            errors.append(f"Key {key} shouldn't be in tree but was found")
    
    # Test 3: Verify leaf node links
    def get_leaf_nodes(node):
        if node.is_leaf:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(get_leaf_nodes(child))
        return leaves
    
    leaves = get_leaf_nodes(tree.root)
    for i in range(len(leaves)-1):
        if leaves[i].next != leaves[i+1]:
            errors.append(f"Leaf link broken between nodes {leaves[i].keys} and {leaves[i+1].keys}")
    
    verification_time = time.time() - start_time
    print(f"\nTree verification time: {verification_time:.4f} seconds")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nAll verifications passed! Tree is correct.")

if __name__ == "__main__":
    size_keys = '9000000'
    order = 10
    file_path = r"DataSet/" + size_keys + r"_unique_positive_integers.csv"

    # Đọc file và lưu các key
    start_time_read = time.time()
    initial_keys = read_csv_data(file_path)
    end_time_read = time.time()
    print(f"Read file {size_keys} keys with m = {order}:  {end_time_read - start_time_read:.4f} seconds")
    # initial_keys = list(range(1, 9000000))
    # print("Initial keys:", initial_keys)

    # Build tree
    tree = BPlusTree(order - 1)
    tree.batch_construct(initial_keys)
    # tree.print_tree()

    # # Verify tree
    # verify_tree(tree, initial_keys)

    # # Test some searches
    # test_keys = [1, 12, 100, 45, 192]
    # print("\nTesting individual searches:")
    # for key in test_keys:
    #     start_time = time.time()
    #     found, path = tree.search(key)
    #     search_time = time.time() - start_time
    #     print(f"Searching for {key}: {'Found' if found else 'Not found'} "
    #           f"(depth={len(path)}, time={search_time:.6f}s)")
    #     print(f"Path: {' -> '.join(str(node.keys) for node in path)}")