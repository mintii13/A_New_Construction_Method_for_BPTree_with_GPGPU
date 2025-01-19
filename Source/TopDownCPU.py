import bisect
import csv
import time
import numpy as np

def read_csv_data(file_path):
    """Đọc dữ liệu từ file CSV và trả về danh sách các key"""
    keys = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            keys.append(int(row[0]))
    return keys

class Node:
    def __init__(self, order):
        self.order = order
        self.keys = []
        self.children = []
        self.is_leaf = True
        self.parent = None
        self.next = None

    def split(self):
        mid_index = len(self.keys) // 2
        mid_key = self.keys[mid_index]

        right_node = Node(self.order)
        right_node.is_leaf = self.is_leaf

        if self.is_leaf:
            right_node.keys = self.keys[mid_index:]
            self.keys = self.keys[:mid_index]
            right_node.next = self.next
            self.next = right_node
        else:
            right_node.keys = self.keys[mid_index + 1:]
            right_node.children = self.children[mid_index + 1:]
            for child in right_node.children:
                child.parent = right_node
            self.keys = self.keys[:mid_index]
            self.children = self.children[:mid_index + 1]

        return mid_key, right_node

class BPlusTree:
    def __init__(self, order):
        self.order = order
        self.root = Node(order)
        self.head = None

    def insert(self, key):
        leaf = self._find_leaf_node(self.root, key)
        self._insert_in_leaf(leaf, key)

        if len(leaf.keys) >= self.order:
            mid_key, right_node = leaf.split()
            
            if leaf == self.root:
                new_root = Node(self.order)
                new_root.is_leaf = False
                new_root.keys.append(mid_key)
                new_root.children.append(leaf)
                new_root.children.append(right_node)
                leaf.parent = new_root
                right_node.parent = new_root
                self.root = new_root
            else:
                self._insert_in_parent(leaf, mid_key, right_node)
        
        if self.head is None:
            self.head = leaf

    def _find_leaf_node(self, node, key):
        if node.is_leaf:
            return node
        
        pos = bisect.bisect_right(node.keys, key)
        pos = min(pos, len(node.children) - 1)
        return self._find_leaf_node(node.children[pos], key)

    def _insert_in_leaf(self, leaf, key):
        bisect.insort(leaf.keys, key)

    def _insert_in_parent(self, node, mid_key, right_node):
        parent = node.parent
        if parent is None:
            new_root = Node(self.order)
            new_root.is_leaf = False
            new_root.keys.append(mid_key)
            new_root.children.append(node)
            new_root.children.append(right_node)
            node.parent = new_root
            right_node.parent = new_root
            self.root = new_root
            return

        pos = bisect.bisect_right(parent.keys, mid_key)
        parent.keys.insert(pos, mid_key)
        parent.children.insert(pos + 1, right_node)
        right_node.parent = parent

        if len(parent.keys) >= self.order:
            mid_key, new_right = parent.split()
            self._insert_in_parent(parent, mid_key, new_right)

    def print_tree(self):
        print("\nCấu trúc cây B+:")
        self._print_tree(self.root, 0)
        print()

    def _print_tree(self, node, level):
        indent = "  " * level
        print(f"{indent}Level {level}: keys={node.keys}")
        if not node.is_leaf:
            for i, child in enumerate(node.children):
                print(f"{indent}Child {i}:")
                self._print_tree(child, level + 1)

    def print_linked_list(self):
        print("Linkedlist các key lá:")
        current = self.head
        while current is not None:
            print(current.keys, end=" -> ")
            current = current.next
        print("None\n")

def main():
    bpt = BPlusTree(order)
    
    start_time_insert = time.time()
    
    for key in keys:
        bpt.insert(key)
    
    end_time_insert = time.time()
    print(f"Thời gian xây dựng cây: {end_time_insert - start_time_insert:.4f} giây")
    
    # print tree after construction
    # print("\nCấu trúc cuối cùng của cây:")
    # bpt.print_tree()
    # print("Linkedlist các key lá cuối cùng:")
    # bpt.print_linked_list()

if __name__ == "__main__":
    # test with input file
    size_keys = '900000'
    order = 10
    file_path = r"D:/FPTU-sourse/Term3/ResFes/Python/DataSet/" + size_keys + r"_unique_positive_integers.csv"
    # Đọc file và lưu các key
    start_time_read = time.time()
    keys = read_csv_data(file_path)
    end_time_read = time.time()
    print(f"Thời gian đọc file {size_keys} keys: {end_time_read - start_time_read:.4f} giây")

    # test with input list
    # num_keys = 15  # Giảm số lượng key để dễ theo dõi
    # order = 4   # Giảm order để dễ thấy việc split
    # keys = list(range(1, num_keys + 1))
    main()