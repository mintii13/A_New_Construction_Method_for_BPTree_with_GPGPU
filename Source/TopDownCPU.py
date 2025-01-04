import bisect
import csv
import time

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
    """Node của B+ Tree"""
    def __init__(self, order):
        self.order = order
        self.keys = []
        self.children = []
        self.is_leaf = True  # Node này có phải là node lá hay không
        self.parent = None  # Tham chiếu đến node cha

    def split(self):
        """Chia node khi số lượng khóa vượt quá giới hạn"""
        mid_index = len(self.keys) // 2
        mid_key = self.keys[mid_index]

        right_node = Node(self.order)
        right_node.is_leaf = self.is_leaf
        right_node.keys = self.keys[mid_index + 1:]  # Lấy phần sau mid_key
        if not self.is_leaf:
            right_node.children = self.children[mid_index + 1:]  # Cập nhật children
            for child in right_node.children:
                child.parent = right_node

        # Cắt node hiện tại
        self.keys = self.keys[:mid_index]
        self.children = self.children[:mid_index + 1] if not self.is_leaf else []

        return mid_key, right_node


class BPlusTree:
    """Cây B+ với các thao tác insert, search và print"""
    def __init__(self, order):
        self.order = order
        self.root = Node(order)

    def insert(self, key):
        """Chèn một khóa vào cây"""
        # Tìm node lá để chèn
        leaf = self._find_leaf_node(self.root, key)
        self._insert_in_leaf(leaf, key)

        # Nếu node lá đầy, cần chia
        if len(leaf.keys) >= self.order:
            mid_key, right_node = leaf.split()
            if leaf == self.root:
                # Tạo node cha mới
                new_root = Node(self.order)
                new_root.is_leaf = False
                new_root.keys.append(mid_key)
                new_root.children.append(leaf)
                new_root.children.append(right_node)
                leaf.parent = new_root
                right_node.parent = new_root
                self.root = new_root
            else:
                # Chèn mid_key vào node cha
                self._insert_in_parent(leaf, mid_key, right_node)

    def _find_leaf_node(self, node, key):
        """Tìm node lá chứa key"""
        if node.is_leaf:
            return node
        else:
            pos = bisect.bisect_right(node.keys, key)
            pos = min(pos, len(node.children) - 1)  # Đảm bảo không vượt quá giới hạn
            return self._find_leaf_node(node.children[pos], key)

    def _insert_in_leaf(self, leaf, key):
        """Chèn khóa vào node lá"""
        bisect.insort(leaf.keys, key)

    def _insert_in_parent(self, node, mid_key, right_node):
        """Chèn mid_key vào node cha của node"""
        parent = node.parent
        if parent is None:
            # Nếu không có cha (root bị chia), tạo node cha mới
            new_root = Node(self.order)
            new_root.is_leaf = False
            new_root.keys.append(mid_key)
            new_root.children.append(node)
            new_root.children.append(right_node)
            node.parent = new_root
            right_node.parent = new_root
            self.root = new_root
            return

        # Chèn mid_key và cập nhật children trong parent
        pos = bisect.bisect_right(parent.keys, mid_key)
        parent.keys.insert(pos, mid_key)
        parent.children.insert(pos + 1, right_node)
        right_node.parent = parent

        # Nếu parent đầy, tiếp tục chia
        if len(parent.keys) >= self.order:
            mid_key, new_right = parent.split()
            self._insert_in_parent(parent, mid_key, new_right)

    def print_tree(self):
        """In cấu trúc của cây"""
        self._print_tree(self.root, 0)

    def _print_tree(self, node, level):
        print(f"Level {level} : {node.keys}")
        if not node.is_leaf:
            for child in node.children:
                self._print_tree(child, level + 1)

    def search(self, key):
        """Tìm một khóa trong cây"""
        leaf = self._find_leaf_node(self.root, key)
        return key in leaf.keys


def main():
    size_keys = '900000'
    order = 10
    file_path = r"D:/FPTU-sourse/Term3/ResFes/Python/DataSet/" + size_keys + r"_unique_positive_integers.csv"

    # Đọc file và lưu các key
    start_time_read = time.time()
    keys = read_csv_data(file_path)
    end_time_read = time.time()
    print(f"Thời gian đọc file {size_keys} keys: {end_time_read - start_time_read:.4f} giây")
    # Tạo cây B+ Tree
    bpt = BPlusTree(order)

    # Bắt đầu tính thời gian chạy toàn bộ chương trình
    start_time_total = time.time()

    # Bắt đầu chèn các key vào cây và đo thời gian xây dựng cây
    start_time_insert = time.time()
    for key in keys:
        bpt.insert(key)
    end_time_insert = time.time()
    print(f"Thời gian xây dựng cây B+ không tính đọc File: {end_time_insert - start_time_insert:.4f} giây")

    # In cấu trúc cây nếu cần (bỏ comment dòng sau để in cây)
    # bpt.print_tree()

    # Kết thúc tính thời gian toàn bộ chương trình
    end_time_total = time.time()
    print(f"Thời gian chạy toàn bộ chương trình: {end_time_total - start_time_total:.4f} giây")


if __name__ == "__main__":
    main()

