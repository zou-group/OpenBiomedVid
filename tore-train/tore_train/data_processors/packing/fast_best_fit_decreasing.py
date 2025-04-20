"""
This module implements a fast version of the Best Fit Decreasing algorithm for bin packing.
It uses a segment tree data structure to efficiently find the best bin for each item.
The algorithm is optimized for performance and can handle large-scale bin packing problems.

Author: Qingyang Wu
"""

import tqdm

class FastBestFitTree:
    def __init__(self, bin_capacity):
        self.bin_capacity = bin_capacity

        self.num_items = 0  # Current number of items
        self.num_bins = 0  # Current number of bins
        
        self.item_to_bin = []  # Item->bin matching: item_to_bin[i] = bin of i-th item
        self.bin_to_items = []  # Bin->item matching: bin_to_items[i] = list of items in i-th bin

        self.bins_remaining_capacity = []  # Bin remaining sizes used by segment tree
        self.segment_tree = []  # Segment tree for maximum range queries on the remaining bin spaces

    def expand_capacity(self, length):
        """
        Expands the capacity of the bin_remaining_sizes and segment_tree to accommodate the given length.

        Parameters:
        length (int): The new length to expand to.

        This method ensures that the bin_remaining_sizes list and the segment_tree list are expanded to the specified length.
        New bins will be initialized with full capacity.
        """
        # expand the bin_remaining_sizes
        for _ in range(len(self.bins_remaining_capacity), length):
            self.bins_remaining_capacity.append(self.bin_capacity)  # New bins will be empty with full capacity
        
        # expand the segment tree based on the current height
        current_height = len(self.segment_tree).bit_length() - 1
        new_height = self.height
        if new_height > current_height:
            for _ in range(len(self.segment_tree), 2 ** (new_height + 1) - 1):
                self.segment_tree.append(self.bin_capacity)  # These values will be overwritten by build

        # build the new segment tree from root
        self.build(node=0, height=0)

    def build(self, node, height):
        """
        Build the segment tree with the bins remaining capacity. Below is an example of the segment tree with index in array:
            Bins remaining capacity: [30, 20, 40, 10]
            Segment Tree with index in array [index, value]:

                                        [0, 40]
                                        /     \
                                    [1, 30]   [2, 40]
                                    /   \      /    \
                            [3, 30] [4, 20] [5, 40] [6, 10]
            
            (Bin index)       0        1        2       3
        The time complexity of build is O(n), where n is the number of bins.
        """
        left = 2 * node + 1
        right = 2 * node + 2
        if height == self.height - 1:
            # if the current node is a leaf node, 
            self.segment_tree[node] = self.bins_remaining_capacity[node - (2 ** height - 1)]
        else:
            # build the left and right child
            self.build(left, height + 1)
            self.build(right, height + 1)
            # BST, the current node's value is bigger than the children
            self.segment_tree[node] = max(self.segment_tree[left], self.segment_tree[right])

    def update(self, bin_index, value):
        """
        Update the segment tree with the bin index and value.
        """
        self.update_tree(node=0, height=0, bin_index=bin_index, start=0, end=len(self.bins_remaining_capacity) - 1, value=value)

    def update_tree(self, node, height, bin_index, start, end, value):
        """
        Update the segment tree with the new bins remaining capacity. Below is an example of the segment tree with index in array:
            Bins remaining capacity: [30, 20, 40, 10]
            Segment Tree with index in array [index, value]:

                                        [0, 40]*
                                        /     \
                                    [1, 30]   [2, 40]*
                                    /   \      /    \
                            [3, 30] [4, 20] [5, 40]* [6, 10]
            
            (Bin index)       0        1        2*       3

        We only update the segment tree with the paths with *. 
        We can find the path by comparing the bin_index with the mid_bin_index to get the mapping of left and right child.
        The time complexity of update is O(log n), where n is the number of bins.
        """
        if height == self.height - 1:
            self.bins_remaining_capacity[bin_index] = self.segment_tree[node] = value
        else:
            left = 2 * node + 1
            right = 2 * node + 2

            mid_bin_index = (start + end) // 2

            if bin_index <= mid_bin_index:
                # update the left child
                self.update_tree(left, height + 1, bin_index, start, mid_bin_index, value)
            else:
                # update the right child
                self.update_tree(right, height + 1, bin_index, mid_bin_index + 1, end, value)
            
            self.segment_tree[node] = max(self.segment_tree[left], self.segment_tree[right])

    def search(self, value: int) -> int:
        """Get the bin index that has remaining space >= value"""
        # search the entire segment tree for the leaf-node that has remaining space >= value
        tree_node = self.search_tree(node=0, height=0, value=value)
        return self.segment_tree_to_bin_index(tree_node)

    def search_tree(self, node, height, value: int) -> int:
        """
        Binary search on the segment tree recursively
        This is array-based implementation, the root of the tree is at index 0, and for any node at index i:
        •	The left child is at index 2*i + 1
        •	The right child is at index 2*i + 2
        """
        # when reach the leaf node
        if height == self.height - 1:
            return node

        left = 2 * node + 1
        right = 2 * node + 2

        # if the left child is larger than the value, search on the left child
        if self.segment_tree[left] >= value:
            # search on the left child
            return self.search_tree(node=left, height=height+1, value=value)
        else:
            # search on the right child
            return self.search_tree(node=right, height=height+1, value=value)

    def segment_tree_to_bin_index(self, node):
        return node - (2 ** (self.height - 1) - 1)
    

    @property
    def height(self):
        return (len(self.bins_remaining_capacity) - 1).bit_length() + 1

    def insert(self, item_size):
        # if the segment tree is empty or the item size is larger than the smallest bin size
        if not self.segment_tree or self.segment_tree[0] < item_size:
            length = 1 if not self.bins_remaining_capacity else 2 * len(self.bins_remaining_capacity)
            self.expand_capacity(length)

        bin_index = self.search(value=item_size)
        if bin_index < self.num_bins:
            self.bin_to_items[bin_index].append(self.num_items)  # idx-th bin contains the current item
        else:
            bin_index = self.num_bins
            self.bin_to_items.append([self.num_items])  # The new bin contains the current item
            self.num_bins += 1  # Increased number of used bins

        self.bins_remaining_capacity[bin_index] -= item_size
        self.item_to_bin.append(bin_index)  # Current item was put in new bin            
        # update the segment tree
        self.update(bin_index, self.bins_remaining_capacity[bin_index])

        self.num_items += 1  # Increased number of items

        return bin_index


def fast_best_fit_decreasing(items, bin_capacity, progress_bar=False):
    packer = FastBestFitTree(bin_capacity)
    # We need to sort the items for best fit decreasing
    indexed_items = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    
    if progress_bar:
        for _, item in tqdm.tqdm(indexed_items):
            packer.insert(item)
    else:
        for _, item in indexed_items:
            packer.insert(item)

    # Recover the original indexing for bin_to_items
    bin_to_items = [[] for _ in range(packer.num_bins)]
    for bin_index, item_indices in enumerate(packer.bin_to_items):
        for item_index in item_indices:
            bin_to_items[bin_index].append(indexed_items[item_index][0])
    
    return bin_to_items