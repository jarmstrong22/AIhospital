import tkinter as tk



import heapq
import sys
import tkinter as tk
from PIL import ImageTk, Image, ImageOps 
from queue import PriorityQueue
import queue as PriorityQueue
from queue import PriorityQueue
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
    
    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = []
    
    def add_edge(self, from_node, to_node, weight):
        self.add_node(from_node)
        self.add_node(to_node)
        self.edges[from_node].append((to_node, weight))
        self.edges[to_node].append((from_node, weight))  # Assuming bidirectional edges
    
    def get_neighbors(self, node):
        neighbors = []
        for neighbor, weight in self.edges[node]:
            if weight != -1:
                neighbors.append((neighbor, weight))
        return neighbors
    
def main(): 
    # Create the hospital graph and define edges and weights
    matrix = [
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, -1121, -121, -121, -2121, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
        [0, -1, 1, -21, 1, 1, -21, 1, 2, 2, 2, 2, 2, -2, 2, 2, 2, 2, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 1, 1, -21, 1, 1, 1, 2, 2, 2, -2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, -12, -12, -12, -12, -12, -12, -12, -12, 2, -12, -12, -12, -12, -12, 0, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, 3, 2, 2, 2, -212, 2, -122, 2, 2, -2, 2, 2, 2, 2, 2, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 3, 2, 2, 2, -212, 2, -122, 2, 2, -2, 2, 2, 2, 2, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 0, 2, 2, 2, -212, 2, -122, 2, 2, -2, 2, 2, 2, 2, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 0, 2, 2, 2, -212, 2, -122, 2, 2, -2, 4, 4, 3, 3, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 0, 0, 0, -12, -12, -12, -12, -12, -12, -2, 4, 4, 3, 3, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 5, 5, 5, 2, 2, 2, -6, 2, 2, 2, -2, 4, 4, 5, 4, 4, 0, 8, 7, 7, 7, 7],
        [-1, 0, 0, 5, 5, 5, 2, 2, 2, -6, 2, 2, 2, -2, 4, 4, 5, 4, 4, 0, 8, 7, 7, 7, 7],
        [-1, 0, 0, 5, 5, 5, -62, -61, -61, -66, 6, 2, 2, -2, 3, 3, 5, 5, 5, 0, 8, 8, 8, 8, 8],
        [-1, 0, 0, 5, 5, 5, 6, 6, 6, -6, 2, 2, 2, -2, 3, 3, 5, 5, 5, 0, 8, 8, 8, 8, 8],
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8],
        [0, -1, 0, 0, 5, 5, 7, 7, 7, 13, 13, 13, 13, 0, 10, 10, 10, 5, 5, 5, 5, 5, 5, -1, -1],
        [0, -1, 0, 0, 5, 5, 5, 7, 7, 13, 13, 13, 13, 0, 10, 10, 10, 5, 5, 5, 5, 5, 5, -1, 0],
        [0, -1, 0, 0, 5, 5, 5, 12, 13, 13, 13, 13, 12, 0, 10, 10, 10, 5, 5, 5, 5, 5, 5, -1, 0],
        [0, -1, 0, 0, 5, 5, 5, 12, 12, 12, 12, 12, 12, 0, 10, 10, 10, 5, 5, 5, 5, 5, 5, -1, 0],
        [0, -1, 0, 0, 5, 5, 12, 12, 12, 12, 12, 12, 12, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, -1, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 11, 11, 11, 10, 10, 10, 10, -1, 0],
        [0, -1, 0, 5, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 11, 11, 11, 10, 10, 10, 10,-1, 0],
        [0, -1, 0, 0, 0, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 11, 11, 11, 10, 10, 10, 10, -1, 0],
        [0, -1, 3, 3, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 11, 11, 11, 10, 10, 10, 10, -1, 0]
    ]
    graph = Graph()
    valid_values = [-1121, -121, -21, -2121, 0, -12, -122, -212, -6, 3, 4, 5, 7, 8, 10, 11, 12, 13, 1]
    obstacle = (10,3)

    for row_index, row in enumerate(matrix):
        for col_index, cell_value in enumerate(row):
            if cell_value != -1 and (obstacle is None or (row_index, col_index) != obstacle):
                graph.add_node((row_index, col_index))

                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_row = row_index + dr
                    new_col = col_index + dc
                    # Check if the new cell is within bounds and is not an obstacle or is not equal to the obstacle coordinate if defined
                    if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]) and (obstacle is None or (new_row, new_col) != obstacle):
                        if matrix[new_row][new_col] in valid_values:
                            graph.add_edge((row_index, col_index), (new_row, new_col), 1)

    
    # Assuming graph is already created and populated
    node = (9, 4)
    print("Node:", node)
    print("Connections:")
    for neighbor, weight in graph.get_neighbors(node):
        print(f"  Neighbor: {neighbor}, Weight: {weight}")
