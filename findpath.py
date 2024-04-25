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

def heuristic(start, goal):
    return (abs(start[0] - goal[0]) + abs(start[1] - goal[1]))


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def a_star(graph, start, goal):
    open_set = [(0, start)]  # Priority queue
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(start, goal)

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor, weight in graph.get_neighbors(current):
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

def dijkstra(graph, start, goal):
    open_set = [(0, start)]  # Priority queue
    came_from = {}
    cost_so_far = {node: float('inf') for node in graph.nodes}
    cost_so_far[start] = 0

    while open_set:
        current_cost, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor, weight in graph.get_neighbors(current):
            new_cost = cost_so_far[current] + weight
            if new_cost < cost_so_far[neighbor]:
                came_from[neighbor] = current
                cost_so_far[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost, neighbor))
    
    return None  # No path found


def read_input_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Extract delivery algorithm, start location, and delivery locations
        delivery_algorithm = None
        start_location = None
        delivery_locations = []
        obstacle = None

        for line in lines:
            if line.startswith('Delivery algorithm:'):
                delivery_algorithm = line.split(':')[1].strip()
            elif line.startswith('Start location:'):
                start_location_str = line.split(':')[1].strip()
                # Correctly parse the start location string
                start_location = tuple(map(int, start_location_str.split(',')))
            elif line.startswith('Delivery locations:'):
                delivery_locations_str = line.split(':')[1].strip()
                # Correctly parse the delivery locations string
                delivery_locations = [loc.strip() for loc in delivery_locations_str.split(',')]
            elif line.startswith('Obstacle:'):
                obstacle_str = line.split(':')[1].strip()
                if obstacle_str:
                    obstacle = tuple(map(int, obstacle_str.split(',')))

        if delivery_algorithm is None or start_location is None or not delivery_locations:
            raise ValueError("Input file format is incorrect.")

    return delivery_algorithm, start_location, delivery_locations, obstacle




def arrange_delivery_requests(delivery_locations, ward_priorities):
    delivery_queue = PriorityQueue.PriorityQueue()
    for location in delivery_locations:
        ward_priority = ward_priorities.get(location)
        if ward_priority is not None:
            delivery_queue.put((ward_priority, location))
        else:
            print(f"No priority assigned for location: {location}")
    return delivery_queue



def main():
    if len(sys.argv) != 2:
        print("Usage: python FindPath.py inputfile.txt")
        return
    
    filename = sys.argv[1]
    delivery_algorithm, start_location, delivery_locations, obstacle = read_input_file(filename)
    
    print("Delivery Algorithm:", delivery_algorithm)
    print("Start Location:", start_location)
    print("Delivery Locations:", delivery_locations)
    if obstacle is None:
        print("No obstacle specified.")
    else:
        print("Obstacle location:", obstacle)
    
    # Create the hospital graph and define edges and weights
    matrix = [
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 0, 4, 4, 4, 7, 7],
        [-1, 0, 0, 5, 5, 5, 2, 2, 2, 6, 2, 2, 2, 2, 4, 4, 5, 4, 4, 0, 8, 7, 7, 7, 7],
        [-1, 0, 0, 5, 5, 5, 2, 2, 2, 6, 2, 2, 2, 2, 4, 4, 5, 4, 4, 0, 8, 7, 7, 7, 7],
        [-1, 0, 0, 5, 5, 5, 6, 6, 6, 6, 6, 2, 2, 2, 3, 3, 5, 5, 5, 0, 8, 8, 8, 8, 8],
        [-1, 0, 0, 5, 5, 5, 6, 6, 6, 6, 2, 2, 2, 2, 3, 3, 5, 5, 5, 0, 8, 8, 8, 8, 8],
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

    # Assuming your graph data is defined as a list of lists
    for row_index, row in enumerate(matrix):
        for col_index, cell_value in enumerate(row):
            if cell_value != -1 and (obstacle is None or (row_index, col_index) != obstacle):
                graph.add_node((row_index, col_index))

                # Add edges for adjacent cells (assuming 4-directional movement)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_row = row_index + dr
                    new_col = col_index + dc
                    # Check if the new cell is within bounds and is not an obstacle or is not equal to the obstacle coordinate if defined
                    if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]) and (obstacle is None or (new_row, new_col) != obstacle):
                        if matrix[new_row][new_col] != -1:
                            graph.add_edge((row_index, col_index), (new_row, new_col), 1)  # Assuming uniform weight for edges

    ward_priorities = {
        'ICU': 5, 'ER': 5, 'Oncology': 5, 'Burn Ward': 5,
        'Surgical Ward': 4, 'Maternity Ward': 4,
        'Hematology': 3, 'Pediatric Ward': 3,
        'Medical Ward': 2, 'General Ward': 2,
        'Admissions': 1, 'Isolation Ward': 1
    }
    ward_assignments = { 
        'ICU': 8, 'ER': 4, 'Oncology': 5, 'Burn Ward': 6,
        'Maternity Ward': 1, 'Surgical Ward': 10, 
        'Hematology': 13, 'Pediatric Ward': 12,
        'Medical Ward': 11, 'General Ward': 2,
        'Admissions': 7, 'Isolation Ward': 3

    }
    def create_ward_coordinate_map(matrix):
        ward_coordinate_map = {
            'ICU': [],
            'ER': [],
            'Oncology': [],
            'Burn Ward': [],
            'Maternity Ward': [],
            'Surgical Ward': [],
            'Hematology': [],
            'Pediatric Ward': [],
            'Medical Ward': [],
            'General Ward': [],
            'Admissions': [],
            'Isolation Ward': [], 
            'Wall': [], 
            'Hallway': []
        }

        for i, row in enumerate(matrix):
            for j, ward in enumerate(row):
                if ward == 8:
                    ward_coordinate_map['ICU'].append((i, j))
                elif ward == 0: 
                    ward_coordinate_map['Hallway'].append((i,j))
                elif ward == -1: 
                    ward_coordinate_map['Wall'].append((i,j))
                elif ward == 4:
                    ward_coordinate_map['ER'].append((i, j))
                elif ward == 5:
                    ward_coordinate_map['Oncology'].append((i, j))
                elif ward == 6:
                    ward_coordinate_map['Burn Ward'].append((i, j))
                elif ward == 1:
                    ward_coordinate_map['Maternity Ward'].append((i, j))
                elif ward == 10:
                    ward_coordinate_map['Surgical Ward'].append((i, j))
                elif ward == 13:
                    ward_coordinate_map['Hematology'].append((i, j))
                elif ward == 12:
                    ward_coordinate_map['Pediatric Ward'].append((i, j))
                elif ward == 11:
                    ward_coordinate_map['Medical Ward'].append((i, j))
                elif ward == 2:
                    ward_coordinate_map['General Ward'].append((i, j))
                elif ward == 7:
                    ward_coordinate_map['Admissions'].append((i, j))
                elif ward == 3:
                    ward_coordinate_map['Isolation Ward'].append((i, j))

        return ward_coordinate_map
    
    # Create ward_coordinate_map
    ward_coordinate_map = create_ward_coordinate_map(matrix)

    # Create a list of tuples containing location name and priority
    location_priority_pairs = [(location, ward_priorities.get(location, 0)) for location in delivery_locations]
    
    # Sort the list based on priority (descending order) and original order
    sorted_location_priority_pairs = sorted(location_priority_pairs, key=lambda x: (-x[1], delivery_locations.index(x[0])))
    
    # Extract sorted location names
    sorted_delivery_locations = [pair[0] for pair in sorted_location_priority_pairs]
    print("Delivery locations based on priorities:", sorted_delivery_locations)

    # Define function to get the optimum path
    def find_optimum_path(graph, start, goal, algorithm):
        if algorithm == 'A*':
            return a_star(graph, start, goal)
        elif algorithm == 'Dijkstra':
            return dijkstra(graph, start, goal)
        else:
            print("Invalid delivery algorithm specified in the input file.")
            return None

    # Try to visit all the delivery locations in the sorted_delivery_locations graph 
    def visit_delivery_locations(sorted_delivery_locations, ward_coordinate_map, current_location, final_path):
        for location in sorted_delivery_locations:
            coordinates = ward_coordinate_map.get(location)
            if coordinates:
                next_location = coordinates[0]  # Choose the first coordinate as the next location
                path = find_optimum_path(graph, current_location, next_location, delivery_algorithm)
                if path:
                    print(f"Path found from {current_location} to {next_location}: {path}")
                    final_path.extend(path)
                    current_location = next_location  # Update the current location
                else:
                    print(f"Warning: No path found from {current_location} to {next_location}.")
                    return  # Break out of the loop if no path is found
            else:
                print(f"Warning: No coordinates found for location: {location}")
                return  # Break out of the loop if no coordinates are found
        print("All delivery locations visited successfully.")
        print("The final path is: ", final_path)
        return final_path
    
    # Create final_path list 
    final_path = []
    # Attempt to visit all the delivery locations
    visit_delivery_locations(sorted_delivery_locations, ward_coordinate_map, start_location, final_path)

if __name__ == "__main__":
    main()
