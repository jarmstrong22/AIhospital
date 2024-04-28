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
                        neighbor_value = matrix[new_row][new_col]
                        if neighbor_value in valid_values:
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
    print(ward_coordinate_map)

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

    def visit_delivery_locations(sorted_delivery_locations, ward_coordinate_map, current_location, final_path):
        for location in sorted_delivery_locations:
            coordinates = ward_coordinate_map.get(location)
            if coordinates:
                # Iterate through all coordinates associated with the current location (ward)
                path_found = False
                for next_location in coordinates:
                    # Check if there is a path from the current location to the next coordinate
                    path = find_optimum_path(graph, current_location, next_location, delivery_algorithm)
                    if path:
                        print(f"Path found from {current_location} to {next_location}: {path}")
                        final_path.extend(path)
                        current_location = next_location  # Update the current location
                        path_found = True
                        break  # Exit the loop if a path is found
                if not path_found:
                    print(f"Warning: No path found from {current_location} to any coordinates of {location}.")
            else:
                print(f"Warning: No coordinates found for location: {location}")
        print("All delivery locations visited successfully.")
        print("The final path is: ", final_path)
        return final_path
   
    root = tk.Tk()
    maze = matrix

    # Create final_path list 
    final_path = []
    path = visit_delivery_locations(sorted_delivery_locations, ward_coordinate_map, start_location, final_path )
    
    rows = len(maze)
    cols = len(maze[0])
    
    

    cell_size = 20
    canvas = tk.Canvas(root, width=cols * cell_size, height=rows * cell_size, bg='white')
    canvas.pack()
    

    

    color_mapping = {
            0: '#FFFFFF',       # White
            1: '#ADD8E6',       # Light Blue
            -1: '#000000',      # Black
            -2: '#FF0000',      # Red for cell value -2
            -12: '#FF0000',     # Red up and down wall
            -122: '#FF0000',     # Red up and down wall
            -212: '#FF0000',     # Red up and down wall
            2: '#FF0000',       # Red
            3: '#B19CD9',       # Light Purple
            4: '#FFFF00',       # Yellow
            5: '#90EE90',       # Light Green
            6: '#6A0DAD',       # Dark Purple
            7: '#A9A9A9',       # Dark Grey
            8: '#FFCC80',       # Light Orange
            10: '#FFC0CB',      # Pink
            11: '#98FB98',      # Really Light Green
            12: '#4B5320',      # Army Green
            13: '#FF8C00'       # Dark Orange
        }
    
    def draw_maze():
        for x in range(rows):
            for y in range(cols):
                if maze[x][y] in [-1, -2]:  # Check if the cell is a wall (-1 or -2)
                    color = color_mapping[maze[x][y]]
                else:
                    color = color_mapping.get(maze[x][y], '#FFFFFF')  # Default to white

                # If the cell coordinate is in last_coordinates, set its color to black
                #if (x, y) in path:
                #    color = '#000000'
                
                # Draw cell rectangle
                canvas.create_rectangle(
                    y * cell_size, x * cell_size,
                    (y + 1) * cell_size, (x + 1) * cell_size,
                    fill=color, outline='',  # Set outline to empty string to remove the border
                )

                if maze[x][y] in [-1, -2, -12, -122]:  # Check if the cell is a wall (-1 or -2)
                    if maze[x][y] == -2: 
                        # Draw black lines along the farthest right and left sides of the cell wall
                        line_start_left = (y * cell_size, x * cell_size)
                        line_end_left = (y * cell_size, (x + 1) * cell_size)
                        line_start_right = ((y + 1) * cell_size, x * cell_size)
                        line_end_right = ((y + 1) * cell_size, (x + 1) * cell_size)
                        canvas.create_line(line_start_left, line_end_left, fill='#000000', width=2)
                        canvas.create_line(line_start_right, line_end_right, fill='#000000', width=2)

                    if maze[x][y] == -12:  # Check if the cell value is -12
                        # Draw black lines along the top and bottom sides of the cell wall
                        line_start_top = (y * cell_size, x * cell_size)
                        line_end_top = ((y + 1) * cell_size, x * cell_size)
                        line_start_bottom = (y * cell_size, (x + 1) * cell_size)
                        line_end_bottom = ((y + 1) * cell_size, (x + 1) * cell_size)
                        canvas.create_line(line_start_top, line_end_top, fill='#000000', width=2)
                        canvas.create_line(line_start_bottom, line_end_bottom, fill='#000000', width=2)

                    if maze[x][y] == -122: 
                        # Draw black lines along right cell wall
                        line_start_right = ((y + 1) * cell_size, x * cell_size)
                        line_end_right = ((y + 1) * cell_size, (x + 1) * cell_size)
                        canvas.create_line(line_start_right, line_end_right, fill='#000000', width=2)

                    if maze[x][y] == -212: 
                        # Draw black lines along the farthest right and left sides of the cell wall
                        line_start_left = (y * cell_size, x * cell_size)
                        line_end_left = (y * cell_size, (x + 1) * cell_size)
                        canvas.create_line(line_start_left, line_end_left, fill='#000000', width=2)

    draw_maze()

    def draw_path():
        if path:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                canvas.create_line(y1 * cell_size + cell_size / 2, x1 * cell_size + cell_size / 2,
                                        y2 * cell_size + cell_size / 2, x2 * cell_size + cell_size / 2,
                                        fill='black', width=2)
                
    draw_path()
    
   



    
    
    root.title("Maze")

    

    root.mainloop()

if __name__ == "__main__":
    main()

