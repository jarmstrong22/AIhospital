import heapq
import sys
import queue as PriorityQueue

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
        return self.edges[node]

def heuristic(self, pos):
        return (abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1]))

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
        delivery_algorithm = lines[0].split(':')[1].strip()
        start_location = lines[1].split(':')[1].strip()
        delivery_locations = [loc.strip() for loc in lines[2].split(':')[1].split(',')]
    return delivery_algorithm, start_location, delivery_locations

def main():
    if len(sys.argv) != 2:
        print("Usage: python FindPath.py inputfile.txt")
        return
    
    filename = sys.argv[1]
    delivery_algorithm, start_location, delivery_locations = read_input_file(filename)
    
    print("Delivery Algorithm:", delivery_algorithm)
    print("Start Location:", start_location)
    print("Delivery Locations:", delivery_locations)
    
    # Create the hospital graph and define edges and weights
    graph = [
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

    ward_priorities = {
        'ICU': 5, 'ER': 5, 'Oncology': 5, 'Burn Ward': 5,
        'Surgical Ward': 4, 'Maternity Ward': 4,
        'Hematology': 3, 'Pediatric Ward': 3,
        'Medical Ward': 2, 'General Ward': 2,
        'Admissions': 1, 'Isolation Ward': 1
    }


    def find_optimum_path(graph, start, goal, algorithm):
        if algorithm == 'A*':
            return a_star(graph, start, goal)
        elif algorithm == 'Dijkstra':
            return dijkstra(graph, start, goal)
        else:
            print("Invalid delivery algorithm specified in the input file.")
            return None
    for location in delivery_locations:
        # Assume location is a string representing the delivery location
        ward_priority = ward_priorities.get(location)
        if ward_priority is not None:
            # Assign the ward based on the priority
            if ward_priority == 5:
                # Assign to ICU, ER, Oncology, or Burn Ward
                pass  # Add your code to assign the delivery to the corresponding ward
            elif ward_priority == 4:
                # Assign to Surgical Ward or Maternity Ward
                pass  # Add your code to assign the delivery to the corresponding ward
            # Continue with similar conditions for priorities 3, 2, and 1
        else:
            print(f"No priority assigned for location: {location}")

def create_graph(hospital_path):
    graph = Graph()
    # Create nodes and edges based on hospital path
    rows = len(hospital_path)
    cols = len(hospital_path[0])
    for i in range(rows):
        for j in range(cols):
            if hospital_path[i][j] == 1:
                graph.add_node((i, j))
                if i > 0 and hospital_path[i - 1][j] == 1:
                    graph.add_edge((i, j), (i - 1, j), 1)
                if i < rows - 1 and hospital_path[i + 1][j] == 1:
                    graph.add_edge((i, j), (i + 1, j), 1)
                if j > 0 and hospital_path[i][j - 1] == 1:
                    graph.add_edge((i, j), (i, j - 1), 1)
                if j < cols - 1 and hospital_path[i][j + 1] == 1:
                    graph.add_edge((i, j), (i, j + 1), 1)
    return graph

if __name__ == "__main__":
    main()
