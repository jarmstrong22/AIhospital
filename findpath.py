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
    delivery_algorithm = a_star
    filename = sys.argv[1]
    hospital_path = read_hospital_path(filename)
    delivery_algorithm, start_location, delivery_locations = read_input_file(sys.argv[2])
    
    # Create the hospital graph and define edges and weights
    graph = {
        (0, 0): [(0, 1), (1, 0)],
        (0, 1): [(0, 0), (0, 2), (1, 1)],
        (0, 2): [(0, 1), (0, 3), (1, 2)],
        (0, 3): [(0, 2), (0, 4), (1, 3)],
        (0, 4): [(0, 3), (0, 5), (1, 4)],
        (0, 5): [(0, 4), (0, 6), (1, 5)],
        (0, 6): [(0, 5), (0, 7), (1, 6)],
        (0, 7): [(0, 6), (0, 8), (1, 7)],
        (0, 8): [(0, 7), (0, 9), (1, 8)],
        (0, 9): [(0, 8), (0, 10), (1, 9)],
        (0, 10): [(0, 9), (0, 11), (1, 10)],
        (0, 11): [(0, 10), (0, 12), (1, 11)],
        (0, 12): [(0, 11), (0, 13), (1, 12)],
        (0, 13): [(0, 12), (0, 14), (1, 13)],
        (0, 14): [(0, 13), (0, 15), (1, 14)],
        (0, 15): [(0, 14), (0, 16), (1, 15)],
        (0, 16): [(0, 15), (1, 16)],
        (0, 17): [(1, 17)],
        (0, 18): [],
        (0, 19): [],
        (0, 20): [],
        (0, 21): [],
        (0, 22): [(1, 22)],
        (0, 23): [(1, 23), (0, 24)],
        (0, 24): [(0, 23), (0, 25), (1, 24)],
        (0, 25): [(0, 24), (0, 26), (1, 25)],
        (0, 26): [(0, 25), (0, 27), (1, 26)],
        (0, 27): [(0, 26), (0, 28), (1, 27)],
        (0, 28): [(0, 27), (0, 29), (1, 28)],
        (0, 29): [(0, 28), (0, 30), (1, 29)],
        (0, 30): [(0, 29), (0, 31), (1, 30)],
        (0, 31): [(0, 30), (0, 32), (1, 31)],
        (0, 32): [(0, 31), (0, 33), (1, 32)],
        (0, 33): [(0, 32), (0, 34), (1, 33)],
        (0, 34): [(0, 33), (0, 35), (1, 34)],
        (0, 35): [(0, 34), (0, 36), (1, 35)],
        (0, 36): [(0, 35), (0, 37), (1, 36)],
        (0, 37): [(0, 36), (0, 38), (1, 37)],
        (0, 38): [(0, 37), (0, 39), (1, 38)],
        (0, 39): [(0, 38), (0, 40), (1, 39)],
        (0, 40): [(0, 39), (0, 41), (1, 40)],
        (0, 41): [(0, 40), (0, 42), (1, 41)],
        (0, 42): [(0, 41), (0, 43), (1, 42)],
        (0, 43): [(0, 42), (0, 44), (1, 43)],
        (0, 44): 
    }
    
    # Create a priority queue for delivery requests
    delivery_queue = PriorityQueue()
    for location in delivery_locations:
        priority = get_priority(location)
        delivery_queue.put((priority, location))
    
    current_location = start_location
    while not delivery_queue.empty():
        _, next_location = delivery_queue.get()
        path = find_optimum_path(graph, current_location, next_location, delivery_algorithm)
        if path:
            print(f"Optimum path from {current_location} to {next_location}: {path}")
            current_location = next_location
        else:
            print(f"Warning: No path found from {current_location} to {next_location}. Terminating...")
            break
    else:
        print("All delivery requests successfully completed.")

def read_hospital_path(filename):
    with open(filename, 'r') as file:
        hospital_path = [list(map(int, line.strip())) for line in file]
    return hospital_path

def get_priority(location):
    # Define priorities based on the location
    priority_map = {
        "ICU": 5,
        "ER": 5,
        "Oncology": 5,
        "Burn Ward": 5,
        "Surgical Ward": 4,
        "Maternity Ward": 4,
        "Hematology": 3,
        "Pediatric Ward": 3,
        "Medical Ward": 2,
        "General Ward": 2,
        "Admissions": 1,
        "Isolation Ward": 1
    }
    for ward, priority in priority_map.items():
        if ward in location:
            return priority
    return 0

def find_optimum_path(graph, start, goal, algorithm):
    if algorithm == 'A*':
        return a_star(graph, start, goal)
    elif algorithm == 'Dijkstra':
        return dijkstra(graph, start, goal)
    else:
        print("Invalid delivery algorithm specified in the input file.")
        return None

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
