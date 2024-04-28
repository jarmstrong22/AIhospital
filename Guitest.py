import tkinter as tk

class MazeGame:
    def __init__(self, root, maze, path=[]):
        self.root = root
        self.maze = maze
        self.path = path
        
        self.rows = len(maze)
        self.cols = len(maze[0])
        
        # Define color mapping
        self.color_mapping = {
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

        self.cell_size = 20
        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size, height=self.rows * self.cell_size, bg='white')
        self.canvas.pack()
        

        self.draw_maze()
    
    def draw_maze(self):
        for x in range(self.rows):
            for y in range(self.cols):
                if self.maze[x][y] in [-1, -2]:  # Check if the cell is a wall (-1 or -2)
                    color = self.color_mapping[self.maze[x][y]]
                else:
                    color = self.color_mapping.get(self.maze[x][y], '#FFFFFF')  # Default to white

                # If the cell coordinate is in last_coordinates, set its color to black
                if (x, y) in self.path:
                    color = '#000000'
                
                # Draw cell rectangle
                self.canvas.create_rectangle(
                    y * self.cell_size, x * self.cell_size,
                    (y + 1) * self.cell_size, (x + 1) * self.cell_size,
                    fill=color, outline='',  # Set outline to empty string to remove the border
                )

                if self.maze[x][y] in [-1, -2, -12, -122]:  # Check if the cell is a wall (-1 or -2)
                    if self.maze[x][y] == -2: 
                        # Draw black lines along the farthest right and left sides of the cell wall
                        line_start_left = (y * self.cell_size, x * self.cell_size)
                        line_end_left = (y * self.cell_size, (x + 1) * self.cell_size)
                        line_start_right = ((y + 1) * self.cell_size, x * self.cell_size)
                        line_end_right = ((y + 1) * self.cell_size, (x + 1) * self.cell_size)
                        self.canvas.create_line(line_start_left, line_end_left, fill='#000000', width=2)
                        self.canvas.create_line(line_start_right, line_end_right, fill='#000000', width=2)

                    if self.maze[x][y] == -12:  # Check if the cell value is -12
                        # Draw black lines along the top and bottom sides of the cell wall
                        line_start_top = (y * self.cell_size, x * self.cell_size)
                        line_end_top = ((y + 1) * self.cell_size, x * self.cell_size)
                        line_start_bottom = (y * self.cell_size, (x + 1) * self.cell_size)
                        line_end_bottom = ((y + 1) * self.cell_size, (x + 1) * self.cell_size)
                        self.canvas.create_line(line_start_top, line_end_top, fill='#000000', width=2)
                        self.canvas.create_line(line_start_bottom, line_end_bottom, fill='#000000', width=2)

                    if self.maze[x][y] == -122: 
                        # Draw black lines along right cell wall
                        line_start_right = ((y + 1) * self.cell_size, x * self.cell_size)
                        line_end_right = ((y + 1) * self.cell_size, (x + 1) * self.cell_size)
                        self.canvas.create_line(line_start_right, line_end_right, fill='#000000', width=2)

                    if self.maze[x][y] == -212: 
                        # Draw black lines along the farthest right and left sides of the cell wall
                        line_start_left = (y * self.cell_size, x * self.cell_size)
                        line_end_left = (y * self.cell_size, (x + 1) * self.cell_size)
                        self.canvas.create_line(line_start_left, line_end_left, fill='#000000', width=2)

    def draw_path(self):
        if self.path:
            for i in range(len(self.path) - 1):
                x1, y1 = self.path[i]
                x2, y2 = self.path[i + 1]
                self.canvas.create_line(y1 * self.cell_size + self.cell_size / 2, x1 * self.cell_size + self.cell_size / 2,
                                        y2 * self.cell_size + self.cell_size / 2, x2 * self.cell_size + self.cell_size / 2,
                                        fill='black', width=2)
    
    ############################################################
    #### This is for the GUI part. No need to modify this unless
    #### screen changes are needed.
    ############################################################
    def move_agent(self, event):
    
        #### Move right, if possible
        if event.keysym == 'Right' and self.agent_pos[1] + 1 < self.cols and not self.cells[self.agent_pos[0]][self.agent_pos[1] + 1].is_wall:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)


        #### Move Left, if possible            
        elif event.keysym == 'Left' and self.agent_pos[1] - 1 >= 0 and not self.cells[self.agent_pos[0]][self.agent_pos[1] - 1].is_wall:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        
        #### Move Down, if possible
        elif event.keysym == 'Down' and self.agent_pos[0] + 1 < self.rows and not self.cells[self.agent_pos[0] + 1][self.agent_pos[1]].is_wall:
            self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
   
        #### Move Up, if possible   
        elif event.keysym == 'Up' and self.agent_pos[0] - 1 >= 0 and not self.cells[self.agent_pos[0] - 1][self.agent_pos[1]].is_wall:
            self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])

        #### Erase agent from the previous cell at time t
        self.canvas.delete("agent")

        
        ### Redraw the agent in color navy in the new cell position at time t+1
        self.canvas.create_rectangle(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size, 
                                    (self.agent_pos[1] + 1) * self.cell_size, (self.agent_pos[0] + 1) * self.cell_size, 
                                    fill='navy', tags="agent")


maze = [
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
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

path = [
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), (3, 9), (3, 10), (3, 11), 
    (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (4, 18), (5, 18), (5, 19), (6, 19), 
    (6, 20), (7, 20), (8, 20), (9, 20), (10, 20), (11, 20), (11, 20), (11, 19), (11, 18), (11, 17), 
    (11, 16), (11, 15), (11, 14), (11, 13), (11, 12), (11, 11), (11, 10), (11, 9), (11, 9), (11, 10), 
    (11, 11), (11, 12), (11, 13), (11, 14), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (16, 14), 
    (16, 13), (16, 12), (16, 11), (16, 10), (16, 9), (16, 9), (15, 9), (14, 9), (13, 9), (12, 9), (11, 9), 
    (10, 9), (9, 9), (8, 9), (7, 9), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), 
    (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23)
]

root = tk.Tk()
root.title("Maze")

game = MazeGame(root, maze, path)
root.bind("<KeyPress>", game.move_agent)

root.mainloop()
