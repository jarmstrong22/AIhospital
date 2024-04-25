import tkinter as tk

class MazeGame:
    
    def __init__(self, root, maze):
        self.root = root
        self.maze = maze
        
        self.rows = len(maze)
        self.cols = len(maze[0])
        
        # Define color mapping
        self.color_mapping = {
            0: '#FFFFFF',       # White
            1: '#ADD8E6',       # Light Blue
            -1: '#000000',      # Black
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
                if self.maze[x][y] == 1: 
                    color = '#ADD8E6'
                elif self.maze[x][y] == -1:
                    color = '#000000'
                elif self.maze[x][y] == 0:
                    color = '#FFFFFF'
                elif self.maze[x][y] == 2:
                    color = '#FF0000'
                elif self.maze[x][y] == 3:
                    color = '#B19CD9'
                elif self.maze[x][y] == 4:
                    color = '#FFFF00'
                elif self.maze[x][y] == 5:
                    color = '#90EE90'
                elif self.maze[x][y] == 6:
                    color = '#6A0DAD'
                elif self.maze[x][y] == 7:
                    color = '#A9A9A9'
                elif self.maze[x][y] == 8:
                    color = '#FFCC80'
                elif self.maze[x][y] == 10:
                    color = '#FFC0CB'
                elif self.maze[x][y] == 11:
                    color = '#98FB98'
                elif self.maze[x][y] == 12:
                    color = '#4B5320'
                elif self.maze[x][y] == 13:
                    color = '#FF8C00'
                
                self.canvas.create_rectangle(y * self.cell_size, x * self.cell_size, (y + 1) * self.cell_size, (x + 1) * self.cell_size, fill=color)
                

    

maze = [
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

root = tk.Tk()
root.title("Maze")

game = MazeGame(root, maze)

root.mainloop()
