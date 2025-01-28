import curses
from curses import wrapper
import random

from algorithms import bfs, dfs, a_star, gbfs, dijkstra, bidirectional
from utils import print_results

    
def random_grid_maze(rows, cols):
    """
    Generates a random grid maze.
    
    Parameters:
        - rows (int) - The number of rows in the maze.
        - cols (int) - The number of columns in the maze.
    
    Returns:
        - maze (list) - A 2D list representing the maze, where '#' represents walls and ' ' represents paths.
        The maze also contains a start point 'O' and an end point 'X'.
    """
    # Create an empty maze with walls
    maze = [['#' if i == 0 or i == rows-1 or j == 0 or j == cols-1 else ' ' for j in range(cols)] for i in range(rows)]

    # Add internal walls and paths
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if i % 2 == 0 and j % 2 == 0:
                maze[i][j] = '#'
            elif i % 4 == 0 and j % 4 == 0:
                maze[i][j] = '#'
            elif i % 6 == 0 and j % 6 == 0:
                maze[i][j] = '#'

    # Place the start and end points, randomly
    maze[random.randint(1, rows-2)][random.randint(1, cols-2)] = 'O'
    maze[random.randint(1, rows-2)][random.randint(1, cols-2)] = 'X'
    
    return maze

def random_maze(rows, cols, p=0.3):
    """
    Generates a random maze with the specified number of rows and columns.
    
    Parameters:
        - rows (int) - The number of rows in the maze.
        - cols (int) - The number of columns in the maze.
        - p (float) - The probability of a cell being a wall ('#'). Default is 0.3.
    
    Returns:
        - maze (list) - A 2D list representing the generated maze, where '#' represents a wall and ' ' represents an empty cell.
    """
    maze = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            if random.random() < p:
                row.append("#")
            else:
                row.append(" ")
        maze.append(row)
    maze[0][1] = "O"
    maze[-1][-2] = "X"
    
    # Surround the maze with walls
    for row in maze:
        row.insert(0, "#")
        row.append("#")
    maze.insert(0, ["#" for _ in range(len(maze[0]))])
    maze.append(["#" for _ in range(len(maze[0]))])
    
    return maze

        
def main(stdscr):    
    # Initialize the curses window, set the colors
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    # Maze generation
    # -------------------------------------------   
    H, W = 10, 10
    maze = random_grid_maze(H, W)
    # maze = random_maze(H, W)
    
    # Run the path finding algorithms
    # -------------------------------------------
    bfss = bfs(maze, stdscr)
    bfss = ["bfs"] + list(bfss)
    
    gbfss = gbfs(maze, stdscr)
    gbfss = ["Greedy bfs"] + list(gbfss)
    
    dfss = dfs(maze, stdscr)
    dfss = ["dfs"] + list(dfss)
    
    astar_m = a_star(maze, stdscr, "manhattan")
    astar_m = ["astar-manhattan"] + list(astar_m)
    
    astar_e = a_star(maze, stdscr, "euclidean")
    astar_e = ["astar-euclidean"] + list(astar_e)
    
    astar_c = a_star(maze, stdscr, "chebyshev")
    astar_c = ["astar-chebyshev"] + list(astar_c)
    
    astar_o = a_star(maze, stdscr, "octile")
    astar_o = ["astar-octile"] + list(astar_o)
    
    dijk = dijkstra(maze, stdscr)
    dijk = ["dijkstra"] + list(dijk)
    
    bi = bidirectional(maze, stdscr)
    bi = ["bidirectional"] + list(bi)
    
    
    # Print the results, and wait for a key press        
    print_results(stdscr, [bfss,gbfss, dfss, dijk, bi, astar_m, astar_e, astar_c, astar_o] ,maze)
    
if __name__ == "__main__":
    wrapper(main)