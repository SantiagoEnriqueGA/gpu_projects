import time
import os
import sys
import curses

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '1'

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__} is {execution_time:.4f} seconds")
        return result
    return wrapper

def avg_timing_decorator(func):
    def wrapper(*args, **kwargs):
        N = 5
        total_time = 0
        for i in range(N):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
        avg_time = total_time / N
        print(f"Average execution time for {func.__name__} is {avg_time:.4f} seconds")
        return result
    return wrapper

def suppress_output():
    sys.stdout = open(os.devnull, 'w')

def enable_output():
    sys.stdout = sys.__stdout__
    
def check_numba_cuda():
    try:
        suppress_output()
        import numba
        from numba import cuda
        cuda.detect()
        enable_output()
        return True
    except:
        return False
    
def check_openCl():
    try:
        suppress_output()
        import pyopencl as cl
        os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION
        cl.create_some_context()
        enable_output()
        return True
    except:
        enable_output()
        return False

def check_cupy():
    try:
        # suppress_output()
        import cupy as cp
        cp.cuda.Device(0)
        # enable_output()
        return True
    except:
        enable_output()
        return False
    
    
# Maze Utils
def find_neighbors(maze, row, col):
    """
    Finds the neighbors of a given position in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        row (int) - The row of the position.
        col (int) - The column of the position.
    
    Returns:
        list - A list of neighboring positions.
    """
    neighbors = []  # List to store neighbors

    if row > 0:                             # UP
        neighbors.append((row - 1, col))
    if row + 1 < len(maze):                 # DOWN
        neighbors.append((row + 1, col))
    if col > 0:                             # LEFT
        neighbors.append((row, col - 1))
    if col + 1 < len(maze[0]):              # RIGHT
        neighbors.append((row, col + 1))

    return neighbors

def find_val(maze, val):
    """
    Finds the position of a given value in a maze.

    Parameters:
        maze (list) - A 2D list representing the maze.
        val - The value to be found in the maze.

    Returns:
        tuple - The position (row, column) of the value in the maze.
        Returns None if the value is not found.
    """
    for i, row in enumerate(maze):          # For each row in the maze
        for j, value in enumerate(row):     # For each column in the row
            if value == val:              # If the value is the start
                return i, j                 # Return the position
    return None

def print_maze(maze, stdscr, path=[], start=None, end=None, steps=0, offset=(0, 0), visited=None, path_len=None):
    """
    Prints the maze on the screen using curses library.
    
    Args:
        maze (list) -  The maze represented as a 2D list.
        stdscr: The curses window object.
        path (list, optional) -  The list of positions in the path. Defaults to an empty list.
        start (tuple, optional) -  The start position. Defaults to None.
        end (tuple, optional) -  The end position. Defaults to None.
        steps (int, optional) -  The number of steps taken. Defaults to 0.
        offset (tuple, optional) -  The offset for printing the maze. Defaults to (0, 0).
        visited (set, optional) -  The set of visited positions. Defaults to None.
        path_len (int, optional) -  The length of the path. Defaults to None.
    """
    # Define colors
    BLUE = curses.color_pair(1)
    RED = curses.color_pair(2)
    GREEN = curses.color_pair(3)
    YELLOW = curses.color_pair(4)

    for i, row in enumerate(maze):                                      # For each row in the maze
        for j, value in enumerate(row):                                 # For each column in the row
            if (i, j) == start:                                         # If the current position is the start
                stdscr.addstr(i+offset[0], j*2+offset[1], value, YELLOW)
            elif (i, j) == end:                                         # If the current position is the end
                stdscr.addstr(i+offset[0], j*2+offset[1], value, YELLOW)
            elif (i, j) in path:                                        # If the current position is in the path
                stdscr.addstr(i+offset[0], j*2+offset[1], "X", RED)
            elif visited and (i, j) in visited:                         # If the current position has been visited
                stdscr.addstr(i+offset[0], j*2+offset[1], "X", GREEN) 
            else:                                                       # Otherwise
                stdscr.addstr(i+offset[0], j*2+offset[1], value, BLUE)  # Print the value
    
    # Add markers for start, end, path, steps, and visited count
    stdscr.addstr(len(maze)//2 -1+offset[0], len(maze[0])*2+offset[1]+1, "O-Start", YELLOW)
    stdscr.addstr(len(maze)//2   +offset[0], len(maze[0])*2+offset[1]+1, "X-End", YELLOW)
    stdscr.addstr(len(maze)//2 +1+offset[0], len(maze[0])*2+offset[1]+1, "X-Path", RED)
    stdscr.addstr(len(maze)//2 +2+offset[0], len(maze[0])*2+offset[1]+1, f"Step count: {steps}", RED)
    if visited: 
        stdscr.addstr(len(maze)//2 +3+offset[0], len(maze[0])*2+offset[1]+1, f"Visited count: {len(visited)}", GREEN)
    if path_len:
        stdscr.addstr(len(maze)//2 +4+offset[0], len(maze[0])*2+offset[1]+1, f"Path length: {len(path)-1}", RED)


def print_results(stdscr, methods, maze, cols=3):
    """
    Prints the results of the path finding algorithms on the screen.
    
    Parameters:
        stdscr - The curses window object.
        methods (list) - A list of tuples containing the name of the algorithm and the results.
        maze (list) - A 2D list representing the maze
        cols (int) - Number of columns to print the results in
    """
    # Find the start and end positions
    start = find_val(maze, "O")
    end = find_val(maze, "X")
    
    # Clear the screen
    stdscr.clear()
    
    height, width = stdscr.getmaxyx()
    
    # Calculate the width of each column based on the screen width
    col_width = width // cols
    
    # Initialize offsets for each column
    offsets = [(1, col_width * i) for i in range(cols)]
    
    # For each method, print the path
    for i, method in enumerate(methods):
        # Method name and results
        name = method[0]
        path_found, path, path_length, steps, visited = method[1:]
        
        # Determine the current column and offset
        col_index = i % cols
        offset = offsets[col_index]
        
        # Print the path in the current column
        stdscr.addstr(offset[0] - 1, offset[1], f"{name.upper()} Path:")
        print_maze(maze, stdscr, path, start, end, steps, offset=offset, visited=visited, path_len=True)
        
        # Increment the offset for the current column
        offsets[col_index] = tuple(map(sum, zip(offset, (len(maze) + 2, 0))))
    
    # Refresh the screen and wait for a key press
    stdscr.refresh()
    stdscr.getch()