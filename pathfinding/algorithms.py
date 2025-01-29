import queue
import heapq
import math
import cupy as cp
from utils import *

def bfs(maze):
    """
    Breadth-First Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
    
    Returns:
        bool - True if the path is found, False otherwise.
        list - The path from the start to the end position.
        int - The length of the path.
        int - The number of steps taken.
        list - The list of visited positions.
    """
    start = "O"                         # Start position
    end = "X"                           # End position
    start_pos = find_val(maze, start)   # Find the start position
    end_pos = find_val(maze, end)       # Find the end position

    q = queue.Queue()   # Create a queue
    q.put([start_pos])  # Put the start position in the queue
    visited = set()     # Create a set to store visited positions

    steps = 0
    while not q.empty():            # While the queue is not empty
        path = q.get()              # Get the path from the queue
        row, col = path[-1]         # Get the current position

        steps += 1

        # If the current position is the end position
        if maze[row][col] == end:   
            return True, path, len(path)-1, steps, list(visited)
        
        # Else, find the neighbors of the current position
        neighbors = find_neighbors(maze, row, col)
        for neighbor in neighbors:      # For each neighbor
            if neighbor in visited:     # If the neighbor has been visited
                continue                # Skip

            r, c = neighbor
            if maze[r][c] == "#":       # If the neighbor is a wall
                continue                # Skip

            new_path = path + [neighbor]    # Add the neighbor to the path
            q.put(new_path)                 # Put the new path in the queue
            visited.add(neighbor)           # Add the neighbor to the visited set
    
    # return True, path, len(path)-1, steps, list(visited)
    # If no path is found, return False
    return False, [], 0, steps, list(visited)

def bfs_cuda(maze):
    """
    Breadth-First Search algorithm to find the shortest path in a maze.
    Implemented using CP CUDA Raw Kernels.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
    
    Returns:
        bool - True if the path is found, False otherwise.
        list - The path from the start to the end position.
        int - The length of the path.
        int - The number of steps taken.
        list - The list of visited positions.
    """
    start = "O"
    end = "X"
    start_pos = find_val(maze, start)
    end_pos = find_val(maze, end)

    # Convert maze to a 1D array for CUDA processing
    maze_1d = cp.array([cell for row in maze for cell in row], dtype=cp.int32)
    rows, cols = len(maze), len(maze[0])
    
    # Define CUDA kernel
    bfs_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void bfs_kernel(int* maze, int rows, int cols, int* start_pos, int* end_pos, int* path, int* path_len, int* steps, int* visited) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= rows * cols) return;

        // Initialize BFS structures
        int q[1024];
        int q_start = 0, q_end = 0;
        int visited_local[1024];
        int visited_count = 0;

        // Enqueue start position
        q[q_end++] = start_pos[0] * cols + start_pos[1];
        visited_local[visited_count++] = q[q_start];

        // BFS loop
        while (q_start < q_end) {
            int current = q[q_start++];
            int row = current / cols;
            int col = current % cols;

            // Check if we reached the end
            if (row == end_pos[0] && col == end_pos[1]) {
                path[path_len[0]++] = current;
                steps[0] = q_end;
                for (int i = 0; i < visited_count; ++i) {
                    visited[i] = visited_local[i];
                }
                return;
            }

            // Explore neighbors
            int neighbors[4][2] = {{row-1, col}, {row+1, col}, {row, col-1}, {row, col+1}};
            for (int i = 0; i < 4; ++i) {
                int r = neighbors[i][0];
                int c = neighbors[i][1];
                if (r >= 0 && r < rows && c >= 0 && c < cols && maze[r * cols + c] != '#' && maze[r * cols + c] != 'X') {
                    int neighbor_idx = r * cols + c;
                    bool already_visited = false;
                    for (int j = 0; j < visited_count; ++j) {
                        if (visited_local[j] == neighbor_idx) {
                            already_visited = true;
                            break;
                        }
                    }
                    if (!already_visited) {
                        q[q_end++] = neighbor_idx;
                        visited_local[visited_count++] = neighbor_idx;
                    }
                }
            }
        }

        // If no path is found
        path_len[0] = 0;
        steps[0] = q_end;
        for (int i = 0; i < visited_count; ++i) {
            visited[i] = visited_local[i];
        }
    }
    ''', 'bfs_kernel')

    # Allocate memory for results
    path = cp.zeros((rows * cols, 2), dtype=cp.int32)
    path_len = cp.zeros(1, dtype=cp.int32)
    steps = cp.zeros(1, dtype=cp.int32)
    visited = cp.zeros((rows * cols, 2), dtype=cp.int32)

    # Launch kernel
    bfs_kernel((1,), (1,), (maze_1d, rows, cols, cp.array(start_pos, dtype=cp.int32), cp.array(end_pos, dtype=cp.int32), path, path_len, steps, visited))

    # Convert results back to host
    path_host = cp.asnumpy(path)[:path_len[0]]
    steps_host = int(steps[0])
    visited_host = cp.asnumpy(visited)[:steps_host]

    if path_len[0] > 0:
        return True, path_host.tolist(), path_len[0] - 1, steps_host, visited_host.tolist()
    else:
        return False, [], 0, steps_host, visited_host.tolist()

def dfs(maze):
    """
    Depth-First Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
    
    Returns:
        bool - True if the path is found, False otherwise.
        list - The path from the start to the end position.
        int - The length of the path.
        int - The number of steps taken.
        list - The list of visited positions.
    """
    start = "O"                         # Start position
    end = "X"                           # End position
    start_pos = find_val(maze, start)   # Find the start position
    end_pos = find_val(maze, end)       # Find the end position

    # Stack, visited set, and path list
    stack = [start_pos]
    visited = set()
    path = []

    while stack:                        # While the stack is not empty
        current_pos = stack.pop()       # Pop the top position from the stack
        path.append(current_pos)        # Add the position to the path
        row, col = current_pos          # Get the row and column of the position
        
        # If the current position is the end, return the path
        if maze[row][col] == end:
            return True, path, len(path)-1, len(path), path

        # Else, find the neighbors of the current position
        neighbors = find_neighbors(maze, row, col)
        
        # For each neighbor, if it has not been visited and is not a wall, add it to the stack
        for neighbor in neighbors:
            if neighbor not in visited and maze[neighbor[0]][neighbor[1]] != "#":
                stack.append(neighbor)
                visited.add(neighbor)

    # If no path is found, return False
    return False, path, len(path)-1, len(path), path

def a_star(maze, heuristic_type="manhattan"):
    """
    A* Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        heuristic_type (str) - The heuristic function to use. Default is "manhattan".
    
    Returns:
        bool - True if the path is found, False otherwise.
        list - The path from the start to the end position.
        int - The length of the path.
        int - The number of steps taken.
        list - The list of visited positions.
    """
    start = "O"                         # Start position
    end = "X"                           # End position
    start_pos = find_val(maze, start)   # Find the start position
    end_pos = find_val(maze, end)       # Find the end position

    open_set = queue.PriorityQueue()    # Priority queue
    open_set.put((0, start_pos))        # Put the start position in the queue
    came_from = {}                      # Dictionary to store the path
    g_score = {start_pos: 0}                                # Dictionary to store the g-score, the cost from the start to the current position
    f_score = {start_pos: heuristic(start_pos, end_pos, heuristic_type)}    # Dictionary to store the f-score, the sum of the g-score and the heuristic
    visited = set()                     # Set to store visited positions
    
    # While the open set is not empty
    while not open_set.empty():
        current = open_set.get()[1]     # Get the current position

        # If the current position is the end position
        if current == end_pos:
            path = []
            while current in came_from:         # While the current position is in the path
                path.append(current)            # Add the current position to the path
                current = came_from[current]    # Move to the next position
            path.append(start_pos)              # Add the start position to the path
            path.reverse()                      # Reverse the path

            # Return the path
            return True, path, len(path)-1, len(visited), visited

        visited.add(current)    # Add the current position to the visited set
        row, col = current      # Get the row and column of the current position

        # Else, find the neighbors of the current position
        neighbors = find_neighbors(maze, row, col)
        
        # For each neighbor, if it has not been visited and is not a wall, calculate the g-score and f-score
        for neighbor in neighbors:
            if neighbor in visited or maze[neighbor[0]][neighbor[1]] == "#":
                continue
            tentative_g_score = g_score[current] + 1

            # If the neighbor is not in the g-score dictionary or the tentative g-score is less than the current g-score
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_pos, heuristic_type)
                open_set.put((f_score[neighbor], neighbor))

    # If no path is found, return False
    return False, [], 0, len(visited), visited

def heuristic(pos1, pos2, type="manhattan"):
    """Calculate the Manhattan distance between two positions."""
    if type == "manhattan":
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    if type == "euclidean":
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    if type == "chebyshev":
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
    if type == "octile":
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def dijkstra(maze):
    """
    Dijkstra's algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
    
    Returns:
        bool - True if the path is found, False otherwise.
        list - The path from the start to the end position.
        int - The length of the path.
        int - The number of steps taken.
        list - The list of visited positions.
    """
    start = "O"                         # Start position
    end = "X"                           # End position
    start_pos = find_val(maze, start)   # Find the start position
    end_pos = find_val(maze, end)       # Find the end position

    open_set = []                               # Priority queue
    heapq.heappush(open_set, (0, start_pos))    # Put the start position in the queue
    came_from = {}                              # Dictionary to store the path
    g_score = {start_pos: 0}                    # Dictionary to store the g-score, the cost from the start to the current position
    visited = set()                             # Set to store visited positions

    # While the open set is not empty
    while open_set:
        current_cost, current = heapq.heappop(open_set)     # Get the current position

        # If the current position is the end position
        if current == end_pos:
            path = []
            while current in came_from:         # While the current position is in the path
                path.append(current)            # Add the current position to the path
                current = came_from[current]    # Move to the next position
            path.append(start_pos)              # Add the start position to the path
            path.reverse()                      # Reverse the path

            # Return the path
            return True, path, len(path)-1, len(visited), visited

        # Else, add the current position to the visited set
        visited.add(current)
        row, col = current

        # Else, find the neighbors of the current position
        neighbors = find_neighbors(maze, row, col)
        
        # For each neighbor, if it has not been visited and is not a wall, calculate the g-score
        for neighbor in neighbors:
            if neighbor in visited or maze[neighbor[0]][neighbor[1]] == "#":
                continue

            tentative_g_score = g_score[current] + 1    # Calculate the g-score

            # If the neighbor is not in the g-score dictionary or the tentative g-score is less than the current g-score
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current                               # Add the current position to the path
                g_score[neighbor] = tentative_g_score                       # Update the g-score
                heapq.heappush(open_set, (tentative_g_score, neighbor))     # Put the neighbor in the queue
    
    # If no path is found, return False
    return False, [], 0, len(visited), visited
