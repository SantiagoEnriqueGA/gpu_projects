import queue
import heapq
import curses
import math
import time
from utils import find_val, find_neighbors, print_maze

def bfs(maze, stdscr):
    """
    Breadth-First Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        stdscr - The curses window object.
    
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

        stdscr.clear()              # Clear the screen
        steps += 1
        print_maze(maze, stdscr, path, start_pos, end_pos, steps, visited=list(visited))  # Print the maze
        # time.sleep(0.1)             # Sleep 
        stdscr.refresh()            # Refresh the screen

        # If the current position is the end position
        if maze[row][col] == end:   
            stdscr.addstr(len(maze), len(maze[0])//2, "Path found!")
            stdscr.addstr(len(maze)+1, len(maze[0])//2, f"Path length: {len(path)-1}")
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
    
    stdscr.addstr(len(maze), len(maze[0])//2, "No path found!")
    return True, path, len(path)-1, steps, list(visited)

def dfs(maze, stdscr):
    """
    Breadth-First Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        stdscr - The curses window object.
    
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

        # Clear the screen and print the maze
        stdscr.clear()
        print_maze(maze, stdscr, path, start_pos, end_pos, len(path))
        stdscr.refresh()
        
        # If the current position is the end, return the path
        if maze[row][col] == end:
            stdscr.addstr(len(maze), len(maze[0])//2, "Path found!")
            stdscr.addstr(len(maze)+1, len(maze[0])//2, f"Path length: {len(path)-1}")
            return True, path, len(path)-1, len(path), path

        # Else, find the neighbors of the current position
        neighbors = find_neighbors(maze, row, col)
        
        # For each neighbor, if it has not been visited and is not a wall, add it to the stack
        for neighbor in neighbors:
            if neighbor not in visited and maze[neighbor[0]][neighbor[1]] != "#":
                stack.append(neighbor)
                visited.add(neighbor)

    # If no path is found, return False
    stdscr.addstr(len(maze), len(maze[0])//2, "No path found!")
    return False, path, len(path)-1, len(path), path

def a_star(maze, stdscr, heuristic_type="manhattan"):
    """
    A* Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        stdscr - The curses window object.
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
            
            # Print the maze with the path
            stdscr.clear()
            print_maze(maze, stdscr, path=path, start=start_pos, end=end_pos, steps=len(visited),visited=visited)
            stdscr.refresh()

            # Return the path
            return True, path, len(path)-1, len(visited), visited

        visited.add(current)    # Add the current position to the visited set
        row, col = current      # Get the row and column of the current position

        # Clear the screen and print the maze
        stdscr.clear()
        print_maze(maze, stdscr, path=[], start=start_pos, end=end_pos, steps=len(visited),visited=visited)
        stdscr.refresh()

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
    stdscr.addstr(len(maze), len(maze[0])//2, "No path found!")
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

def gbfs(maze, stdscr):
    """
    Greedy Best-First Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        stdscr - The curses window object.
    
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
            
            # Print the maze with the path
            stdscr.clear()
            print_maze(maze, stdscr, path=path, start=start_pos, end=end_pos, steps=len(visited), visited=visited)
            stdscr.refresh()

            # Return the path
            return True, path, len(path)-1, len(visited), visited

        # Else, add the current position to the visited set
        visited.add(current)
        row, col = current

        # Clear the screen and print the maze
        stdscr.clear()
        print_maze(maze, stdscr, path=[], start=start_pos, end=end_pos, steps=len(visited), visited=visited)
        stdscr.refresh()

        # Else, find the neighbors of the current position
        neighbors = find_neighbors(maze, row, col)
        
        # For each neighbor, if it has not been visited and is not a wall, calculate the heuristic
        for neighbor in neighbors:
            if neighbor in visited or maze[neighbor[0]][neighbor[1]] == "#":
                continue

            if neighbor not in visited:
                came_from[neighbor] = current               # Add the current position to the path
                priority = heuristic(neighbor, end_pos)     # Calculate the heuristic
                open_set.put((priority, neighbor))          # Put the neighbor in the queue

    # If no path is found, return False
    stdscr.addstr(len(maze), len(maze[0])//2, "No path found!")
    return False, [], 0, len(visited), visited    

def dijkstra(maze, stdscr):
    """
    Dijkstra's algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        stdscr - The curses window object.
    
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
            
            # Print the maze with the path
            stdscr.clear()
            print_maze(maze, stdscr, path=path, start=start_pos, end=end_pos, steps=len(visited), visited=visited)
            stdscr.refresh()

            # Return the path
            return True, path, len(path)-1, len(visited), visited

        # Else, add the current position to the visited set
        visited.add(current)
        row, col = current

        # Print the maze
        stdscr.clear()
        print_maze(maze, stdscr, path=[], start=start_pos, end=end_pos, steps=len(visited), visited=visited)
        stdscr.refresh()

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
    stdscr.addstr(len(maze), len(maze[0])//2, "No path found!")
    return False, [], 0, len(visited), visited

def bidirectional(maze, stdscr):
    """
    Bidirectional Search algorithm to find the shortest path in a maze.
    
    Parameters:
        maze (list) - A 2D list representing the maze.
        stdscr - The curses window object.
    
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

    def reconstruct_path(came_from_start, came_from_end, meeting_point):
        """Function to reconstruct the path from the start to the end position."""
        path = []
        
        # Find the path from the start to the meeting point
        current = meeting_point
        while current in came_from_start:
            path.append(current)
            current = came_from_start[current]
        path.reverse()
        
        # Find the path from the meeting point to the end
        current = meeting_point
        while current in came_from_end:
            current = came_from_end[current]
            path.append(current)
            
        return path

    # Queues, dictionaries, and sets, to store the positions, paths, and visited positions
    open_set_start = queue.Queue()  
    open_set_end = queue.Queue()    
    open_set_start.put(start_pos)    
    open_set_end.put(end_pos)            
    came_from_start = {}              
    came_from_end = {}                
    visited_start = set()             
    visited_end = set()               

    # While the queues are not empty
    while not open_set_start.empty() and not open_set_end.empty():
        # Get the current positions, start/end
        current_start = open_set_start.get()
        current_end = open_set_end.get()

        # If start meets end, reconstruct the path
        if current_start in visited_end:
            path = reconstruct_path(came_from_start, came_from_end, current_start)
            
            # Print the maze with the path
            stdscr.clear()
            print_maze(maze, stdscr, path=path, start=start_pos, end=end_pos, steps=len(visited_start) + len(visited_end), visited=visited_start.union(visited_end))
            stdscr.refresh()
            
            # Add the end position to the path, return the path
            path += [current_end]
            return True, path, len(path)-1, len(visited_start) + len(visited_end), visited_start.union(visited_end)
        
        # If end meets start, reconstruct the path
        if current_end in visited_start:
            path = reconstruct_path(came_from_start, came_from_end, current_end)
            
            # Print the maze with the path
            stdscr.clear()
            print_maze(maze, stdscr, path=path, start=start_pos, end=end_pos, steps=len(visited_start) + len(visited_end), visited=visited_start.union(visited_end))
            stdscr.refresh()
            
            # Add the end position to the path, return the path
            path += [current_end]
            return True, path, len(path)-1, len(visited_start) + len(visited_end), visited_start.union(visited_end)

        # Else, add the current positions to the visited sets
        visited_start.add(current_start)
        visited_end.add(current_end)
        row_start, col_start = current_start
        row_end, col_end = current_end

        # Print the maze
        stdscr.clear()
        print_maze(maze, stdscr, path=[], start=start_pos, end=end_pos, steps=len(visited_start) + len(visited_end), visited=visited_start.union(visited_end))
        stdscr.refresh()

        # Find the neighbors of the current positions from the start
        neighbors_start = find_neighbors(maze, row_start, col_start)
        
        # For each neighbor in the start set, if it has not been visited and is not a wall, add it to the queue
        for neighbor in neighbors_start:
            if neighbor not in visited_start and maze[neighbor[0]][neighbor[1]] != "#":
                open_set_start.put(neighbor)
                came_from_start[neighbor] = current_start

        # Find the neighbors of the current positions from the end
        neighbors_end = find_neighbors(maze, row_end, col_end)
        
        # For each neighbor in the end set, if it has not been visited and is not a wall, add it to the queue
        for neighbor in neighbors_end:
            if neighbor not in visited_end and maze[neighbor[0]][neighbor[1]] != "#":
                open_set_end.put(neighbor)
                came_from_end[neighbor] = current_end

    # If no path is found, return False
    stdscr.addstr(len(maze), len(maze[0])//2, "No path found!")
    return False, [], 0, len(visited_start) + len(visited_end), visited_start.union(visited_end)