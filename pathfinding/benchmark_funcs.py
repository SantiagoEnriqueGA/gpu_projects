import matplotlib.pyplot as plt

from path_finder import random_maze
from algorithms import *

def benchmark_pathfinding_algorithms():
    # Maze generation
    # -------------------------------------------   
    RUNS = 10
            
    def time_function(func, mazes):
        start = time.time()
        paths_found = 0; path_lengths = 0; steps_taken = 0; visited_cells = 0
        
        if func.__name__ == "<lambda>": name = "A*"
        else: name = func.__name__
        print(f"Running {name} for {RUNS} mazes, each of size {H}x{W}")
        
        for step in range(RUNS):
            maze = mazes[step]
            path_found, path, path_length, steps, visited = func(maze)
            if path_found: paths_found += 1           
            path_lengths += path_length
            steps_taken += steps
            visited_cells += len(visited)
        total_time = time.time() - start
        return (total_time / RUNS)
    
    maze_sizes = [(10, 10), (25, 25), (50, 50), (100, 100), (250, 250), (500, 500), (1000, 1000)]
    # maze_sizes = [(10, 10), (25, 25), (50, 50)] # For testing
    bfs_times = []; dfs_times = []; dijkstra_times = []; 
    a_star_man_times = []; a_star_euc_times = []; a_star_che_times = []; a_star_oct_times = []
    for H, W in maze_sizes:
        print() 
        mazes = [random_maze(H, W) for _ in range(RUNS)]
        funcs_to_time = [
            ("BFS", time_function(bfs, mazes)),
            ("DFS", time_function(dfs, mazes)),
            ("Dijkstra", time_function(dijkstra, mazes)),
            ("A* (Manhattan)", time_function(lambda maze: a_star(maze, "manhattan"), mazes)),
            ("A* (Euclidean)", time_function(lambda maze: a_star(maze, "euclidean"), mazes)),
            ("A* (Chebyshev)", time_function(lambda maze: a_star(maze, "chebyshev"), mazes)),
            ("A* (Octile)", time_function(lambda maze: a_star(maze, "octile"), mazes)),
        ]
        
        for f in funcs_to_time:
            if f[0] == "BFS": bfs_times.append(f[1])
            elif f[0] == "DFS": dfs_times.append(f[1])
            elif f[0] == "Dijkstra": dijkstra_times.append(f[1])
            elif f[0] == "A* (Manhattan)": a_star_man_times.append(f[1])
            elif f[0] == "A* (Euclidean)": a_star_euc_times.append(f[1])
            elif f[0] == "A* (Chebyshev)": a_star_che_times.append(f[1])
            elif f[0] == "A* (Octile)": a_star_oct_times.append(f[1])
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot([f[0] for f in maze_sizes], bfs_times, label="BFS", color='green')
    plt.plot([f[0] for f in maze_sizes], dfs_times, label="DFS", color='red')
    plt.plot([f[0] for f in maze_sizes], dijkstra_times, label="Dijkstra", color='purple')
    plt.plot([f[0] for f in maze_sizes], a_star_man_times, label="A* (Manhattan)", color='blue')
    plt.plot([f[0] for f in maze_sizes], a_star_euc_times, label="A* (Euclidean)", color='lightblue')
    plt.plot([f[0] for f in maze_sizes], a_star_che_times, label="A* (Chebyshev)", color='darkblue')
    plt.plot([f[0] for f in maze_sizes], a_star_oct_times, label="A* (Octile)", color='skyblue')
    plt.xlabel("Maze Size (H and W)")
    plt.ylabel("Average Time (seconds)")
    plt.title("Average Time Taken by Pathfinding Algorithms")
    plt.legend()
    plt.grid()
    plt.show()
    
    
if __name__ == "__main__":
    benchmark_pathfinding_algorithms()