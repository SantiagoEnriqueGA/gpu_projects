from path_finder import random_maze
from algorithms import *
from tqdm import tqdm

def benchmark_pathfinding_algorithms():
    # Maze generation
    # -------------------------------------------   
    H, W = 250, 250
    RUNS = 10
    # Generate #RUNS random mazes
    MAZES = [random_maze(H, W) for _ in range(RUNS)]
    
    def time_function(func):
        start = time.time()
        paths_found = 0; path_lengths = 0; steps_taken = 0; visited_cells = 0
        
        if func.__name__ == "<lambda>": name = "A*"
        else: name = func.__name__
        
        for step in tqdm(range(RUNS), desc=f"Running {name}"):
            maze = MAZES[step]
            path_found, path, path_length, steps, visited = func(maze)
            if path_found: paths_found += 1           
            path_lengths += path_length
            steps_taken += steps
            visited_cells += len(visited)
        total_time = time.time() - start
        return (total_time / RUNS), paths_found, path_lengths, steps_taken, visited_cells
    
    funcs_to_time = [
        ("BFS", time_function(bfs)),
        ("DFS", time_function(dfs)),
        ("Dijkstra", time_function(dijkstra)),
        ("A* (Manhattan)", time_function(lambda maze: a_star(maze, "manhattan"))),
        ("A* (Euclidean)", time_function(lambda maze: a_star(maze, "euclidean"))),
        ("A* (Chebyshev)", time_function(lambda maze: a_star(maze, "chebyshev"))),
        ("A* (Octile)", time_function(lambda maze: a_star(maze, "octile"))),
    ]
    
    print(f"\nResults for {RUNS} mazes, each of size {H}x{W}")
    bfs_paths = dfs_paths = dijkstra_paths = a_star_man_paths = a_star_euc_paths = a_star_che_paths = a_star_oct_paths = 0
    for f in funcs_to_time:
        print(f"Algorithm: {f[0]}")
        print(f"\tAverage time:        {f[1][0]:.6f} seconds")
        print(f"\tPaths found:         {f[1][1]:,}")
        print(f"\tAverage path length: {f[1][2] / f[1][1]:.2f}")
        print(f"\tTotal steps taken:   {f[1][3]:,}")
        print(f"\tTotal visited cells: {f[1][4]:,}")
        
        if f[0] == "BFS": bfs_paths = f[1][1]
        elif f[0] == "DFS": dfs_paths = f[1][1]
        elif f[0] == "Dijkstra": dijkstra_paths = f[1][1]
        elif f[0] == "A* (Manhattan)": a_star_man_paths = f[1][1]
        elif f[0] == "A* (Euclidean)": a_star_euc_paths = f[1][1]
        elif f[0] == "A* (Chebyshev)": a_star_che_paths = f[1][1]
        elif f[0] == "A* (Octile)": a_star_oct_paths = f[1][1]
        
    # Assert that all algorithms found the same number of paths
    assert bfs_paths == dfs_paths == dijkstra_paths == a_star_man_paths == a_star_euc_paths == a_star_che_paths == a_star_oct_paths
    print(f"\nALL ALGORITHMS FOUND {bfs_paths} PATHS!")

if __name__ == "__main__":
    benchmark_pathfinding_algorithms()