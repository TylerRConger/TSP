# Library imports
import sys
import time

# Local importspyt
from TSP import *



# TODO implement functionality for both MaxClique and TSP in one

# Will need to capture arguments as such python ./main.py <TSP/Max> <Flags>

    # Call the appropriate function either TSP or Max Clique with the appropriate arguments
    # Update display.py to function with Max Clique

# CODE START

# Method Calls

if __name__ == "__main__":
    # Command-line argument parsing
    args = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        value = None if i+1 >= len(sys.argv) or sys.argv[i+1].startswith("--") else sys.argv[i+1]

        if value:
            i += 1

        if arg == "--gs" or arg == "--graph-size":
            arg = "--graph_size"
        elif arg == "--f":
            arg = "--filename"
        elif arg == "--d":
            arg = "--display"
        elif arg == "--a":
            arg = "--algorithm"
            if value == "nn":
                value = "nearest_neighbor"
            elif value == "rc":
                value = "random_cycle"
            elif value == "h":
                value = "heuristic"

        args[arg] = value
        i += 1

    # Display help commands
    if "--help" in args:
        print("Usage:")
        print("  --gs <int>: Number of vertices in the graph (default: 100)")
        print("  --f <str>: Filename to write or read the graph (default: <graph_size>.graph)")
        print("  --a <str>: Algorithm to solve the TSP (choices: brute_force, nearest_neighbor, heuristic, random_cycle, default: heuristic)")
        print("  --d <str>: Will display the resultant path as a png file (No extension needed in input)")
        print("  --help: Displays all possible commands")
        exit()

    print(args)
    # Set default values if not provided
    graph_size = int(args["--graph_size"]) if "--graph_size" in args else 100
    filename = args["--filename"] if "--filename" in args else f"{graph_size}.graph"
    algorithm = args["--algorithm"] if "--algorithm" in args else "heuristic"
    display = True if "--display" in args else False
    
    if graph_size < 2:
        print("Graph size must be 2 or larger, you cannot have a signle node graph")
        exit()

    display_fileName = None
    if "--display" in args and args["--display"] is not None:
        display_fileName = args["--display"]

    graph = generate_graph(graph_size)
    print("Graph generated.")

    # Include the file path
    directory = './graphs/'
    file_path = os.path.join(directory, filename)

    # Write to file
    with open(file_path, 'w') as f:
        for row in graph:
            f.write(' '.join(map(str, row)) + '\n')
    print(f"Graph written to {filename}")

    start = time.time()

    pathway = []

    graph = read_graph_from_file(file_path)
    if algorithm == "brute_force" or algorithm == "bf":
        print("Brute force algorithm called.")
        pathway = brute_force_tsp(graph)
    elif algorithm == "nearest_neighbor" or algorithm == "nn":
        print("Nearest neighbor algorithm called.")
        pathway = nearest_neighbor(graph)
    elif algorithm == "heuristic":
        print("Heuristic algorithm called.")
        pathway = heuristic_approach(graph)
    elif algorithm == "random_cycle":
        print("Random cycle algorithm called.")
        pathway = random_cycle_distance(graph)
    else:
        print("Invalid algorithm choice.")

    end = time.time()
    print(f"Algorithm execution time: {end - start} seconds")

    if display:
        displayTheGraph(graph, pathway, fileName=display_fileName if display_fileName else None)