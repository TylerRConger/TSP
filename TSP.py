# Tyler Conger
# CS 570
# TSP Problem
# 2.23.2024

# Import statements
import random
import itertools
import sys
import time
import threading
import os.path

# Import all the stuff we use in display
from display import *

# Generation and Printing of Graphs

def generate_graph(num_points):
    """
    Generates a fully connected graph with random weights between 1 and 500
    
    Arguments:
        num_points (int): The number of vertices in the graph
    
    Returns:
        graph (array): 2-d array representing adj matrix
    """
    # Initialize an empty 2D array to represent the adjacency matrix
    graph = [[0] * num_points for _ in range(num_points)]
    
    # Iterate over each pair of vertices
    for i in range(num_points):
        for j in range(i, num_points): 
            if i != j: 
                # Generate a random weight between 1 and 500
                weight = random.randint(1, 500)
                
                # Assign the same weight to both directions since 
                # it's a fully connected graph
                graph[i][j] = weight
                graph[j][i] = weight
                
    return graph

def write_graph_to_file(graph, filename):
    """
    Writes the graph to a file structured like

    0
    1 0
    1 2 3 0
    1 2 3 4 0
    ...
    
    Arguments:
        graph (array): 2-d array representing adj matrix
        filename (str): The name of the file to write to
    """
    with open(filename, 'w') as f:
        # Write each row to file
        for i in range(len(graph)):
            for j in range(i + 1): 
                f.write(str(graph[i][j]) + ' ')
            # Make sure each is a new line
            f.write('\n')

def generate_and_write_graph(num_points):
    """
    Generates a fully connected graph and writes it to a file
    Using the two above functions
    
    Arguments:
        num_points (int): The number of vertices in the graph
    """
    # Generate the graph
    graph = generate_graph(num_points)
    
    # Construct the filename
    filename = f"Size{num_points}.graph"
    
    # Write the graph to the file
    write_graph_to_file(graph, filename)
    
    # Print a message to confirm the file has been written
    print(f"Graph generated and written to {filename}")

def read_graph_from_file(filename):
    """
    Read the graph from file
    
    Arguments:
        filename (str): The name of the file
    
    Returns:
        graph (array): 2-d array representing adj matrix
    """
    graph = []
    with open(filename, 'r') as f:
        for line in f:
            # Split the line by spaces and convert each element to an int
            row = list(map(int, line.strip().split()))
            graph.append(row)
    return graph

def print_graph(graph):
    """
    Print the graph in the format it appears in the file
    
    Arguments:
        graph (array): 2-d array representing adj matrix
    """
    # Print the graph header
    print("\n\n=================================================\n\n")
    print("Printing a graph of size "  + str(len(graph)) + ":\n")
    print("=================================================")
    # Print each line of the graph
    for row in graph:
        print(' '.join(map(str, row)))

    # print the footer
    print("=================================================\n\n")

# Save the solution
    
def save_graph_solution(distance, pathway):
    """
    Save the TSP solution to a file
    
    Arguments:
        distance (int): The total distance traveled in the TSP solution
        pathway (tuple): The pathway taken in the TSP

    Returns:
        pathway (tuple): The pathway taken in the TSP -- paththru value
    """
    # Name the file appropriately
    filename = "s" + str(distance) + "_tconger1.sol"

    # Include the file path
    directory = './solutions/'
    file_path = os.path.join(directory, filename)
    # Open then write to file
    with open(file_path, 'w') as f:
        # Write the cycle taken to the file
        f.write(' '.join(map(str, pathway)) + " " + str(pathway[0]) +'\n')  # Add the starting point at the end
    
    # Save file name for later usage
    return pathway

# Brute Force attempt
        
def calculate_total_distance(cycle, graph):
    """
    Calculate the total distance traveled for a given group of cities
    
    Arguments:
        cycle (tuple): A list of cities taken
        graph (array): 2-d array representing adj matrix
    
    Returns:
        int: The total distance traveled for the given path
    """
    # Initialize
    total_distance = 0
    graph_size = len(graph)
    # Loop through the entire pathway taken
    for i in range(len(cycle)):
        # Get the distance between each individual set of points
        from_point = cycle[i]
        to_point = cycle[(i + 1) % graph_size]
        # Sum the distance
        total_distance += graph[max(from_point, to_point)][min(from_point, to_point)]

    return total_distance

def brute_force_tsp(graph):
    """
    Implementation of the brute force algorithm for the TSP, uses
    calculate_total_distance as a helper func
    
    Arguments:
        graph (array): 2-d array representing adj matrix

    Returns:
        filename (string): The associated filename written
    """

    # Set the shortest path to be extremely high to start with
    num_vertecies = len(graph)
    shortest_distance = float('inf')
    shortest_perm = None
    
    # Generate all possible paths that exsist within the graph
    paths = itertools.permutations(range(num_vertecies))

    # Loop through each permutation and calculate the total distance
    for perm in paths:
        # Make sure to capture the return trip aswell

        # Find  the distance and see if it is shorter
        distance = calculate_total_distance(perm, graph)
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_perm = perm
    
    # Save the solution to a file
    return save_graph_solution(shortest_distance, shortest_perm)

# Verification and Graphing

def convert_to_symmetric_matrix(graph):
    """
    Convert a 2-d representing a lower triangular matrix into a square matrix
    This just makes it easier to index later on

    Arguments:
        graph (array): 2-d array representing adj matrix (triangular)

    Returns:
        graph (array): 2-d array representing adj matrix (square)
    """
    # Initialize symmetric matrix with 0
    n = len(graph)
    symmetric_matrix = [[0] * n for _ in range(n)] 

    # Fill in the upper triangular part of the matrix
    for i in range(n):
        for j in range(i+1, n):
            symmetric_matrix[i][j] = graph[j][i]

    # Fill in the lower triangular part by copying the upper triangular piece
    for i in range(n):
        for j in range(i+1, n):
            symmetric_matrix[j][i] = symmetric_matrix[i][j]

    return symmetric_matrix

def verify_solution(graph, solution_filename):
    """
    Verify that reported distance travled matches actual distance traveled
    
    Arguments:
        graph (array): 2-d array representing adj matrix
        solution_filename (str): The filename of the file to verified
    
    Returns:
        bool: T/F, True if expected matches the actual distance traveled, False otherwise
    """
    # Read the solution from the solution file name
    solution_distance = int(solution_filename.split('_')[0][1:])  
    with open(solution_filename, 'r') as f:
        # Get the distance from the filename
        solution_perm = list(map(int, f.readline().strip().split()))
    
    # Calculate the total distance traveled based on the solution permutation
    total_distance = 0
    num_vert = len(graph)

    # TODO: Using - 1 as unsure if the final trip back home counts as a path or is necessary
    for i in range(len(solution_perm) - 1):
        from_point = solution_perm[i]
        to_point = solution_perm[(i + 1) % num_vert]
        total_distance += graph[max(from_point, to_point)][min(from_point, to_point)]

    # Check if reported = actual
    if total_distance == solution_distance:
        # Print correct and return
        print("Solution is correct. Total distance:", total_distance)
        return True
    else:
        # Print incorrect and return
        print("Solution is incorrect. Calculated distance:", total_distance, 
        "Expected distance:", solution_distance)
        return False

    # Could do this but want to have a print statement too
    # return total_distance == solution_distance

# Nearest Neighbor
 
def nearest_neighbor_helper(start_point, graph):
    """
    Nearest Neighbor wrapper function, because my heuristic uses NN aswell

    Arguments:
        graph (array): 2-d array representing adj matrix
        start_point(int): Position to start traveling from

    Returns:
        filename (string): The associated filename
    """
    nn_return = nearest_neighbor(start_point, graph)

    total_distance = nn_return[0]
    tour = nn_return[1]

    # Save the solution to a file
    return save_graph_solution(total_distance, tour)
 
def nearest_neighbor(start_point, graph):
    """
    Implementation of the Nearest Neighbor (NN) Algo
    0. Mark all nodes as unvisisted
    1. Start at a point
    2. Pick the shortest path from that node
    3. Mark Node as visisted 
    4. Repeat until all nodes have been visited

    Arguments:
        graph (array): 2-d array representing adj matrix
        start_point(int): Position to start traveling from

    Returns:
        total_distance (int): The distance of the path taken
        cycle (array): The path that was taken
    """
    # Save graph len for later as recalcuating takes time
    graph_len = len(graph)
    
    # Set all nodes to unvisited
    visited = [False] * graph_len

    # Start the cycle
    cycle = [start_point]
    
    # Start with the chosen start point (sometimes 0, sometimes randomly chosen)
    current_point = start_point
    visited[current_point] = True
    
    total_distance = 0
    
    # Repeat until all cities have been visited
    while len(cycle) < graph_len:
        nearest_point = None
        min_distance = float('inf')  # Initialize min_distance to infinity
        
        # Find the nearest unvisited point
        for point in range(graph_len):
            if not visited[point] and graph[current_point][point] < min_distance:
                nearest_point = point
                min_distance = graph[current_point][point]
        
        # Move to the nearest unvisited point
        cycle.append(nearest_point)
        visited[nearest_point] = True
        total_distance += min_distance
        
        # Update current point
        current_point = nearest_point
    
    # Add distance from the last point back to the starting point to complete the trip
    total_distance += graph[cycle[-1]][start_point]

    return total_distance, cycle

 # My Solution

def thread_helper(start_point, graph, results):
    """
    Used to be targeted by the heuristic algo

    This function executes the nearest_neighbor function and stores its 
    result in the shared results list

    Arguements:
        start_point (int): Randomly chosen point to start NN from
        graph (array): 2-d array representing adj matrix
        results (array): List of all the results to be appended to
    
    Returns:
        results (array): Appends to the results list as a return
    """
    result = nearest_neighbor(start_point, graph)
    results.append(result)

def heuristic_approach(graph):
    """
    Implementation of my own self created shortest path algorithm.

    Spawn threads and start each thread at a random spot then attempt 
    to solve the graph using nearest neighbor, then takes the best solution
    and starts finding nearby nodes to attempt to find shortest starting point
    continues up to num_threads

    Arguments:
        graph (array): 2-d array representing adj matrix

    Returns:
        filename (string): The associated filename
    """

    # Change number of threads as needed based on graphs
    number_threads = 40
    threads = []
    results = []

    # Spawn X threads where X is number_threads
    for _ in range (number_threads):
        start = random.randint(0, len(graph) - 1)
        thread = threading.Thread(target=thread_helper, args=(start, graph, results))
        threads.append(thread)
        thread.start()

    # Join the threads
    for thread in threads:
        thread.join()
    
    # Get the smallest found path
    smallest_tuple = min(results, key=lambda x: x[0])

    
    # Start the refinement loop
    for i in range(number_threads):
        # What number iteration are we on
        print("Running iteration " + str(i) +" of refinement")

        # Update the best starting node
        best_start_node = smallest_tuple[1][0]

        # Get the top X nodes closest to the best start node
        refine_start_nodes = get_closest_nodes(graph, best_start_node, number_threads)

        # Rerun the algorithm starting from each of the newly chosen start nodes
        refine_results = []
        refine_threads = []

        # Rerun the thread helper algo from each node with newly chosen best nodes
        for start_node in refine_start_nodes:
            thread = threading.Thread(target=thread_helper, args=(start_node, graph, refine_results))
            refine_threads.append(thread)
            thread.start()

        # Wait for all refined threads to finish
        for thread in refine_threads:
            thread.join()

        # Get the best solution found among all refinement threads
        refinement_smallest_tuple = min(refine_results, key=lambda x: x[0])

        # If we find a smaller start node, try again from that node
        if refinement_smallest_tuple[0] < smallest_tuple[0]:
            smallest_tuple = refinement_smallest_tuple
        else:
        # We didn't find a smaller start node, just return
            return save_graph_solution(smallest_tuple[0], smallest_tuple[1])
    
    # We ran through number threads use this return (unlikely)
    return save_graph_solution(smallest_tuple[0], smallest_tuple[1])

def get_closest_nodes(graph, start_node, num_nodes):
    """
    Get the top X nodes closest to a given start node

    Arguments:
        graph (array): 2-d array representing adj matrix
        start_node (int): The starting node
        num_nodes (int): The number of closest nodes to return

    Returns:
        closest_nodes (array): An array of the top num_nodes closest to the start_node
    """
    # Calculate distances from the start_node to all other nodes
    distances_from_start = []

    for node in range(len(graph)):
        # Verify we aren't adding start node
        if node != start_node:
            # Add distance information
            distance = graph[start_node][node]
            distances_from_start.append((node, distance))
    
    # Sort the distances so we can get the top ones
    distances_from_start.sort(key=lambda x: x[1])
    
    closest_nodes = []
    # Loop through distances and get associated nodes
    for node, _ in distances_from_start[:num_nodes]:
        # Make sure we have the node associated and not the distance it represents
        closest_nodes.append(node)
    
    # Return sorted closest nodes
    return closest_nodes

def random_cycle_distance(graph):
    """
    Generates a random cycle by shuffling the order of nodes
    It is worth noting there exsiststs a universe in which this works the perfectly first time, hopefully
    we live in that universe on Friday. It is extremely improbably this is going to work though

    Args:
        graph (array): 2-d array representing adj matrix

    Returns:
        total_distance (int): The total distance traveled in the random cycle
    """
    num_nodes = len(graph)
    # Generate a random order of nodes
    cycle = list(range(num_nodes))
    random.shuffle(cycle)

    # Calculate total distance traveled
    total_distance = 0
    for i in range(num_nodes):
        current_node = cycle[i]
        # Wrap around for the last node
        next_node = cycle[(i + 1) % num_nodes]  
        total_distance += graph[current_node][next_node]

    # Return randomly chosen path
    return save_graph_solution(total_distance, cycle)


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
            

