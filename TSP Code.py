# Tyler Conger
# CS 570
# TSP Problem Homework
# 2.16.2024

import random
import itertools
import time
import threading

# Generation and Printing of Graphs

def generate_graph(num_points):
    """
    Generates a fully connected graph with random weights between 1 and 500
    
    Arguments:
        num_points (int): The number of vertices in the graph
    
    Returns:
        array: A 2D array representing the lower triangular part of the adjacency matrix of the graph
    """
    # Initialize an empty 2D array to represent the adjacency matrix
    graph = [[0] * num_points for _ in range(num_points)]
    
    # Iterate over each pair of vertices
    for i in range(num_points):
        for j in range(i, num_points):  # Modified to start from i
            if i != j:  # Modified to exclude diagonal
                # Generate a random weight between 1 and 500
                weight = random.randint(1, 500)
                
                # Assign the same weight to both directions since it's a fully connected graph
                graph[i][j] = weight
                graph[j][i] = weight
                
    return graph

def write_graph_to_file(graph, filename):
    """
    Writes the lower triangular part of the graph to a file
    
    Arguments:
        graph (array): A 2D array representing the lower triangular part of the adjacency matrix of the graph
        filename (str): The name of the file to write to
    """
    with open(filename, 'w') as f:
        # Write each row of the lower triangular part of the adjacency matrix to the file
        for i in range(len(graph)):
            for j in range(i + 1):  # Modified to include up to i
                f.write(str(graph[i][j]) + ' ')
            f.write('\n')

def generate_and_write_graph(num_points):
    """
    Generates a fully connected graph with random weights and writes it to a file
    
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
    Read the graph from a file
    
    Arguments:
        filename (str): The name of the file containing the graph data
    
    Returns:
        array: A 2D array representing the adjacency matrix of the graph
    """
    graph = []
    with open(filename, 'r') as f:
        for line in f:
            # Split the line by spaces and convert each element to an integer
            row = list(map(int, line.strip().split()))
            graph.append(row)
    return graph

def print_graph(graph):
    """
    Print the graph in the format it appears in the file
    
    Arguments:
        graph (array): A 2D array representing the adjacency matrix of the graph
    """
    print("\n\n=================================================\n\n")
    print("Printing a graph of size "  + str(len(graph)) + ":\n")
    print("=================================================")
    for row in graph:
        print(' '.join(map(str, row)))

    print("=================================================\n\n")

# Save the solution
    
def save_graph_solution(distance, permutation):
    """
    Save the TSP solution to a file
    
    Arguments:
        distance (int): The total distance traveled in the TSP solution
        permutation (tuple): The optimal permutation representing the tour

    Returns:
        filename (string): The associated filename
    """
    filename = "s" + str(distance) + "_tconger1.sol"

    with open(filename, 'w') as f:
        # Write the permutation to the file
        f.write(' '.join(map(str, permutation)) + " " + str(permutation[0]) +'\n')  # Add the starting point at the end
    
    return filename

# Brute Force attempt
        
def calculate_total_distance(permutation, graph):
    """
    Calculate the total distance traveled for a given permutation of cities
    
    Arguments:
        permutation (tuple): A permutation of point indices
        graph (array): A 2D array representing the adjacency matrix of the graph
    
    Returns:
        int: The total distance traveled for the given permutation
    """
    total_distance = 0
    num_vertecies = len(graph)
    for i in range(len(permutation)):
        # Use modular arithmetic to handle indexing for lower triangular matrix
        from_point = permutation[i]
        to_point = permutation[(i + 1) % num_vertecies]
        total_distance += graph[max(from_point, to_point)][min(from_point, to_point)]

    return total_distance

def brute_force_tsp(graph):
    """
    Implementation of the brute force algorithm
    
    Arguments:
        graph (array): A 2D array representing the adjacency matrix of the graph

    Returns:
        filename (string): The associated filename

    """
    num_vertecies = len(graph)
    shortest_distance = float('inf')
    shortest_permutation = None
    
    # Generate all possible permutations of point indices
    all_permutations = itertools.permutations(range(num_vertecies))

    # Iterate through each permutation and calculate the total distance
    for permutation in all_permutations:

        
        # Make sure to capture the return trip aswell
        # update_permutation = permutation + (permutation[0])

        distance = calculate_total_distance(permutation, graph)
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_permutation = permutation
    
    # Save the solution to a file
    return save_graph_solution(shortest_distance, shortest_permutation)

# Verification and Graphing

def convert_to_symmetric_matrix(input_array):
    """
    Convert a array of arrays representing an upper triangular matrix into a symmetric matrix

    Arguments:
        input_array (array): A array of arrays representing an upper triangular matrix where each inner array contains the elements of a row

    Returns:
        array: A 2-D array of the symmetric matrix
    """
    n = len(input_array)
    symmetric_matrix = [[0] * n for _ in range(n)]  # Initialize symmetric matrix with zeros

    # Fill in the upper triangular part of the symmetric matrix
    for i in range(n):
        for j in range(i+1, n):
            symmetric_matrix[i][j] = input_array[j][i]

    # Fill in the lower triangular part by transposing the upper triangular part
    for i in range(n):
        for j in range(i+1, n):
            symmetric_matrix[j][i] = symmetric_matrix[i][j]

    return symmetric_matrix

def verify_solution(graph, solution_filename):
    """
    Verify the correctness of a solution file for a given graph
    
    Arguments:
        graph (array): A 2D array representing the adjacency matrix of the graph
        solution_filename (str): The filename of the solution file
    
    Returns:
        bool: True if the expected matches the actual distance traveled
    """
    # Read the solution from the solution file name
    solution_distance = int(solution_filename.split('_')[0][1:])  # Extract the distance from the filename
    with open(solution_filename, 'r') as f:
        solution_permutation = list(map(int, f.readline().strip().split()))
    
    # Calculate the total distance traveled based on the solution permutation
    total_distance = 0
    num_vertecies = len(graph)

    # TODO: Using - 1 as unsure if the final trip back home counts as a path or is necessary
    for i in range(len(solution_permutation) - 1):
        from_point = solution_permutation[i]
        to_point = solution_permutation[(i + 1) % num_vertecies]
        total_distance += graph[max(from_point, to_point)][min(from_point, to_point)]
    
        #print(total_distance)

    # Verify the correctness of the solution
    if total_distance == solution_distance:
        print("Solution is correct. Total distance:", total_distance)
        return True
    else:
        print("Solution is incorrect. Calculated distance:", total_distance, "Expected distance:", solution_distance)
        return False

# Nearest Neighbor
    
def nearest_neighbor_helper(start_point, graph):
    """
    Nearest Neighbor wrapper function

    Arguments:
        graph (array): A 2D array representing the adjacency matrix of the graph
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

    Arguments:
        graph (array): A 2D array representing the adjacency matrix of the graph
        start_point(int): Position to start traveling from

    Returns:
        total_distance (int): The distance of the tour taken
        tour (array): The tour that was taken
    """
    # Save graph len for later as recalcuating takes time
    graph_len = len(graph)
    
    # Set all nodes to unvisited
    visited = [False] * graph_len

    # Start the tour
    tour = [start_point]
    
    # Start with the chosen start point (sometimes 0, sometimes randomly chosen)
    current_point = start_point
    visited[current_point] = True
    
    total_distance = 0
    
    # Repeat until all cities have been visited
    while len(tour) < graph_len:
        nearest_point = None
        min_distance = float('inf')  # Initialize min_distance to infinity
        
        # Find the nearest unvisited point
        for point in range(graph_len):
            if not visited[point] and graph[current_point][point] < min_distance:
                nearest_point = point
                min_distance = graph[current_point][point]
        
        # Move to the nearest unvisited point
        tour.append(nearest_point)
        visited[nearest_point] = True
        total_distance += min_distance
        
        # Update current point
        current_point = nearest_point
    
    # Add distance from the last point back to the starting point to complete the trip
    total_distance += graph[tour[-1]][start_point]

    return total_distance, tour
 # My Solution

def thread_helper(start_point, graph, results):
    """
    Used to be targeted by the heuristic algo

    This function executes the nearest_neighbor function and stores its result in the shared results list
    """
    result = nearest_neighbor(start_point, graph)
    results.append(result)

def heuristic_approach(graph):
    """
    Implementation of my own self created shortest path algorithm.

    Spawn threads and start each thread at a random spot then attempt to solve the graph using nearest neighbor

    Arguments:
        graph (array): A 2D array representing the graph

    Returns:
        filename (string): The associated filename
    """

    number_threads = 10
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
        thread.join
    
    # Get the smallest found path
    smallest_tuple = min(results, key=lambda x: x[0])

    best_start_node = smallest_tuple[1][0]

    # Refinement step: Get the top X nodes closest to the best start node
    refinement_start_nodes = get_closest_nodes(graph, best_start_node, number_threads)

    # Rerun the algorithm starting from each of the refinement start nodes
    refinement_results = []
    refinement_threads = []
    for start_node in refinement_start_nodes:
        thread = threading.Thread(target=thread_helper, args=(start_node, graph, refinement_results))
        refinement_threads.append(thread)
        thread.start()

    # Wait for all refinement threads to finish
    for thread in refinement_threads:
        thread.join()

    # Get the best solution found among all refinement threads
    refinement_smallest_tuple = min(refinement_results, key=lambda x: x[0])

    # Compare the best solution found among all threads with the best solution found during refinement
    if refinement_smallest_tuple[0] < smallest_tuple[0]:
        return save_graph_solution(refinement_smallest_tuple[0], refinement_smallest_tuple[1])
    else:
        return save_graph_solution(smallest_tuple[0], smallest_tuple[1])

def get_closest_nodes(graph, start_node, num_nodes):
    """
    Function to get the top X nodes closest to a given start node.

    Arguements:
        graph (list of lists): The adjacency matrix representing the graph.
        start_node (int): The starting node.
        num_nodes (int): The number of closest nodes to return.

    Returns:
        list: A list of the top X nodes closest to the start node.
    """
    distances = [(node, graph[start_node][node]) for node in range(len(graph)) if node != start_node]
    distances.sort(key=lambda x: x[1])
    return [node for node, _ in distances[:num_nodes]]

def random_tour_distance(graph):
    """
    Generates a random tour by shuffling the order of nodes
    It is worth noting there exsiststs a universe in which this works the perfectly first time, hopefully
    we live in that universe on Friday. It is extremely improbably this is going to work though


    Args:
        graph (2-D Array): The adjacency matrix representing the graph

    Returns:
        total_distance (int): The total distance traveled in the random tour
    """
    num_nodes = len(graph)
    # Generate a random order of nodes
    tour = list(range(num_nodes))
    random.shuffle(tour)

    # Calculate total distance traveled
    total_distance = 0
    for i in range(num_nodes):
        current_node = tour[i]
        next_node = tour[(i + 1) % num_nodes]  # Wrap around for the last node
        total_distance += graph[current_node][next_node]

    return save_graph_solution(total_distance, tour)

# Method Calls

# CODE START

# Declare an appropriate graph size, change this value based on what the selected graph is
currentGraphSize = 4000
generate_and_write_graph(currentGraphSize)  # Generates a fully connected graph with 5 points and writes it to a file
graph = read_graph_from_file("Size" + str(currentGraphSize) + ".graph")
#print_graph(graph)

sym_graph = convert_to_symmetric_matrix(graph)

#print(sym_graph)

start = time.time()
#file = brute_force_tsp(graph)
#file = nearest_neighbor_helper(2, sym_graph)
#file = heuristic_approach(sym_graph)
file = random_tour_distance(sym_graph)

end = time.time()

print(f"{end-start}")

# Verify that we got the correct solution
if verify_solution(graph, file):
    print(f"It took {end-start} seconds to reach the correct answer")