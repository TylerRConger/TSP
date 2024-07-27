# Traveling Salesman Problem Solver

This repo contains a write up about my solution to the Traveling Salesman problem as well as including the associated code for my solution to the TSP. 

## Introduction

The basic idea of the Traveling Salesman Problem (TSP) is a salesman is given a list of cities and must visit each individual city. What is the optimal route to visit each of these different cities. In this 'graph' each city is connected to every other city within the network, hence a fully connected graph. The salesman must visit each city only one time and return to his original starting location, with the goal of the smallest possible route between them. 

## Information

- Author: Tyler Conger
- Date: 2.23.2024

## Code Overview

There are various ways to solve the TSP and as such three seperate algorithms have been included to solve the problem in different ways. As it is extremely difficult to find a 'true' solution there are various heuristics that have been created to solve them. This codebase covers a brute force method, the nearest neighbor method, as well as a self made heuristic algorithm, and finally an extremely quick 'random cycle' method that just works to generate a potential path. There are also other various helper methods that were created to both help setup and create the graphs as well as work with them. The sections are marked below.

0. Graph Functions
1. Brute Force
2. Nearest Neighbor
3. Heuristic Approach
4. Random Cycle
5. Solution Verification

## Implementation Details

### 0. Graph Related Functions

- **`generate_graph(num_points)`**: Generates a fully connected graph with random weights between 1 and 500.
- **`write_graph_to_file(graph, filename)`**: Writes the graph to a file.
- **`generate_and_write_graph(num_points)`**: Generates a fully connected graph and writes it to a file.
- **`read_graph_from_file(filename)`**: Reads the graph from a file.
- **`print_graph(graph)`**: Prints the graph in the format it appears in the file.


### 1. Brute Force Attempt

- **`calculate_total_distance(cycle, graph)`**: Calculates the total distance traveled for a given group of cities.
- **`brute_force_tsp(graph)`**: Implementation of the brute force algorithm for the TSP.

### 2. Nearest Neighbor

- **`nearest_neighbor_helper(start_point, graph)`**: Nearest Neighbor wrapper function.
- **`nearest_neighbor(start_point, graph)`**: Implementation of the Nearest Neighbor algorithm.

### 3. Heuristic Approach

- **`thread_helper(start_point, graph, results)`**: Used to be targeted by the heuristic algorithm.
- **`heuristic_approach(graph)`**: Implementation of a self-created shortest path algorithm.
- **`get_closest_nodes(graph, start_node, num_nodes)`**: Get the top X nodes closest to a given start node.

### 4. Random Cycle

- **`random_cycle_distance(graph)`**: Generates a random cycle by shuffling the order of nodes.

### 5. Verification and Saving of Solution

- **`convert_to_symmetric_matrix(graph)`**: Converts a 2D array representing a lower triangular matrix into a square matrix.
- **`verify_solution(graph, solution_filename)`**: Verifies that reported distance traveled matches actual distance traveled.
- **`save_graph_solution(distance, pathway)`**: Saves the TSP solution to a file.


## Method Calls

### Code Start

The provided code starts by generating a fully connected graph with a specified number of points. The following method calls are made:

1. Generate and write a graph to a file.
2. Read the graph from the file.
3. Perform TSP solving algorithms:
    - Brute Force
    - Nearest Neighbor
    - Heuristic Approach
    - Random Cycle

The execution time for each algorithm is also measured.


### Modifying the information

To modify the output or data one simply opens the python file and changes information as needed, each section to be changed by the user is marked appriopriately. Some commonly changed information may be as follows.

1. The size of the graph
This can be changed by modifying the variable 'currentGraphSize'

2. Method used for solution
This can be modified by un-commenting the appriopriately needed solution, e.g. removing the # at the begining of the line associated with the run for brute force will also run the brute force algorithm. 

