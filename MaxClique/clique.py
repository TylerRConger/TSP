from itertools import combinations

def read_graph(filename):
    """
    Read the graph file

    Arguments:
        filename (string): The name of the file
    Returns:
        graph (array): 2-d array representing adj matrix (square)
    """
    # Open the file and parse the data from each line to an array
    # each space is an item in the array
    graph = []
    with open(filename, 'r') as file:
        for line in file:
            data = list(map(int, line.split()))
            graph.append(data)
    return graph

def print_max_clique(max_clique):
    """
    Take a found max clique and print it out

    Arguments:
        max_clique: The clique that was found to be maximal

    Returns:
        None: Prints the clique out
    """
    if not max_clique:
        print("No clique found.")
    else:
        print("Maximum clique:", ', '.join(str(node) for node in max_clique))


def print_graph(graph):
    """
    Print out the graph

    Arguments:
        graph (array): 2-d array representing adj matrix (triangular)

    Returns:
        None: all data is printed
    """
    for row in graph:
        print(' '.join(map(str, row)))


def convert_to_symmetric_matrix(graph):
    """
    Convert a 2-d representing a lower triangular matrix into a square matrix
    This just makes it easier to index later on

    Arguments:
        graph (array): 2-d array representing adj matrix (triangular)

    Returns:
        graph (array): 2-d array representing adj matrix (square)
    """
    # Initialize symmetric matrix with 0 in each position
    n = len(graph)
    symmetric_matrix = [[0] * n for _ in range(n)] 

    # Fill in the info with current graph info
    for i in range(n):
        for j in range(i+1, n):
            symmetric_matrix[i][j] = graph[j][i]

    # Copy it across the diagonal
    for i in range(n):
        for j in range(i+1, n):
            symmetric_matrix[j][i] = symmetric_matrix[i][j]

    return symmetric_matrix

def convert_to_square_matrix(diagonal_matrix):
    """
    Convert a 2-d representing a lower triangular matrix into a square matrix
    This just makes it easier to index later on

    Arguments:
        graph (array): 2-d array representing adj matrix (triangular)

    Returns:
        graph (array): 2-d array representing adj matrix (square)
    """    
    # Delete the first element, as that's just the row number anyway
    for ele in diagonal_matrix:
        del ele[0]

    # reverse it so we have an upside down matrix
    for row in reversed(diagonal_matrix):
        row.append(0)

    # Call the helper function from previous project
    diagonal_matrix = convert_to_symmetric_matrix(diagonal_matrix)

    return diagonal_matrix
    
def is_clique(vertices, graph):
    """
    See if a set of nodes does in fact represent a clique.

    Arguments:
        vertices (list): A list of the vertices selected
        graph (2-d list): The associated matrix with the graph

    Returns:
        boolean: True if it is a clique
    """
    # Check if a clique exsists with the selected vertices
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not graph[vertices[i]][vertices[j]]:
                return False
    return True

def find_max_clique_brute_force(graph):
    """
    Brute force method to find a max clique, tries all possible combos.

    Arguments:
        graph (2-d array): The assocaited graph
    Returns:
        maxClique (set): Set of nodes in the max clique
    """
    max_clique = []
    for size in range(1, len(graph) + 1):
        # Test every possible combo
        for subset in combinations(range(len(graph)), size):
            if is_clique(subset, graph):
                max_clique = subset
    return max_clique

def find_max_clique(graph):
    """
    Order the graph by most connected node to least connected nodes. 
    Try to find the max clique by starting with the most connected and working down the list

    Arguments:
        graph (2-d array): The assocaited graph
    Returns:
        maxClique (set): Set of nodes in the max clique
    """
    largest_clique = set()
    nodes_by_degree = []

    # Create a list of nodes sorted by their degree
    for node in range(len(graph)):
        degree = sum(graph[node])
        nodes_by_degree.append((node, degree))

    # Sort the nodes by highest degree -> lowest degree
    nodes_by_degree.sort(key=lambda x: x[1], reverse=True)


    # Loop through the nodes and attempt add nodes to the clique
    for node, _ in nodes_by_degree:
        if graph[node]:
            clique = new_clique(graph, node, largest_clique)
            # Found a new maxclique
            if len(clique) > len(largest_clique):
                largest_clique = clique
        
    return largest_clique

def new_clique(graph, node, current_clique):
    """
    Attempt to increase the maxClque by adding a single node to it

    Arguments:
        graph (2-d array): The assocaited graph
        node (node value): The node to be added
        current_clique (set): The set of nodes in the clique
    Returns:
        maxClique (set): Set of nodes in the max clique (with or without the new node)
    """
    clique = set(current_clique)
    for n in current_clique:
        if not graph[node][n]:
            return current_clique
    clique.add(node)
    return clique

def invert_graph(graph):
    """
    Convert graph to G' by switching the values

    Arguments:
    graph (2-d array): The associated graph

    Returns:
    graph (2-d array): G' the inverse graph
    """
    inverted_graph = []
    for row in graph:
        inverted_row = []
        for edge in row:
            # Switch 1 and 0 
            inverted_row.append(1 - edge)
        # Piece the graph back togeter
        inverted_graph.append(inverted_row)
    return inverted_graph

def invert_graph_adjlist(graph):
    """
    Convert graph to G' by adding each opposing node to the node list
    Uses the adjlist graph not the adjmat graph

    Arguments:
    graph (2-d array): The associated graph

    Returns:
    graph (2-d array): G' the inverse graph
    """
    inverted_graph = [[] for _ in range(len(graph))]
    
    for u, edges in enumerate(graph):
        for v in edges:
            inverted_graph[v].append(u)
    
    return inverted_graph

def find_vertex_cover(graph):
    """
    https://www.geeksforgeeks.org/introduction-and-approximate-solution-for-vertex-cover-problem/
    Find a vertex cover following geeks for geeks algo
    This method only works with adjlist and not adjmat based graphs.

    Arguments:
    graph (2-d array): The associated graph

    Returns:
    vertex_cover (set): A set of the vertex cover
    """
    vertex_cover = set()
    visited = [False] * len(graph)

    # Consider all of the edges
    for u in range(len(graph)):
        # Pick an edge
        if not visited[u]:
            # Loop over adjacent vertices of u and pick the first not yet visited vertex
            for v in graph[u]:
                if not visited[v]:
                    # Add the vertices (u, v) to the result set
                    vertex_cover.add(u)
                    vertex_cover.add(v)
                    # Mark the vertices u and v as visited so that all edges from/to them would be ignored
                    visited[u] = True
                    visited[v] = True
                    break

    return vertex_cover


if __name__ == "__main__":
    filename = ""  # File name goes here
    graph = read_graph(filename)

    #graph = convert_to_square_matrix(graph)


    #graph = invert_graph(graph)



    #graph = invert_graph_adjlist(graph)



    print(find_max_clique(graph))
    