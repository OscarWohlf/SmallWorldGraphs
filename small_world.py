import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def fill_graph(graph,n,k,std):
    """
    Function for creating a graph, 
    where the number of edges per vertex is drawn from a normal distribution
    """
    for i in range(n):
        nbrs = -1
        while nbrs < 1:
            nbrs = np.random.normal(k,std)
            nbrs = int(np.floor(nbrs))
        if nbrs % 2 == 0: 
            for j in range(-int(nbrs/2), int(nbrs/2)+1):
                if j != 0:
                    graph[i][(i + j) % n] = 1
                    graph[(i+j) % n][i] = 1
        else:
            for j in range(-int((nbrs-1)/2), int((nbrs-1)/2)+2):
                if j != 0:
                    graph[i][(i + j) % n] = 1
                    graph[(i+j) % n][i] = 1
            graph[i][(i+(int((nbrs-1)/2)+1)) % n] = 1
            graph[(i+(int((nbrs-1)/2)+1)) % n][i] = 1

def get_valid_random(graph,n,j):
    """
    Helper function for finding a valid edge to rewire to.
    """
    rand_2 = 0
    while True:
        rand_2 = random.randint(0,n-1)
        if rand_2 == j:
            continue
        elif graph[j][rand_2] == 1:
            continue
        else:
            break
    return rand_2

def rewire_normal_graph(graph,p,n):
    """
    Function used to rewire a given graph, as explained in the paper.
    """
    rand_1 = 0
    rand_2 = 0
    new_adj_mat = graph.copy()
    for i in range(n):
        for j in range(n):
            if graph[i][j] == 0:
                continue
            else:
                rand_1 = random.random()
                if rand_1 <= p:
                    rand_2 = get_valid_random(new_adj_mat,n,i)
                    graph[i][j] = 0
                    graph[j][i] = 0

                    new_adj_mat[i][j] = 0
                    new_adj_mat[j][i] = 0
                    new_adj_mat[i][rand_2] = 1
                    new_adj_mat[rand_2][i] = 1

    return new_adj_mat

def compute_l(graph,n):
    """
    Function to calculate the characteristic path length as defined in the paper. 
    """
    nodes = list(range(n))
    dijk_graph = csr_matrix(graph)
    dist_matrix = dijkstra(csgraph = dijk_graph, directed = False, indices = nodes ,return_predecessors = False)
    
    sum = 0
    disconnected = 0
    for i in range(n):
        for j in range(i,n):
            if dist_matrix[i][j] == np.inf:
               disconnected += 1
               continue
            sum += dist_matrix[i][j]

    L = round(sum / ((n * (n - 1) - (disconnected)) / 2),3)
    return L

def compute_c(graph,n):
    """
    Function to calculate the clustering coefficient as defined in the paper. 
    """
    fracs = []
    for i in range(n):
        nbrs = []
        for j in range(n):
            if graph[i][j] == 1:
                nbrs.append(j)
        
        if len(nbrs) < 2:
            fracs.append(0.0)
            continue

        edges = 0
        max_edges = int(len(nbrs )* (len(nbrs)-1) / 2)
        for k in range(len(nbrs)):
            for l in range(k, len(nbrs)):
                if nbrs[k] == nbrs[l]: 
                    continue
                else:
                    if graph[nbrs[k]][nbrs[l]] == 1:
                        edges += 1
        frac_edges = edges / max_edges
        fracs.append(frac_edges)
    avg = sum(fracs)/len(fracs)
    return avg

def normalized_plot(n,k,std):
    """
    Function that was used the create the plot of normalized L and C shown in the paper.
    The function could take quite a long time to run for large values of n or for many realizations.
    """
    num_realizations = 10
    
    base_graph = np.zeros((n, n), dtype=int)
    fill_graph(base_graph, n,k,0)
    L_0 = compute_l(base_graph,n)
    C_0 = compute_c(base_graph,n)

    p_values = np.logspace(-4,0,20)
    
    L_ratios = []
    C_ratios = []
    for p in p_values:
        L_vals = []
        C_vals = []
        for _ in range(num_realizations):
            adj_matrix = np.zeros((n, n), dtype=int)
            fill_graph(adj_matrix,n,k,std)
            rewired_mat = rewire_normal_graph(adj_matrix,p,n)
            L_vals.append(compute_l(rewired_mat, n))
            C_vals.append(compute_c(rewired_mat, n))
        
        L_avg = np.mean(L_vals)
        C_avg = np.mean(C_vals)
        L_ratios.append(L_avg/L_0)
        C_ratios.append(C_avg/C_0)

    plt.figure(figsize=(8,6))
    plt.semilogx(p_values, L_ratios, 'o', color='black', label=r'$L(p)/L(0)$')
    plt.semilogx(p_values, C_ratios, 's', markerfacecolor='none', color='black', label=r'$C(p)/C(0)$')
    plt.xlabel('p')
    plt.ylabel('Normalized L and C')
    plt.title('Characteristic Path Length and Clustering Coefficient vs p')
    plt.legend()
    plt.grid(True)
    plt.show()

def distance_plot():
    """
    Function used to create the plot for the difference between the neural network 
    of C. Elegans and a modeled network, which was used in the paper. 
    """
    n = 282
    k = 14
    std = 1
    L_actual = 2.65
    C_actual = 0.28
    base_graph = np.zeros((n, n), dtype=int)
    fill_graph(base_graph,n,k,std)
    p_values = np.linspace(0.1,0.2,25)

    distances = []
    num_realizations = 25
    for p in p_values:
        dist_vals = []
        for _ in range(num_realizations):
            rewired_mat = rewire_normal_graph(base_graph.copy(),p,n)
            L = compute_l(rewired_mat,n)
            C = compute_c(rewired_mat,n)
            dist = (L-L_actual)**2 + (C-C_actual)**2
            dist_vals.append(dist)

        dist_mean = np.mean(dist_vals)
        distances.append(dist_mean)

    min_dist = min(distances)
    optimal_p = p_values[np.argmin(distances)]

    print("Optimal p: ",optimal_p, " with distance: ",min_dist)

    plt.figure(figsize=(8, 6))
    plt.semilogx(p_values,distances, 'o-',color='blue')
    plt.xlabel('p')
    plt.ylabel('Distance ((L-L_actual)^2 + (C-C_actual)^2)')
    plt.title('Distance to Actual Graph vs p')
    plt.grid(True)
    plt.show()

def main():
    n = 282     # number of vertices
    k = 15  # average number of edges 
    p = 0.14    # rewiring probability
    std = 1

    adj_matrix = np.zeros((n, n),dtype=int)

    fill_graph(adj_matrix,n,k,std)
    rewired_mat = rewire_normal_graph(adj_matrix,p,n)

    #Creating the plots used in the paper, generation could take a while for large graphs
    normalized_plot(n,k,std)
    distance_plot()

    #Plotting the graph
    G = nx.from_numpy_array(rewired_mat)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True,node_size=300,font_size=8)
    plt.show()
    
if __name__ == "__main__":
    main()