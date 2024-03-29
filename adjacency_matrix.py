import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt, dist
import scipy.spatial as spatial
import numpy as np

from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

# Grid cell size
grid_size = 1

# Graph type
graph_type = 'directed'

# Create two 1D arrays of values
x = np.arange(0, 4, grid_size)
y = np.arange(0, 4, grid_size)

# Create a 2D grid of coordinates
X, Y = np.meshgrid(x, y)
points = list(zip(X.flatten(),Y.flatten()))

# Obstables list
obs_idx_list = [5, 6, 10]

# Parallel adjaceny matrix
def adjacency_matrix(points, obs_idx_list):        
    point_tree = spatial.cKDTree(np.array(points))
    adj_mat_sparse = []

    def par_adjacency(i, points, obs_idx_list):      
        idxs = point_tree.query_ball_point(points[i], GRID_SIZE * sqrt(2.0))
        for j in idxs:
            if i!=j and not (i in obs_idx_list or j in obs_idx_list):
                adj_mat_sparse.append([dist(points[i], points[j]), i, j])
    
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores, require='sharedmem')(delayed(par_adjacency)(i, points, obs_idx_list)
                                        for i in tqdm(range(len(points))))
    
    adj_mat_sparse = np.vstack(adj_mat_sparse)
    adjacency_mat = coo_matrix((adj_mat_sparse[:,0], (adj_mat_sparse[:,1], 
                                                      adj_mat_sparse[:,2])), 
                                                      shape=(len(points), len(points)))    

    return adjacency_mat

# Adjacency matrix
adj_mat = adjacency_matrix(points, obs_idx_list)

# Graph (G)
if graph_type == 'undirected':
    gtype = nx.Graph()
else:
    gtype = nx.DiGraph()
    
G = nx.from_scipy_sparse_array(adj_mat, parallel_edges=False, create_using=gtype, edge_attribute='weight')

# Find the shortest path
short_path = nx.shortest_path(G, source=0, target=14, weight='weight', method='dijkstra')

# Create custom layout
pos = {i: point for i, point in enumerate(points)}

# Plot the graph connectivity
fig, ax = plt.subplots()
nx.draw(G, pos=pos, node_color='k', ax=ax)
nx.draw(G, pos=pos, node_size=1500, ax=ax)  # draw nodes and edges
nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names
labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)]) #create custom weigth label for decimals
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax) # draw network
plt.axis("on")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()