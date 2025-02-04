# Create a dataset of graphs from the first 500 spectra 

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import itertools



def distance_matrix(peaks, N =20):
  """
  peaks is the array of length 20 in which the entries are the position of the peaks. N defines the dimension of the distance matrix.
  """
      
    init = np.zeros(shape=(N,N))
    for i in range(N):
            for j in range(N):
                init[i, j] = abs(peaks[i]- peaks[j]) * 0.02
                init[j, i] = init[i, j] 
    return init






graphs_ = []


N = 20 # 20 nodes corresponds to 20 peaks
edges = list(itertools.combinations_with_replacement(range(N), 2))

edge_index = torch.tensor([
    [i[0] for i in edges] + [i[1] for i in edges],  
    [i[1] for i in edges] + [i[0] for i in edges]   
], dtype=torch.long)
"""
Note: to create a graph we need: number and attributes of the nodes, type of connectivity (fully connected, sparse connected, etc.) and the properties of the connections
      described by edges_attr. In our case the nodes have two attributes, peak position and intensity. The connectivity is full with self connection and is defined by 
      the edge_index tensor. This consists of two arrays, the sender and the receiver. The edge attribute is the distance between the sender and the receiver. For that, 
      a simple distance_matrix function is defined.

"""
for j in range(500):
    if j%20 == 0: print("Spectra: ", j)
    peaks, properties = find_peaks(X_data[j, 1:], height=10)

    if len(peaks) > 20:
        x = np.vstack([properties["peak_heights"][:20], peaks[:20]]).transpose()
        x = torch.tensor(x, dtype=torch.float)

        edges_feat = []
        for i in indx:
            edges_feat.append(distance_matrix(peaks, N)[i[0], i[1]])
        
        edges_attr = torch.tensor(edges_feat, dtype = torch.float)
        edges_attr = torch.reshape(edges_attr, (len(edges_attr), 1))

        graphs_.append(Data(x = x, edge_index= edge_index, edge_attr=edges_attr, y=torch.tensor(y_data[j, :], dtype=torch.float)))
