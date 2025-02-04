# Create a dataset of graphs from the first 500 spectra 

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv

import torch.nn as nn
import itertools
import torch.optim as optim



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
      a simple distance_matrix function is defined. The graph is stored in a torch_geometric.data.Data object. 

"""
num_spectra = 500
for j in range(num_spectra):
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


list_graph = torch.load("/Users/loernzopasquetto/Desktop/UZH thesis/Data_torch.pt")




class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # Small constant to avoid division by zero

    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (target + self.eps))) * 100

### model ###



class GATv2Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=4, concat=False, edge_dim=1)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch_idx = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat1(x, edge_index, edge_attr).relu()
        x = self.gat2(x, edge_index, edge_attr).relu()
        x = global_mean_pool(x, batch_idx)
        return self.fc(x)




class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gat = pyg_nn.GATConv(in_dim, hidden_dim, heads=4, concat=False, edge_dim=1)  
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch_idx = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch_idx)  # Pooling over nodes per graph
        return self.fc(x)

model = GATv2Regressor(in_dim=2, hidden_dim=32, out_dim=6)



### training ###



best_val_loss = float('inf')  # Initialize best loss as infinity
save_path = "best_gnn_model.pth"  # Path to save model



batch_size = 32  

loader = DataLoader(graphs_[:6000], batch_size=batch_size, shuffle=True)
loader_val = DataLoader(graphs_[:6000], batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MAPELoss()


for epoch in range(100):
    model.train()
    total_loss_ep = 0
  
    if epoch%10 == 0: print("Epoch: ", epoch)
    total_loss_ep = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)  # Now shape (B, 6)
        loss = criterion(pred, batch.y.view(pred.shape))  # Ensure same shape
        loss.backward()
        optimizer.step()
        total_loss_ep += loss.item()
    total_loss_ep /= len(loader)
    #print(f"Epoch {epoch}, Loss: {total_loss_ep:.4f}")


    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in loader_val:
            batch = batch.to(device)
            pred = model(batch)
            val_loss += criterion(pred, batch.y.view(pred.shape))

    val_loss /= len(loader_val)
    print(f"Epoch {epoch}, Loss: {total_loss_ep:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_gnn_model.pth")
        print("Model Saved!")



### load model ###

model = GATv2Regressor(in_dim=2, hidden_dim=32, out_dim=6)
model.load_state_dict(torch.load("best_gnn_model.pth"))






