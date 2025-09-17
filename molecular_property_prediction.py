"""
Molecular Property Prediction with Graph Neural Networks (GNN)
Dataset: QM9 (PyTorch Geometric)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset = QM9(root="data/QM9")
target_idx = 0  # predict property 0 (U0 energy)

# Train/test split
train_dataset = dataset[:10000]
test_dataset = dataset[10000:12000]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------
# 2. Define GNN model
# -------------------------------
class GNN(nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)  # output one property

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)   # aggregate node embeddings
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                  # shape: [batch_size, 1]
        return x.squeeze()               # shape: [batch_size]

# -------------------------------
# 3. Setup training
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN(dataset.num_features).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# -------------------------------
# 4. Training loop
# -------------------------------
for epoch in range(5):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y[:, target_idx])  # target shape = [batch_size]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

# -------------------------------
# 5. Evaluation
# -------------------------------
model.eval()
total_loss = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y[:, target_idx])
        total_loss += loss.item()

print(f"Test Loss: {total_loss/len(test_loader):.4f}")