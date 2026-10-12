import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Load data
# --------------------------------------------------
features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
edges = pd.read_csv("data/elliptic_txs_edgelist.csv")
classes = pd.read_csv("data/elliptic_txs_classes.csv")

features.rename(columns={0: "txId", 1: "time"}, inplace=True)

# Merge (KEEP unknown labels)
df = features.merge(classes, on="txId", how="left")

df["class"] = pd.to_numeric(df["class"], errors="coerce")

# --------------------------------------------------
# Feature preprocessing
# --------------------------------------------------
X = df.iloc[:, 2:168].apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

X.columns = X.columns.astype(str)

# Normalize features (VERY IMPORTANT)
scaler = StandardScaler()
X = scaler.fit_transform(X)

x = torch.tensor(X, dtype=torch.float)

# --------------------------------------------------
# Label preprocessing
# --------------------------------------------------
# -1 = unknown, 0 = licit, 1 = illicit
y = torch.full((len(df),), -1, dtype=torch.long)
y[df["class"] == 2] = 0
y[df["class"] == 1] = 1

# --------------------------------------------------
# Time
# --------------------------------------------------
time = torch.tensor(df["time"].values, dtype=torch.long)

# --------------------------------------------------
# Graph construction
# --------------------------------------------------
id_map = {tx: i for i, tx in enumerate(df.txId)}

edge_list = []
for _, row in edges.iterrows():
    if row.txId1 in id_map and row.txId2 in id_map:
        edge_list.append([id_map[row.txId1], id_map[row.txId2]])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# --------------------------------------------------
# Create graph object
# --------------------------------------------------
data = Data(x=x, edge_index=edge_index, y=y)
data.time = time

torch.save(data, "data/graph_data.pt")

print("Preprocessing complete")
print("Total nodes      :", data.num_nodes)
print("Labeled nodes    :", (data.y >= 0).sum().item())
print("Illicit samples  :", (data.y == 1).sum().item())
