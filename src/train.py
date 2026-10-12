import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from model import EllipticGCN

# --------------------------------------------------
# Load graph
# --------------------------------------------------
data = torch.load("data/graph_data.pt", weights_only=False)

# --------------------------------------------------
# Normalize features
# --------------------------------------------------
scaler = StandardScaler()
data.x = torch.tensor(
    scaler.fit_transform(data.x.numpy()),
    dtype=torch.float
)

# --------------------------------------------------
# Use ONLY labeled nodes for supervision
# --------------------------------------------------
labeled_idx = torch.where(data.y >= 0)[0]
labels = data.y[labeled_idx]

print("Total nodes        :", data.num_nodes)
print("Labeled nodes      :", labeled_idx.numel())
print("Class distribution :", torch.bincount(labels, minlength=2))

# --------------------------------------------------
# Stratified train/test split on labeled nodes
# --------------------------------------------------
train_idx, test_idx = train_test_split(
    labeled_idx.numpy(),
    test_size=0.2,
    stratify=labels.numpy(),
    random_state=42
)

train_idx = torch.tensor(train_idx)
test_idx = torch.tensor(test_idx)

print("Train distribution :", torch.bincount(data.y[train_idx], minlength=2))
print("Test  distribution :", torch.bincount(data.y[test_idx], minlength=2))

# --------------------------------------------------
# Class weights (handle imbalance)
# --------------------------------------------------
class_counts = torch.bincount(labels, minlength=2).float()
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum()

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = EllipticGCN()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003,
    weight_decay=5e-4
)

# --------------------------------------------------
# Training
# --------------------------------------------------
for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_idx], data.y[train_idx])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)

probs = torch.softmax(out[test_idx], dim=1)[:, 1].numpy()
true = data.y[test_idx].numpy()

print("\nEvaluation Results")
print("------------------")
print("ROC-AUC   :", roc_auc_score(true, probs))
print("Precision :", precision_score(true, probs > 0.5))
print("Recall    :", recall_score(true, probs > 0.5))

torch.save(model.state_dict(), "data/elliptic_gcn.pth")
print("Model saved successfully")