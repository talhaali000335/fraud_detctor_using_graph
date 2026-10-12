import torch
from sklearn.preprocessing import StandardScaler
from model import EllipticGCN

# --------------------------------------------------
# Load graph data
# --------------------------------------------------
data = torch.load("data/graph_data.pt", weights_only=False)

# --------------------------------------------------
# Normalize features (MUST match training)
# --------------------------------------------------
scaler = StandardScaler()
data.x = torch.tensor(
    scaler.fit_transform(data.x.numpy()),
    dtype=torch.float
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = EllipticGCN()
model.load_state_dict(torch.load("data/elliptic_gcn.pth"))
model.eval()

# --------------------------------------------------
# Select nodes for inference
# (example: last 10 labeled transactions)
# --------------------------------------------------
labeled_idx = torch.where(data.y >= 0)[0]
infer_idx = labeled_idx[-10:]

# --------------------------------------------------
# Run inference
# --------------------------------------------------
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out[infer_idx], dim=1)

# --------------------------------------------------
# Display predictions
# --------------------------------------------------
print("\nInference Results")
print("------------------")

for i, idx in enumerate(infer_idx):
    fraud_prob = probs[i, 1].item()
    label = "FRAUD" if fraud_prob > 0.5 else "LICIT"

    print(
        f"Transaction Node {idx.item():>6} | "
        f"Fraud Probability: {fraud_prob:.4f} | "
        f"Prediction: {label}"
    )
