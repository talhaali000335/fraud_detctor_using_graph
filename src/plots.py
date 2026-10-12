import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from model import EllipticGCN

# -----------------------------
# Load data and model
# -----------------------------
data = torch.load("data/graph_data.pt", weights_only=False)

scaler = StandardScaler()
data.x = torch.tensor(
    scaler.fit_transform(data.x.numpy()),
    dtype=torch.float
)

model = EllipticGCN()
model.load_state_dict(torch.load("data/elliptic_gcn.pth"))
model.eval()

# Use labeled test nodes (same logic as training)
labeled_idx = torch.where(data.y >= 0)[0]
test_idx = labeled_idx[int(0.8 * len(labeled_idx)):]

with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out[test_idx], dim=1)[:, 1].numpy()
    true = data.y[test_idx].numpy()

# -----------------------------
# 1️⃣ ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(true, probs)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – GNN Fraud Detection")
plt.show()

# -----------------------------
# 2️⃣ Precision-Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(true, probs)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()

# -----------------------------
# 3️⃣ Confusion Matrix
# -----------------------------
preds = (probs > 0.5).astype(int)
cm = confusion_matrix(true, preds)

disp = ConfusionMatrixDisplay(cm, display_labels=["Licit", "Fraud"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 4️⃣ Threshold vs Precision & Recall
# -----------------------------
thresholds = [0.1 * i for i in range(1, 10)]
precisions, recalls = [], []

for t in thresholds:
    p = (probs > t).astype(int)
    cm = confusion_matrix(true, p)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precisions.append(tp / (tp + fp + 1e-6))
    recalls.append(tp / (tp + fn + 1e-6))

plt.figure()
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs Precision & Recall")
plt.legend()
plt.show()
