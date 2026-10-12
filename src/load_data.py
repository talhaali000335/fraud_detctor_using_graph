import pandas as pd

features = pd.read_csv("data/elliptic_txs_features.csv", header=None)
edges = pd.read_csv("data/elliptic_txs_edgelist.csv")
classes = pd.read_csv("data/elliptic_txs_classes.csv")

print("Features shape:", features.shape)
print("Edges shape:", edges.shape)
print("Classes shape:", classes.shape)

print("\nClasses distribution:")
print(classes["class"].value_counts())
