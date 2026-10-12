# 🚨 Graph Neural Network for Fraud Detection

This project implements a **Graph Neural Network (GNN)** to detect **fraudulent financial transactions** by modeling transactions as nodes in a graph and learning relational patterns using **PyTorch Geometric**.

The model is trained and evaluated on the **Elliptic Bitcoin Transaction Dataset**, a real-world dataset widely used in fraud detection research.

---

## 📌 Project Overview

Traditional machine learning models treat transactions independently. However, fraud often emerges from **relationships between transactions**.  
This project addresses that limitation by:

- Modeling transactions as a **graph**
- Applying a **Graph Convolutional Network (GCN)**
- Leveraging **neighborhood information** to improve fraud detection

---

## 🧠 Key Features

- Graph-based modeling of financial transactions
- GNN architecture using **GCNConv**
- Handles **partially labeled data** using label masking
- Class imbalance handled with **weighted loss**
- End-to-end pipeline: preprocessing → training → evaluation → inference
- Strong performance on real-world fraud data

---

## 📂 Project Structure

Fraud Detection/
│
├── src/
│ ├── model.py # GNN architecture (GCN)
│ ├── build_graph.py # Data preprocessing & graph construction
│ ├── train.py # Training & evaluation pipeline
│ ├── inference.py # Inference on unseen transactions
│ ├── plots.py # ROC, PR curve, confusion matrix
│ └── init.py
│
├── data/
│ ├── elliptic_gcn.pth # Trained model
│ └── graph_data.pt # Preprocessed graph
│
└── README.md


---

## 📊 Dataset

**Elliptic Bitcoin Transaction Dataset**

- Nodes: Bitcoin transactions
- Edges: Money flow between transactions
- Labels:
  - `1` → Illicit (fraud)
  - `0` → Licit
  - `-1` → Unknown (unlabeled)

### Dataset Statistics
- Total nodes: ~203,000
- Labeled nodes: ~46,500
- Fraudulent transactions: ~4,500

Unlabeled nodes are **retained in the graph** to support message passing, but are **excluded from loss computation**.

---

## 🔧 Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- Matplotlib
- Jupyter Notebook (for experimentation & visualization)

---

## 🏗️ Graph Construction

- Nodes represent **transactions**
- Edges represent **transaction-to-transaction money flow**
- Users and accounts are **implicitly modeled** via graph connectivity
- Features are normalized using `StandardScaler`

---

## 🧩 Model Architecture

The GNN is implemented using a **Graph Convolutional Network (GCN)**:

- Input layer: Transaction features (166 dimensions)
- Hidden layer: 64 units with ReLU activation
- Output layer: Binary classification (Fraud / Non-Fraud)

```python
GCNConv → ReLU → GCNConv

## 🏋️ Training Strategy

- **Loss Function:** Cross-Entropy Loss with class weights to handle class imbalance  
- **Optimizer:** Adam  
- **Learning Rate:** 0.003  
- **Label Masking:** Training performed only on labeled nodes while keeping unlabeled nodes for message passing  
- **Data Split:** Stratified train–test split applied on labeled data  

---

## 📈 Evaluation Metrics

The model is evaluated using standard fraud detection metrics:

- **ROC-AUC**
- **Precision**
- **Recall**

### Final Results

ROC-AUC   : 0.968
Precision : 0.560
Recall    : 0.899


These results demonstrate **excellent fraud detection performance**, with a strong emphasis on **recall**, which is crucial in financial fraud scenarios where missing fraudulent transactions is costly.

---

## 🔍 Inference

After training, the model can be used to predict **fraud probabilities for unseen transactions** using the trained GNN.

### Example Output

Transaction Node 203736 | Fraud Probability: 0.5513 | Prediction: FRAUD
Transaction Node 203752 | Fraud Probability: 0.0001 | Prediction: LICIT


---

## ▶️ How to Run

### 1️⃣ Build the transaction graph
```bash
python src/build_graph.py

### 2️⃣ Train GNN Model
```bash
python src/train.py

### 3️⃣ Run inference on unseen transactions
```bash
python src/inference.py

### 4️⃣ Generate evaluation plots
```bash
python src/plots.py

## 📌 Key Takeaways

- Graph Neural Networks (GNNs) are highly effective for detecting fraudulent transactions in relational financial data.
- Modeling transactions as a graph enables the model to capture complex relationships that are missed by traditional tabular approaches.
- Proper handling of **partially labeled data** using label masking is crucial for real-world fraud datasets.
- Class imbalance can be effectively addressed using weighted loss functions.
- The trained GNN achieved strong performance with a high ROC-AUC and recall, making it suitable for fraud prevention systems.

---

## 📜 Conclusion

This project demonstrates the successful application of **Graph Neural Networks** for financial fraud detection using real-world transaction data.  
By leveraging transaction relationships through graph modeling and employing a Graph Convolutional Network, the model learns meaningful patterns that significantly improve fraud detection performance.

The end-to-end pipeline—from data preprocessing and graph construction to training, evaluation, and inference—follows **industry and research best practices** and provides a reproducible framework for deploying graph-based fraud detection systems.

---

## 👤 Author

**Vinaya Sri Mandalapu**  
Machine Learning / AI Enthusiast  

---

## 📄 License

This project is intended for **educational and research purposes**.
