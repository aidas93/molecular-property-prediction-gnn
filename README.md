# 🧪 Molecular Property Prediction with Graph Neural Networks (GNN)

## Overview
Graph Neural Network implemented with **PyTorch Geometric** for molecular property prediction on the **QM9 dataset**.  
Molecules are represented as graphs (atoms = nodes, bonds = edges), and the model predicts quantum chemistry properties.  
In this implementation we focus on **property index 0 (U0 energy)**.

---

## Model
- 2 × GCNConv layers (ReLU)  
- Global mean pooling  
- Dense(64 → 128, ReLU) → Dense(128 → 1)  

---

## Results
- Training: 10k molecules (5 epochs)  
- Testing: 2k molecules  
- Metric: MSE Loss on predicted U0 energy  

Performance improves with larger training splits and more epochs.

---

## Future Work
- Train on the full QM9 dataset (~134k molecules)  
- Multi-target prediction (all 19 properties)  
- Explore advanced GNNs (GIN, Graph Attention Networks)  
- Hyperparameter tuning and scaling
