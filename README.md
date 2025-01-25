# Topology-Aware Activation Functions in Neural Networks

Topology-Aware Activation Functions in Neural Networks

## Overview

This study explores novel activation functions that enhance the ability of neural networks to manipulate data topology during training. Building on the limitations of traditional activation functions like $\mathrm{ReLU}$, we propose $\mathrm{SmoothSplit}$ and $\mathrm{ParametricSplit}$, which introduce topology "cutting" capabilities. 

These functions enable networks to transform complex data manifolds effectively, improving performance in scenarios with low-dimensional layers. Through experiments on synthetic and real-world datasets, we demonstrate that $\mathrm{ParametricSplit}$ outperforms traditional activations in low-dimensional settings while maintaining competitive performance in higher-dimensional ones. 

Our findings highlight the potential of topology-aware activation functions in advancing neural network architectures.

## Installation

```bash
# Clone the repository
git clone https://github.com/Snopoff/Topology-Aware-Activations.git

# Navigate to project directory
cd Topology-Aware-Activations
```

## Structure

```
Topology-Aware-Activations/
├── configs/
├── notebooks/
├── scripts/
├── src/
├── tex/
├── main.py
├── Makefile
└── README.md
```