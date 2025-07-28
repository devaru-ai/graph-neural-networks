# Graph Convolutional Networks (GCN)

## Overview

A **Graph Convolutional Network (GCN)** is a special type of Graph Neural Network (GNN) designed to efficiently combine (or "mix") node features with neighborhood information using well-founded mathematical principles. GCNs are widely used for node classification, link prediction, and more.

## Notation

- **$A$**: Adjacency matrix of the graph  
- **$I$**: Identity matrix  
- **$\tilde{A}=A+I$**: Adjacency with self-loops  
- **$\tilde{D}$**: Degree matrix
- **$X$**: Feature matrix ($N \times F$): each row is a node's features  
- **$H^{(k)}$**: Node features at layer $k$ ($H^{(0)}=X$)  
- **$W^{(k)}$**: Trainable weights at layer $k$  
- **$\sigma$**: Activation function (e.g., ReLU)

## GCN Layer: Main Equation

```math
H^{(k+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)} \right)
```
- **Aggregate:** Each node receives normalized features from its neighbors (and itself, because of self-loops).
- **Transform:** A shared trainable weight matrix is applied.
- **Activate:** Nonlinearity (typically ReLU) increases expressiveness and stability.

## Node-wise Update Rule

```math
H^k_i = \sigma \left(
    \sum_{j \in \mathcal{N}(i) \cup \{i\}}
    \frac{ \tilde{A}_{ij} }{ \sqrt{ \tilde{D}_{ii} \tilde{D}_{jj} } }
    H^{k-1}_j W^{k-1}
\right)
```
- **Explanation:** A node mixes its own features and those of all neighbors, with each contribution normalized by node degrees to ensure balanced influence.

## Spectral Graph Convolution

GCN can be understood as a graph spectral convolution:
```math
g_\theta * x = U\,g_\theta(\Lambda)\,U^T x
```
- $U$: Laplacian eigenvectors  
- $\Lambda$: Laplacian eigenvalues  
- $g_\theta$: Filter parameters in Fourier space

**Chebyshev polynomial expansion:**
```math
g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})
```
- $\theta_k$: Trainable Chebyshev coefficients (the parameters of the model)
- $T_k(x)$: Chebyshev polynomial of order $k$
- $\tilde{\Lambda}$: Scaled Laplacian eigenvalues
  
GCN simplifies this expansion (by taking $K=1$), leading to the simple, symmetric normalization above.

**Chebyshev polynomials** are defined recursively:
- $T_0(x) = 1$
- $T_1(x) = x$
- $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$

The scaling:

$$
\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{\text{max}}} - I
$$

where $\lambda_{\text{max}}$ is the largest eigenvalue of the Laplacian $L$.

### Chebyshev-based Graph Convolution

The resulting spectral convolution operation for a graph signal $x$:

$$
g_{\theta'} * x = \sum_{k=0}^K \theta_k T_k(\tilde{L}) x
$$

where:

- $\tilde{L} = \frac{2L}{\lambda_{\text{max}}} - I$ is the scaled Laplacian matrix
- $T_k(\tilde{L}) x$ is the $k$-th order Chebyshev polynomial of the Laplacian applied to $x$


## Graph Convolutions: From Chebyshev Expansion to Standard GCN Layer

**1. Stacking Simple Graph Convolution Layers**

- Instead of using high-order polynomial filters (many $K$) as in Chebyshev expansion, GCNs make each layer a simple linear function on the normalized graph Laplacian.
- By **stacking multiple such layers** (each followed by a nonlinearity), the network still learns complex functions, while each layer only mixes information from immediate neighbors (1-hop).

**2. Key Approximation Steps**

- **Start with Chebyshev convolution:**
  
$$g_{\theta'} * x \approx \sum_{k=0}^K \theta_k T_k(\tilde{L}) x$$

- **Set $K=1$ and approximate $\lambda_{\max} \approx 2$**:

  $$g_{\theta'} * x \approx \theta'_0 x + \theta'_1 (L - I) x = \theta'_0 x - \theta'_1 D^{-1/2} A D^{-1/2} x \qquad$$

- **Parameter tying (reduce overfitting):**
  - Set $\theta = \theta'_0 = -\theta'_1$ so that:

    $$g_\theta * x \approx \theta (I + D^{-1/2} A D^{-1/2}) x \qquad$$
    
- **Stability concern:** The matrix $I + D^{-1/2} A D^{-1/2}$ has eigenvalues and can cause exploding/vanishing gradients with many layers.

**3. Renormalization Trick for Stable Deep GCNs**

- To fix the stability issues, **add self-loops** ($A \rightarrow \tilde{A} = A + I$) and use corresponding degree matrix $\tilde{D}$.
- Final, standard GCN layer:

$$H = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W \qquad$$
  
  - $X$: input features (possibly multi-dimensional)
  - $W$: trainable weights (can produce many output channels or hidden units)
  - $H$: new node features


## Key Differences from Basic GNN

| Aspect           | Basic GNN                          | GCN (this model)                                             |
|------------------|------------------------------------|--------------------------------------------------------------|
| Self-loops       | Optional                           | **Always adds self-loop ($A+I$)**                      |
| Aggregation      | Mean/sum (un-normalized, any type) | **Normalized by degree: $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$** |
| Weight matrices  | Sometimes only at final layer      | **Per-layer learnable, applies at every propagation step**    |
| Activation       | Optional                           | **Mandatory, usually ReLU**                                  |
| Theory           | None or informal                   | **Rooted in spectral graph theory**                          |

## Summary

GCN is a special case of message-passing GNNs that:
- **Combines neighbor and self-features** (via self-loops)
- **Normalizes contributions** to prevent node degree bias
- **Transforms features** with learnable weights at each layer
- **Is stable** (prevents feature explosion) and justifiable by deep spectral network connections

## References

- Kipf & Welling, 2017: _Semi-Supervised Classification with Graph Convolutional Networks_.
- Wu, L., Cui, P., Pei, J., & Zhao, L. (Eds.). *Graph Neural Networks: Foundations, Frontiers, and Applications*.

