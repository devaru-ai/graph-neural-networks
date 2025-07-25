# General Framework: Supervised Graph Neural Networks (GNNs)

## Overview

Supervised graph neural networks (GNNs) are designed to learn vector representations (embeddings) for nodes in a graph by iteratively aggregating information from each node’s neighbors and itself. These embeddings are then used for downstream tasks, such as **node classification**.

## Step-by-Step Mathematical Framework

Suppose you have a graph with nodes. Each node starts with a feature vector. The general update is performed in layers:

### 1. **Initialization**

Let **X** be the feature matrix (rows = nodes, columns = features):

```
Node | Feature1 | Feature2 | ... | FeatureF
-----|----------|----------|-----|---------
  1  |   x11    |   x12    | ... |   x1F
  2  |   x21    |   x22    | ... |   x2F
  .  |   ...    |   ...    | ... |   ...
  N  |   xN1    |   xN2    | ... |   xNF
```

All node representations are set to their features:
- **H⁰ = X**

### 2. **Layer-wise Updates**

For each GNN layer `k = 1 ... K`, repeat:

a. **AGGREGATE**  
Each node gathers information from its neighbors:
- **cᵥᵏ = AGGREGATE( { Hᵘ⁽ᵏ⁻¹⁾ | u ∈ Neighbors(v) } )**

b. **COMBINE**  
Each node updates its representation with both its previous state and the aggregated info:
- **Hᵥᵏ = COMBINE( Hᵥ⁽ᵏ⁻¹⁾, cᵥᵏ )**

Different GNN variants (GCN, GAT, etc.) use different designs for AGGREGATE and COMBINE.

### 3. **Prediction**

After `K` layers, use the final embeddings for node classification. For node v:

- **ŷ_v = Softmax(W Hᵥᴷ )**

where:
- **W** = trainable weight matrix (does linear transformation)
- **Softmax** = function that turns output into class probabilities

### 4. **Supervised Training**

Train the network on labeled nodes by minimizing a loss (cross-entropy for classification):

- **O = (1 / nₗ) Σ[ loss(ŷᵢ, yᵢ) ]**

where:
- **nₗ** = number of labeled nodes
- **yᵢ** = true label of node i, **ŷᵢ** = predicted probability
- Training is done with backpropagation as in regular neural networks.

## Example Calculation

**Suppose** a node has a representation `[2.0, -1.0]` at the last GNN layer.

A 3-class linear classifier has weight matrix:

```
W = [[1.0, 0.5],
     [-0.5, 2.0],
     [0.2, -1.0]]
```

Compute: `WH`
- Row 1: (1.0)*2.0 + (0.5)*(-1.0) = 2.0 - 0.5 = 1.5
- Row 2: (-0.5)*2.0 + (2.0)*(-1.0) = -1.0 - 2.0 = -3.0
- Row 3: (0.2)*2.0 + (-1.0)*(-1.0) = 0.4 + 1.0 = 1.4

Apply **Softmax** to `[1.5, -3.0, 1.4]`:
- Exponentiate: [4.48, 0.05, 4.05]
- Sum: 8.58
- Probabilities: `[0.52, 0.01, 0.47]`

**Interpretation:**  
This node is mostly class 1 (52%) or class 3 (47%), very unlikely class 2.

## References

- [Xu et al., 2019] https://arxiv.org/abs/1810.00826
- [Kipf & Welling, 2017] https://arxiv.org/abs/1609.02907

**In summary:**  
GNNs update node embeddings by aggregating from neighbors, combine with their own state, and use these learned vectors for node prediction—training with standard neural network techniques.
