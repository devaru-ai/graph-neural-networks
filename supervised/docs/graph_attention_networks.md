## Graph Attention Networks

A Graph Attention Network uses attention to help each node figure out for itself which neighbors are most important.

**Attention lets the network learn which neighbors matter most on its own**, during training. In effect, the model learns to “pay more attention” to neighbors that help it make better predictions for each specific node.

## Algorithm

**1. Linear Transformation:** Each node’s features are multiplied by a weight matrix to get new features.

**2. Compute Attention Scores:** For every neighbor j of node i, calculate a number called the "attention score" to see how important neighbor j is to node i.

**3. Normalize Attention Scores:** Convert the raw attention scores into probabilities using softmax so that the importance weights for all neighbors of a node sum to 1.

**4. Aggregate Neighbor Information:** For each node, multiply each neighbor's features by its attention weight, sum them up, and then apply an activation function (like ReLU).

**5. Multi-Head Attention:** GATs often does the above process several times in parallel (“multiple heads”), each with different parameters. The results from each head are either concatenated or averaged to produce the final node features.

## Mathematical Formulas

1. **Linear Transformation**  
Each node’s feature vector is linearly transformed:

$$h_i' = W h_i$$

- where $$W$$ is a learnable weight matrix.

2. **Attention Coefficient Calculation**  
For each edge (i, j), compute the attention score:

$$e_{ij} = \text{LeakyReLU}\left( \mathbf{a}^\top [h_i' \| h_j'] \right)$$
- $$\mathbf{a}$$ is a learnable weight vector.
- $$[h_i'\|h_j']$$ is the concatenation of the transformed features of nodes i and j.

3. **Softmax Normalization**  
Normalize $$e_{ij}$$ across all neighbors $$j$$ of node $$i$$:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i)} \exp(e_{ik})}$$
- $$\alpha_{ij}$$ determines how much attention node $$i$$ pays to neighbor $$j$$.

4. **Message Aggregation and Update**  
Aggregate the features of neighbors, weighted by the attention coefficients:

$$h_i^{\text{new}} = \sigma \left( \sum_{j \in N(i)} \alpha_{ij} h_j' \right)$$

- where $$\sigma$$ is an activation function (often ReLU).

5. **Multi-Head Attention (Optional)**  
If using T attention heads, concatenate the results from all heads or, in the last layer, average the outputs:

$$h_i^{\text{new}} = \sigma \left( \frac{1}{T} \sum_{t=1}^T \sum_{j \in N(i)} \alpha_{ij}^{(t)} W^{(t)} h_j \right)$$

