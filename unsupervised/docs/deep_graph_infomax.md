## Deep Graph Infomax 

DGI learns graph representations via the principle of mutual information maximization between node-level local features and global summary features of the graph.

It aims to make each node's feature representation $$h_i$$ contain as much information as possible about the global context $$s$$ of the graph.



### 1. **Main Components**

- **Graph Encoder ($$\epsilon$$):** Any GNN (e.g., GCN or GraphSAGE) that outputs node embeddings $$H = \epsilon(X, A)$$.
- **Readout Function ($$R$$):** Aggregates all node embeddings into a single global graph summary $$s = R(H)$$. Commonly an average or pooling operation.
- **Discriminator ($$D$$):** A binary classifier that checks if a pair $$(h, s)$$ is from the true graph (positive) or a corrupted version (negative).


### 2. **Objective Function**

Instead of explicitly computing mutual information, DGI uses a **noise-contrastive estimation objective (binary cross-entropy)**, following Deep Infomax. The intuition is borrowed from GANs, where you train a discriminator to tell apart real and fake samples.

Equation:

$$
L = \frac{1}{N+M} \left[ \sum_{i=1}^{N} \log D(h_i, s) + \sum_{j=1}^{M} \log (1 - D(\tilde{h}_j, s)) \right]
$$

- $$D(h_i, s)$$: probability the (node, graph) pair is "real".
- $$D(\tilde{h}_j, s)$$: probability that the (corrupted node, real graph summary) pair is "fake".

**Maximizing this objective:** The encoder and readout learn to create representations where nodes and graph summary are strongly related for real pairs, and weakly related for fake pairs.


### 3. **How to Create Positive & Negative Samples**

**Positive Samples:**  
- Directly use $$(h_i, s)$$ pairs where $$h_i = \epsilon(X, A)_i$$ and $$s = R(H)$$.

**Negative Samples:**  
- Apply a corruption function $$C$$ to your graph's node features (e.g., row-wise shuffle of $$X$$), keep adjacency $$A$$ the same.
- Then get $$\tilde{H} = \epsilon(\tilde{X}, A)$$; form pairs $$(\tilde{h}_j, s)$$.

  - Other ways to corrupt: Subgraph sampling, Dropout on features, etc.


### 4. **Training Loop Summary**

1. **Sample negatives** using corruption: $$(\tilde{X}, \tilde{A}) = C(X, A)$$
2. **Encode node representations:**
   - **Positive:** $$H$$ from $$(X, A)$$
   - **Negative:** $$\tilde{H}$$ from $$(\tilde{X}, \tilde{A})$$
3. **Calculate global summary:** $$s = R(H)$$
4. **Form pairs:** Each $$h_i$$ with $$s$$ (positive), each $$\tilde{h}_j$$ with $$s$$ (negative)
5. **Train:** Update parameters of encoder ($$\epsilon$$), discriminator ($$D$$), and optionally readout ($$R$$), maximizing the binary classification objective.


