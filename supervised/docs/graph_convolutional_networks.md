## Graph Convolutional Networks

GCNs specifically design the aggregation and combine steps to mathematically mimic traditional convolution on graphs, making the process more analogous to what happens in CNNs.

$$
H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2}\ \tilde{A}\ \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$

**Where:**

- $$H^{(l)}$$: Node features/input at layer $$l$$
- $$\tilde{A} = A + I$$: Adjacency matrix with self-loops (shows node connections, including with itself)
- $$\tilde{D}$$: Degree matrix of $$\tilde{A}$$ (diagonal, counts neighbors including self)
- $$W^{(l)}$$: Weight (parameter) matrix (learned)
- $$\sigma$$: Non-linear activation function (e.g., ReLU)

### GCN is a special case of ChebNet:

By setting the Chebyshev polynomial order $$K = 1$$, only the first-order term is kept in the Chebyshev expansion. This leads to a much simpler formula in which each layer only mixes a node’s own feature with those of its immediate neighbors.

