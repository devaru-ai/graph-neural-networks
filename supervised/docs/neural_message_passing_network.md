## Neural Message Passing Network

A Neural Message Passing Network (MPNN) is a general framework for building graph neural networks. It was originally developed for applications like drug and molecule analysis, but it’s powerful enough for any graph task, including node classification.

MPNNs break each layer’s operation into two main steps for each node:

## 1. Message Function

Each node **receives a “message” from each of its neighbors**.

This message is computed using a small neural network or function ($$M$$), and it can use:
- The current features of the node itself,
- The features of the neighbor,
- And possibly information about the edge connecting them (like the bond type in a molecule).

$$m_i^k = \sum_{j \in N(i)} M_k(h_i^{k-1}, h_j^{k-1}, e_{ij})$$

- $$m_i^k$$: The aggregated messages received by node $$i$$ at layer $$k$$
- The message from neighbor $$j$$ depends not just on the nodes, but also the edge ($$e_{ij}$$) between them.

## 2. Update Function

Next, **the node updates its feature by mixing its own previous state with the gathered aggregate message**.

This is done using another small neural network or function ($$U$$):

$$h_i^k = U_k(h_i^{k-1}, m_i^k)$$

**GCNs, GATs are just special cases of MPNN with specific choices for $$M$$ and $$U$$.**




