
## Variational Graph Autoencoder (VGAE)

VGAE infers the latent variables of nodes in the graph and decodes the edges.

### Step 1: Input Data
- **Node feature matrix** $X$: Size $N \times C$ (where $N$ is the number of nodes, $C$ is the number of features per node).
- **Adjacency matrix** $A$: Size $N \times N$ (with self-loops added, so diagonals are $1$).

### Step 2: Build the Encoder
Use a **Graph Convolutional Network (GCN)** to process $X$ and $A$:

Outputs two matrices:
- **Mean vectors** ($\mu$): Shape $N \times F$ (where $F$ is the latent dimension).
- **Standard deviation vectors** ($\sigma$): Shape $N \times F$.

Each node’s latent variable is sampled from a Gaussian:
- $z_i = \mu_i + \sigma_i \cdot \epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$

### Step 3: Build the Decoder
For each pair of nodes $(i, j)$, **predict the likelihood of an edge**:
- Compute the dot product: $z_i^\top z_j$
- Pass through sigmoid: $p(A_{ij}\mid z_i, z_j) = \sigma(z_i^\top z_j)$
This gives a **probability of an edge** between $i$ and $j$.


### Step 4: Set Prior
Use a **standard Gaussian prior** for latent vectors:
- For each node: $p(z_i) = \mathcal{N}(0, I)$ (mean $0$, identity covariance).

### Step 5: Compute Loss (ELBO)
Optimize the **Evidence Lower Bound**:
L_VGAE = E_{q(Z|X,A)} [ log p(A|Z) ] - KL(q(Z|X,A) || p(Z))
- First term: How well your $Z$ can reconstruct the graph.
- Second term: Regularizes $Z$ to stay similar to a standard Gaussian.


### Step 6: Train the Model
Use **gradient descent** (like Adam optimizer).
Each epoch:
- Forward pass: Encode $X$, $A$; sample $Z$; decode to get edge probabilities.
- Calculate ELBO loss.
- Backward pass: Update encoder’s weights.

### Step 7: Use Embeddings
After training:
- Use node embeddings ($Z$) for tasks like **link prediction**, **node classification**, etc.
