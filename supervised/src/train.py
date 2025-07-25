import torch

def train(gnn, W, features, labels, adj, mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam([W], lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    H0 = features.clone()
    for epoch in range(epochs):
        gnn.train()
        # Forward pass
        H1 = gnn(H0, adj)
        logits = H1 @ W.t()
        loss = criterion(logits[mask], labels[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
    return W
