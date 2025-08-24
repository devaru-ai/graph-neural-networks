import torch

def compute_accuracy(model, features, labels, adj, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        preds = logits[mask].argmax(dim=1)
        acc = (preds == labels[mask]).float().mean().item()
    return acc

def train_basic(gnn, W, features, labels, adj, train_mask, val_mask, test_mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam([W], lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    H0 = features.clone()
    train_accs, val_accs, test_accs = [], [], []
    for epoch in range(epochs):
        gnn.train()
        H1 = gnn(H0, adj)
        logits = H1 @ W.t()
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = compute_accuracy(gnn, features, labels, adj, train_mask)
        val_acc = compute_accuracy(gnn, features, labels, adj, val_mask)
        test_acc = compute_accuracy(gnn, features, labels, adj, test_mask)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.4f} Val={val_acc:.4f} Test={test_acc:.4f} Loss={loss.item():.4f}")
    return train_accs, val_accs, test_accs

def train_gcn(model, features, labels, adj, train_mask, val_mask, test_mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_accs, val_accs, test_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        logits = model(features, adj)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = compute_accuracy(model, features, labels, adj, train_mask)
        val_acc = compute_accuracy(model, features, labels, adj, val_mask)
        test_acc = compute_accuracy(model, features, labels, adj, test_mask)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.4f} Val={val_acc:.4f} Test={test_acc:.4f} Loss={loss.item():.4f}")
    return train_accs, val_accs, test_accs

def train_gat(model, features, labels, adj, train_mask, val_mask, test_mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_accs, val_accs, test_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        logits = model(features, adj)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = compute_accuracy(model, features, labels, adj, train_mask)
        val_acc = compute_accuracy(model, features, labels, adj, val_mask)
        test_acc = compute_accuracy(model, features, labels, adj, test_mask)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.4f} Val={val_acc:.4f} Test={test_acc:.4f} Loss={loss.item():.4f}")
    return train_accs, val_accs, test_accs

def train_mpnn(model, features, labels, adj, train_mask, val_mask, test_mask, epochs=100, lr=0.01, edge_attr=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_accs, val_accs, test_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        logits = model(features, adj, edge_attr) if edge_attr is not None else model(features, adj)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = compute_accuracy(model, features, labels, adj, train_mask)
        val_acc = compute_accuracy(model, features, labels, adj, val_mask)
        test_acc = compute_accuracy(model, features, labels, adj, test_mask)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_acc:.4f} Val={val_acc:.4f} Test={test_acc:.4f} Loss={loss.item():.4f}")
    return train_accs, val_accs, test_accs
