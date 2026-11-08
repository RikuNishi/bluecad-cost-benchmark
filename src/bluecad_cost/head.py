"""Shared frozen-feature logistic head; torch is loaded only on use."""

def pair_features(d, r):
    import torch
    import torch.nn.functional as F
    d, r = F.normalize(d.float(), dim=-1), F.normalize(r.float(), dim=-1)
    return torch.cat((d, r, (d-r).abs(), d*r), dim=-1)


def train(d, r, labels, seed=0, epochs=200, lr=1e-3, weight_decay=1e-4):
    import torch
    torch.manual_seed(seed)
    x = pair_features(d, r)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    head = torch.nn.Linear(x.shape[1], 1).to(x.device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    y = labels.float().reshape(-1, 1).to(x.device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
        loss.backward()
        opt.step()
    return head, mean, scale


def predict(head, mean, scale, d, r):
    import torch
    with torch.no_grad():
        return torch.sigmoid(head((pair_features(d, r) - mean) / scale)).flatten()


def best_threshold(scores, labels):
    """Maximize validation accuracy with >=; deterministic ties favor 0.5."""
    pairs = sorted(zip(map(float, scores), map(int, labels)), reverse=True)
    if not pairs:
        raise ValueError('Empty validation set')
    candidates = [(sum(y == (s >= .5) for s, y in pairs), .5)]
    correct = sum(y == 0 for _, y in pairs)
    candidates.append((correct, 1.0000001))
    i = 0
    while i < len(pairs):
        value = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == value:
            correct += 1 if pairs[i][1] else -1
            i += 1
        candidates.append((correct, value))
    return max(candidates, key=lambda t: (t[0], -abs(t[1] - .5), -t[1]))[1]
