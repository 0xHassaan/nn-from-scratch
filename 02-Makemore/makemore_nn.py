import torch
torch.set_printoptions(sci_mode=False)
xs, ys = [], []

words = open("../names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s, i in stoi.items()}

n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        n += 1
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)


import torch.nn.functional as F

g = torch.Generator().manual_seed(12345)

xenc = F.one_hot(xs, num_classes=27).float()
W = torch.randn((27,27), generator=g, requires_grad=True).float()

for i in range(200):
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(n), ys].log().mean()

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad

    print(i, loss.item())


for i in range(10):
    ix = 0
    out = []

    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))