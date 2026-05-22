import torch

N = torch.zeros(27, 27)

words = open("../names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))

stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s, i in stoi.items()}


# prepare dataset
for w in words:
    chs = ['.'] + list(w) + ['.']
    for char1, char2 in zip(chs, chs[1:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]

        N[ix1, ix2] += 1




# basic sampling
P = (N+1).float()
P /= P.sum(1, keepdim=True)
g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        
        out.append(itos[ix])
        
        if ix == 0:
            break

    print(''.join(out))



#likelihood

loglikelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for char1, char2 in zip(chs, chs[1:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]

        prob = P[ix1, ix2]
        logprob = torch.log(prob)

        loglikelihood += logprob
        n += 1

nll = -loglikelihood
avg_nll = nll/n

print(avg_nll)