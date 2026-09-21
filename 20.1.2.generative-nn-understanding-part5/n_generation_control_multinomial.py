import torch

iter = 1000
token_probs = torch.tensor([0.1, 0.1, 0.3, 0.5])
print(token_probs)
for _ in range(iter):
    max_Y = torch.argmax(token_probs, 0)
    print(max_Y.item())

cnt = [0, 0, 0, 0]
for _ in range(iter):
    sampled_Y = torch.multinomial(token_probs, 1)
    print(sampled_Y[0].item())
    cnt[sampled_Y[0]] += 1

print(cnt)
