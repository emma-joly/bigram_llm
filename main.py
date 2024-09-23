import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import BigramLanguageModel
from helper_functions import get_batch, estimate_loss
import time

## USER INPUTS ##
data_path = 'frankenstein.txt'

train_test_split = 0.8
block_size = 8
batch_size = 4

max_iters = 1000
learning_rate = 3e-4
eval_iters = 250
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

## READ DATA ##
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

strToInt = {c:i for i,c, in enumerate(chars)}
intToStr = {i:c for i,c in enumerate(chars)}

encode = lambda s: [strToInt[c] for c in s]
decode = lambda l: ''.join([intToStr[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)

n = int(train_test_split*len(data))
train_data = data[:n]
test_data = data[n:]

x, y = get_batch(train_data, block_size, batch_size)

model = BigramLanguageModel(vocab_size)
m = model.to(device)

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss(model, train_data, test_data, block_size, batch_size, eval_iters)
        print(f"step: {iter}, train loss: {losses['train']:.3f}, test loss: {losses['test']:.3f}")

    xb, yb = get_batch(train_data, block_size, batch_size)
    logits, loss = model.forward(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()