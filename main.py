import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import BigramLanguageModel
from data_processing import *
from helper_functions import *
import os

## USER INPUTS ##
data_path = 'data'

train_test_split = 0.8
block_size = 8
batch_size = 32

max_iters = 10000
learning_rate = 1e-2
eval_iters = 250
#################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

corpus = ''
for file in os.listdir(data_path):
    path = data_path + '/' + file
    corpus = corpus + load_file(path)
encode, decode, vocab_size = tokenize(corpus)

data = torch.tensor(encode(corpus), dtype=torch.long)

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

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))