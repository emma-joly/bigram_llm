import torch

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, test_data, block_size, batch_size, eval_iters):
    out = {}
    model.eval()
    labels = ['train', 'test']

    for i,data in enumerate([train_data,test_data]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[labels[i]] = losses.mean()
    model.train()
    return out