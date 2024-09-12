import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# hyperparameters
### TODO: Check how thse variables are accessed by the class without passing 
# as they are not defined as global variable

batch_size = 32 # how many independent sequence will we process in parallel
block_size = 8 # what is the `maximum` context length for prediction?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embd = 32 # num of embeding dimensions
# to enable debugging on cuda
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Using :{device}")
# --------------

torch.manual_seed(1337)

# read it in to inspect it
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# print("Length of dataset in charachters", len(text))

# here are al unique charachters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars)) 
# print('vocab_size', vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)

# Let's now split up the data into train and validation setss
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data

  ix = torch.randint(len(data) - block_size, (batch_size,))
  # print('ix', ix)
  x = torch.stack([data[i : i + block_size] for i in ix])
  y = torch.stack([data[i + 1: i + block_size + 1 ] for i in ix])

  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple Bigram model
class BigramLanguageModel(nn.Module):
  # subclass of nn.Module
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from
    # a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # because we added `n_embd` token_embedding_table(idx) will now return token_emb
    # and we now need linear layer to go from token_emb to logits
    self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head: langurage modeling head

  def forward(self, idx, targets=None):
    # idx and targets are both (B, T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B, T, C)
    logits = self.lm_head(tok_emb) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      # here we will strech out the array so that it's 2 dim
      # (4x8 , 65 )
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indicies in the current context
    # the job of this generate function is to take (B, T) and generate (B, T+1)
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(idx) # >>> this will call the forward function
      # focus only on the last time step. So instead of B, T, C
      # we pluck out the -1 : the last element in the time dimension.
      # Because those are the prediction of what comes next
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probablites
      probs = F.softmax(logits, dim=-1) # (B, C)

      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # >>> (B, 1) because in each og the batch dimension we have a single
      # predition of what comes next
      # append sampled index to the running sequence
      # in the below line whatever is predicted it just concatenated with
      # previous char at dim 1
      idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
    return idx


model = BigramLanguageModel()
m = model.to(device)

# create  a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


# training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generating from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) 
generated_output = m.generate(context, max_new_tokens=300)
print("-"* 8)
print(decode(generated_output[0].tolist()))
print("-"* 8)