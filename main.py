import torch  # we use PyTorch https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F

# hypaparameters - use the above settings for a quick run
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for prediction?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3 
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # run on a GPU if you have it
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2
# --------------


# for actually training the model after putting all the components #? this requires more time and power
# batch_size = 64  # how many independent sequences will we process in parallel?
# block_size = 256  # what is the maximum context length for prediction?
# max_iters = 5000
# eval_interval = 300
# learning_rate = 3e-4 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # run on a GPU if you have it
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# --------------

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers , output a string
def decode(l): return ''.join([itos[i] for i in l])


# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # if the split is training s[lit then we will look at train data otherwise val data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # when we load the data, move it to device (GPU)
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # for efficient memory use, let pytorch know for everything that happens inside the function, we will not call .backward()
def estimate_loss():  # averages the loss over multiple batches
    out = {}
    model.eval()  # no much effect here but its good practice
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()  # no much effect here but its good practice
    return out


class Head(nn.Module): #the self-attetion allows the tokens to communicate and gather data 
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout) # a regularization technique to prevent overfitting the model

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     #(B,T,head_size)
        q = self.query(x)   #(B,T,head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) #(B, T, H) @ (B, H, T) ----> (B, T, T) 
        wei = wei * C**-0.5 #scale by square root of embedding dimension
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v #(B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)  # project from concat size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B,T,num_heads*head_size) - output of all heads concatenated
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module): #this allows them to think independently on the data they gathered from the SA (self-attention)
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of SA heads we'd like
        super().__init__()
        head_size = n_embd // n_head #32 // 4 = 8
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) #to normalize our features
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # "x+" is residual connections
        x = x + self.ffwd(self.ln2(x)) # "x+" is residual connections
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):  # constructing the model

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # n_embd = number of embedding dimensions
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each position from  0 to block_size minus one will have its own embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #sequential applicationg of Block - n_layer specifys how many layers of the blocks we will have 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        # lm_head = language modeling head - this is so we can go from token embeedings to logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # (B,T,C) - tok_emb = token_embeddings
        tok_emb = self.token_embedding_table(idx)
        # (T,C) - position embeddings which are basically integers from 0 to T-1 are embbeded through the table to create T by C
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
# when we create the model, move the parameters to device (GPU)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
