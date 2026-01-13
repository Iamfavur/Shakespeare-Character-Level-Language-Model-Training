# this file is the initial step-by-step guide that was used to build the initial process before arranging and transferring all we need in the code to main.py file 
# download the tiny shakespeare dataset
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#TODO: TRAIN ANOTHER SHAKESPEARE MODEL FROM SCRTCH WITHOUT TRIANGULATING THE UPPER RIGHT PART OF THE ATTENTION MATRIX TO ALLOW FUTURE TOKENS COMMUNICATE WITH PAST TOKENS AND MEASURE/COMPARE THE PERFORMANCE - line 176
# to achieve this, remove this line of code "wei.masked_fill(tril == 0, float('-inf'))" - vid time 1:14:50

with open('input.txt','r',encoding='utf-8') as f:
    text  = f.read()

# print ("length of dataset in characters: ", len(text))
# print(text[:500])
    
chars = sorted(list(set(text)))
vocab_size= len(chars)
# print(''.join(chars))
# print(vocab_size)


# create a mapping from characters to integers 
stoi={ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in  enumerate(chars)}
encode = lambda s:[stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for  i in  l]) #decoder: take a list of integers , output a string 

# print(encode("hii there"))
# print(decode(encode("hii there")))


# let's now encode the entire text dataset and store it into a torch.Tensor
import torch  #we use PyTorch https://pytorch.org 
# print(torch.__version__)
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype) #this print the size and the datatype
# print(data[:1000]) #the 1000 characters we looked at earlier will look to GPT like this


# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]



torch.manual_seed(1337) #Here we are setting the  seed in the random number generator so have the same set of numbers in  the tutorial.
batch_size = 4 #This is how many independent sequences will we process in every forward bacward pass in the transformer(parallel) ?
block_size = 8 # This is the maximum context length for to make predictions

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data #if the split is training s[lit then we will look at train data otherwise val data
    ix = torch.randint(len(data) - block_size, (batch_size,))#Here we are generating random positions to get the chunk out of.... we are  generating batchsize number of 4, ix will be four numbers that are randomly generated btw 0 and (len(data) - block_size), ix will be 4 random numbers cause our batchsize is 4.
    x = torch.stack([data[i:i+block_size] for i in ix]) #These are the blocksize characters starting from i., We use the torch.stack to stack up the tensors as rows in a 4x8 tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #These are the offset of x(which means its +1)
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        # print(f"when input is {context.tolist()} the target: {target}")

# print(xb) # our input to the transformer




#Now we feed our batch of inputs (xb) into the transformers, we are usong the bigram Language model. 

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)  #Here we are setting the  seed in the random number generator so have the same set of numbers in  the tutorial.

class BigramLanguageModel(nn.Module): #constructing the model

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) #calling it, passing in the input(xb) and the target(yb)
# print(logits.shape)
# print(loss)

# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


#Optimizer object = it takes the gradients and update the parameters using the gradients
batch_size = 32 #
for steps in range(10): # increase number of steps for good results...  #? video has 100 steps
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print(loss.item())

#? THE MATHEMATICAL TRICK IN SELF-ATTENTION (FOR TRANSFORMER MODELS)
# consider the following toy examples

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C) #this will create 4(B) random matrices with 8rows (T) and 2columns (C) #torch.Size([4, 8, 2])
# print(x.shape)

a = torch.ones(3,3) #this will create a 3x3 matrix of 1s - ones
a = torch.tril(a) #this will triangulate the upper part of the matrix - make them 0s - because future tokens aren't allowed to communicate #?here the result of matrix C will be the sum of each number proceeding it in matrix B
a = a / torch.sum(a, 1 , keepdim=True) #this will make each row sums up to 1 #?here the result of matrix C columns will be the average/mean of each number proceeding it in matrix B
b = torch.randint(0,10,(3,2)).float() #this will create a 3x2 matrix of random numbers between 0 abd 10 with dtype float
c = a @ b #this will multiply the two 3x3 and 3x2 matrices to get a 3x2 matrix
# print("a=")
# print(a)
# print("b=")
# print(b)
# print("c=")
# print(c)


#? now making reference to line 57
tril = torch.tril(torch.ones(T, T)) #this is a 8x8 matrix of 1s where the upper-right region are triangu;ated/converted to 0s
wei = torch.zeros((T,T)) #this is a 8x8matrix of zeros
wei = wei.masked_fill(tril == 0, float('-inf')) #for elements in matrix tril whose values are 0s, replace their corresponding positions with '-inf' in matrix wei #* to ensure future tokens cant communicate with the past/present tokens
wei = F.softmax(wei, dim=-1) #this will expontentiate every single rows to get numbers that add up to 1. using dim=1 instead of dim=-1 will make the columns numbers add up to 1 instead of the rows
xbow3 = wei @ x
# print(wei)
# print(torch.allclose(x,xbow3)) #boolean used to compare if two tensor matrices are the same


#?THE PROBLEM THAT SELF-ATTENTION SOLVES
# DIfferent tokens will find other tokens more/less interesting depending on the data,
# Example a vowel looking for consonant from the past, it might want to know what those consonant are, it will want the information to flow to it from the past
# Future tokens will gather information from the past but in a data dependent way


#? HOW IT SOLVES IT
# Every single token (matrix values), will emit two vectors - query vector - key vector
# Query vector - what am i looking for
# key vector - what do i contain

#* The way we get affinities from tokens in a sequence(row) is by doing dot-product (multiplication) between the keys and queries
#EXAMPLE, My quey dot-product (multiplication) with all the keys of all the other tokens in the sequence - the dot-product becomes wei
# If the key and the query are self-aligned, will intarct to very high amount, and i will learn more about that specific token as opposed to any other token in the sequence

# ? IMPLMENTATION - for a single head of self-attention
torch.manual_seed(1337)
B,T,C = 4,8,32 
x = torch.randn(B,T,C) 
# print(c)

# lets see a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, 16) - 16 is the head_size
q = query(x) # (B, T, 16) - 16 is the head_size
v = value(x)

k = k.transpose(-2, -1) # transposed/swiped the last two dimensions of K so we can multiply it to get (B, T, T)
wei = q @ k #(B, T, 16) @ (B, 16, T) ----> (B, T, T) 
wei = wei * head_size ** -0.5

tril = torch.tril(torch.ones(T, T)) 
wei = wei.masked_fill(tril == 0, float('-inf')) 
wei = F.softmax(wei, dim=-1) 
out = wei @ v

print(out.shape)
print(out[0])

# THE TERMINAL RESULT
# torch.Size([4, 8, 32])
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
#         [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
#         [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
#         [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]],
#        grad_fn=<SelectBackward0>)

# The 8x8th token might say its looking for a consonant with a position upto to 4, then all the nodes/tokens in the sequence(values in the 8th rows) will emit keys, then maybe one of the channels (token in the row) might say i am a consonant with position upto 4, the token will have a higher number(key) in the sequence(row) which will make the 8th token pay more attention to it.
# The 8th token might also find other tokens interesting too, but not as much as the consonant token. The 8th token will then gather information from all the tokens in the sequence but more from the consonant token using Softmax.

#? NOTES from vid 1:10:50
#* Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
#* There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
#* Each example across batch dimension is of course processed completely independently and never "talk" to each other
#* In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
#* "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
#* "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much.   divides wei by 1/sqrt(head_size) is coded as "wei = wei * head_size ** -0.5"

