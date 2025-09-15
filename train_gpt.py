import torch, platform
import torch.nn as nn
from torch.nn import functional as F
import math
import os

#-----------
# hyperparameters
batch_sz = 16#64#16
block_sz = 128#64 #32
max_iter = 100000#10000
eval_interval = 1000
l_rate = 3e-4
eval_iters = 200
n_embd = 200 #384 # 192 # 96
n_head = 5
n_layer = 10
dropout = 0.4#0.2
bias = False
out_dir = 'out_trash'

# -----------
print("Torch:", torch.version) 
print("Arch:", platform.machine()) 
print("MPS built:", torch.backends.mps.is_built()) 
print("MPS available:", torch.backends.mps.is_available()) 
device = "mps" if torch.backends.mps.is_available() else "cpu" 
print("Device:", device)
# device = "cpu"
# x = torch.randn(1000, 1000, device=device) 
# y = x @ x.T 
# print("Device:", device, "Mean:", y.mean().item())

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("config:", config)

os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(12223)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('tiny_edgar_allan_poe.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"length of dataset in characters: {len(text):,}")

# text = text.lower()##################################
chars = sorted(list(set(text)))
vocabulary_size = len(chars)

print(vocabulary_size)
print(''.join(chars))

char2int = {ch:i for i,ch in enumerate(chars)}
int2char = {i:ch for i,ch in enumerate(chars)}
encoder = lambda x: [char2int[c] for c in x]
decoder = lambda y: ''.join([int2char[i] for i in y])

data = torch.tensor(encoder(text), dtype=torch.long) #int64
# print((data.shape, data.dtype, data.min().item(), data.max().item()))
# print(data[:10])
#train and val split
n = int(0.8*len(data))
tra_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # print(batch_sz, block_sz)
    data = tra_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_sz, (batch_sz,))
    # print(ix)
    x = torch.stack([data[i:i+block_sz] for i in ix])
    y = torch.stack([data[i+1:i+block_sz+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb = get_batch(split)
            logits,loss = model(xb,yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_sz, block_sz))
                                        .view(1, 1, block_sz, block_sz))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    
class FeedF(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        # head_sz = n_embd // n_head
        self.sa_head = CausalSelfAttention(n_embd, n_head)
        self.feedf = FeedF(n_embd)
        
        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.ln2 = LayerNorm(n_embd, bias=bias)
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.feedf(self.ln2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embed_table = nn.Embedding(vocabulary_size, n_embd)
        self.pos_embed_table = nn.Embedding(block_sz,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lnf = LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, intx, targets=None):
        B, T = intx.shape
        #intx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embed_table(intx) # (B,T,C))
        pos_embd = self.pos_embed_table(torch.arange(T, device=device)) # (T,C)
        x = tok_embd + pos_embd # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.lnf(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is not None:
            B,T,C = logits.shape
            logitso = logits.view(B*T,C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logitso,targets)
        else:
            loss = None
        return logits,loss

    def generate(self, intx, max_new_tokens):
        #intx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #due to pos enc, we crop intx to the last block_sz tokens
            intx_cond = intx[:,-block_sz:]
            logits,loss = self(intx_cond)
            logits = logits[:,-1,:] #becomes (B, C)
            probs = F.softmax(logits,dim=-1) # (B,C)
            next_token = torch.multinomial(probs,num_samples=1) # (B,1)
            intx = torch.cat((intx,next_token),dim=1) # (B,T+1)
        return intx


model = GPT().to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=l_rate)
best_val_loss = 1e9

for iter in range(1,max_iter+1):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype = torch.long, device = device)#.to(device)
out = model.generate(context,500)[0].tolist()
print(decoder(out)) # [0] to get single batch element


