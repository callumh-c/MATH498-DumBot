import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
import urllib.request

class config:
    d_vocab:int = 10000
    d_model:int = 128
    d_mlp:int = 1024
    n_layers:int = 2
    n_epoch:int = 20
    n_context:int = 512
    act_fn:type[nn.Module] = nn.ReLU

class Tokenization:
    def __init__(self, data:str):
        # all letters lowercase and all characters other than 
        # letters, numbers, and punctuation removed
        data = data.lower()
        data = re.sub(r"[^a-z0-9\s.,!?;:'\"()-]", "", data)

        # break text into tokens by whitespace
        self.tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)*|[.,!?;:\"()-]", data)

        # builds vocab as alphabetized list of unique words
        unique_tokens = sorted(set(self.tokens))
        # map each word to integer token
        self.token_to_id = {tok: i for i, tok in enumerate(unique_tokens)}
        # reverse previous map for decoding
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        # vocab size
        self.d_vocab = len(self.token_to_id)

    # assigns each word to integer token
    def encode(self, data: str) -> list[int]:
        data = data.lower()
        tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)*|[.,!?;:\"()-]", data)
        return [self.token_to_id[tok] for tok in tokens if tok in self.token_to_id]

    # change each integer token to corresponding word
    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.id_to_token[i] for i in token_ids if i in self.id_to_token]
        string = ""
        for token in tokens:
            if token in ".,!?;:\"()-":
                string += token
            else:
                string += " " + token
        return string.strip()

class Transformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg = cfg

        # token embedding, assigns each word in vocab a vector of size d_model
        self.tok_embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        # token unembedding, converts vector to probabilities for each word in vocab
        self.tok_unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        
        # positional embedding, assigns each position in a given sequence a vector of size d_model
        self.pos_embed = nn.Embedding(cfg.n_context, cfg.d_model)
        
        # create weight matrices for attention head, ModuleList allows pytorch to use these values
        self.W_QK = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=False) for i in range(cfg.n_layers)])
        self.W_OV = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=False) for i in range(cfg.n_layers)])

        # create weight matrices for MLP in the same manner, except each vector is of length d_mlp instead
        self.W_in  = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_mlp)  for i in range(cfg.n_layers)])
        self.W_out = nn.ModuleList([nn.Linear(cfg.d_mlp, cfg.d_model) for i in range(cfg.n_layers)])

        # create mask here so it does not need to be calculated each time forward is run
        self.M = torch.full((cfg.n_context, cfg.n_context), float('-inf')).triu(diagonal=1)
    
    def forward(self, tokens): 
        # apply token embedding and positional embedding to input
        tok_out = self.tok_embed(tokens)
        pos_out = self.pos_embed(torch.arange(tokens.shape[1]))

        # combine these to get input for transformer block
        X = tok_out + pos_out
        
        # run number of transformer blocks specified in n_layers
        for i in range(self.cfg.n_layers):
            # Attention
            # implementing attention head formula provided in class
            A_x = F.softmax(X@self.W_QK[i].weight@X.transpose(-2, -1) + self.M[:tokens.shape[1],:tokens.shape[1]], dim=-1)@X@self.W_OV[i].weight
            X = X + A_x

            # MLP
            # implementing MLP with previously defined weight matrices
            MLP = self.W_in[i](X)
            MLP = self.cfg.act_fn()(MLP)
            MLP = self.W_out[i](MLP)
            X = X + MLP
        return self.tok_unembed(X)

    def generate(self, tokens, max_length:int=50):
        for i in range(max_length):
            # run forward operator
            output = self.forward(tokens)

            # convert forward model output to a probability distribution
            prob_dist = F.softmax(output[:, -1, :], dim=-1)

            # sample from this distribution
            generated_token = torch.multinomial(prob_dist, num_samples=1)

            # add new token to previous tokens, make sure shape will still be compatible with mask
            tokens = torch.cat([tokens, generated_token], dim=1)
            tokens = tokens[:, -self.cfg.n_context:]
        return tokens

# importing text
urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/2701/pg2701.txt", "pg2701.txt")

with open("pg2701.txt", "r", encoding="utf-8") as f:
    data = f.read()

cfg = config()
t = Tokenization(data)
cfg.d_vocab = t.d_vocab
model = Transformer(cfg)

while True:
    print('\033[31m' + 'You:' + '\033[0m')
    prompt = str(input())
    if prompt.lower() == 'exit':
        break
    try:
        input_tensor = torch.tensor([t.encode(prompt)])
        output_tensor = model.generate(input_tensor, max_length=50)
        
        print('\033[32m' + 'DumBot:' + '\033[0m')
        print(t.decode(output_tensor[0][input_tensor.shape[1]:].tolist()))
    except:
        print('\033[32m' + 'DumBot:' + '\033[0m')
        print('Sorry, you have not used any words I am familiar with.')
