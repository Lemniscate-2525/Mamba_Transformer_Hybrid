import torch
import torch.nn as nn

class TransformerModel(nn.Module) :
  
    def __init__(self, vocab_size, d_model = 128): 
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        self.attn = nn.MultiheadAttention(d_model, 4, batch_first = True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x) :
        x = self.embed(x)

        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out

        x = x + self.ff(x)
        x = self.ln(x)

        return self.head(x)
