import torch
import torch.nn as nn

# MAMBA Moddel : 
class MambaScan(nn.Module):
  
    def __init__(self, d_model):
      
        super().__init__()
      
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)
      
        self.gate = nn.Linear(d_model, d_model)
        self.lambda_param = nn.Parameter(torch.randn(d_model))

    def forward(self, u):
      
        delta = torch.sigmoid(u)
        A = torch.exp(-delta * self.lambda_param)

        b = self.B(u)
        g = torch.sigmoid(self.gate(u))

        A = g * A
        b = (1 - g) * b

        # Parallel scan : 
        P = torch.cumprod(A + 1e-6, dim=1)
        S = torch.cumsum(b / (P + 1e-6), dim=1)

        x = P * S
        return self.C(x)

# Hybrid Model : 
class HybridModel(nn.Module) :
    def __init__(self, vocab_size, d_model = 128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        self.attn = nn.MultiheadAttention(d_model, 4, batch_first = True)
        self.mamba = MambaScan(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)

        # Attention : 
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out

        # Mamba : 
        x = x + self.mamba(self.ln2(x))

        # FFNN : 
        x = x + self.ff(self.ln3(x))

        return self.head(x)
