import torch
import torch.nn as nn
import torch.optim as optim

import time
import math

import matplotlib.pyplot as plt

import pandas as pd

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from model import HybridModel
from transformer import TransformerModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenization :
def train_bpe(texts, vocab_size = 10000):
  
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens = ["<pad>", "<unk>"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

# Data : 
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]

tokenizer = train_bpe(train_texts)
vocab_size = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

tokens = []

for t in train_texts:
    tokens.extend(encode(t))

tokens = torch.tensor(tokens)

def get_batch(seq_len, batch_size = 8) :
  
    idx = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
  
    x = torch.stack([tokens[i:i+seq_len] for i in idx])
    y = torch.stack([tokens[i+1:i+seq_len+1] for i in idx])
  
    return x.to(device), y.to(device)

# Model Training : 
def train_model(model, seq_len, steps = 200):
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr = 3e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for step in range(steps):
        x, y = get_batch(seq_len)

        logits = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


# Benchmark Model : 
def benchmark(model, seq_len, batch_size = 4):
  
    model = model.to(device)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(2):
        _ = model(x)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        _ = model(x)

    torch.cuda.synchronize()
    end = time.time()

    latency = end - start
    vram = torch.cuda.max_memory_allocated() / (1024**2)
    throughput = (batch_size * seq_len) / latency

    return latency, vram, throughput


# Experiment : 
seq_len = 512

trans_model = TransformerModel(vocab_size)
trans_losses = train_model(trans_model, seq_len)

hybrid_model = HybridModel(vocab_size)
hybrid_losses = train_model(hybrid_model, seq_len)

# Perplexity : 
trans_ppl = [math.exp(l) for l in trans_losses]
hybrid_ppl = [math.exp(l) for l in hybrid_losses]

# Comparing across diff seq_len : 
seq_lengths = [64,128,256,512,1024,2048]
results = []

for l in seq_lengths :
    print(f"Benchmarking L={l}")

    t_lat, t_mem, t_tp = benchmark(trans_model, l)
    h_lat, h_mem, h_tp = benchmark(hybrid_model, l)

    results.append([l, "Transformer", t_lat, t_mem, t_tp])
    results.append([l, "Hybrid", h_lat, h_mem, h_tp])

df = pd.DataFrame(results, columns = [
    "SeqLen", "Model", "Latency", "VRAM_MB", "Throughput"
])

print(df)

# Visualization of Results : 
plt.figure(figsize = (8, 8))

# Perplexity : 
plt.subplot(1,5,1)

plt.plot(trans_ppl, label = "Transformer")
plt.plot(hybrid_ppl, label = "Hybrid")

plt.yscale("log")
plt.title("Perplexity : ")

plt.legend()

# Latency : 
plt.subplot(1,5,2)

for m in ["Transformer", "Hybrid"] :
    sub = df[df["Model"] == m]
    plt.plot(sub["SeqLen"], sub["Latency"], label = m)

plt.xscale("log")
plt.yscale("log")

plt.title("Latency Scaling : ")

plt.legend()

# Throughput : 
plt.subplot(1,5,3)

for m in ["Transformer", "Hybrid"]:
    sub = df[df["Model"] == m]
    plt.plot(sub["SeqLen"], sub["Throughput"], label = m)

plt.xscale("log")
plt.yscale("log")

plt.title("Throughput : ")
plt.legend()

plt.show()
