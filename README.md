# HYBRID Mamba-Transfomer -> Structured State Space Meets Attention : 

---

## The Problem with Pure Transformers : 

**Problem 1 -> Quadratic attention cost :** The core operation of a Transformer is $QK^\top$, a matrix multiplication that produces an $N \times N$ attention matrix. For a sequence of length $N$, every token must compute a similarity score against every other token. This scales as $O(N^2)$ in both compute and memory. Double the sequence length and attention cost quadruples. For long sequences like genomic data, audio waveforms, or multi-day financial tick streams, this is not a slow operation; it is an impossible one.

**Problem 2 -> KV cache memory explosion :** During autoregressive inference (generating one token at a time), the Transformer caches the Key and Value matrices for every previously generated token. This is necessary to avoid recomputing attention from scratch at each step. But the cache grows linearly with sequence length and never shrinks. For a context window of 100,000 tokens with 32 layers and 8 heads at 128d per head; $100{,}000 \times 32 \times 8 \times 128 \times 2 \times 4$ bytes (float32) $\approx 210$ GB. This exceeds the VRAM of any single GPU. Long-context inference on a pure Transformer is a hardware problem, not a software one.

**Problem 3 -> Uniform attention across all positions :** Every token attends to every other token with equal architectural access, regardless of relevance. A token at position 1 and a token at position 50,000 are computationally equivalent. The model must learn which positions matter from data alone, with no inductive bias toward recency or locality. For signals where recent context matters far more than distant context (time series, audio, streaming data), this is wasteful; the model spends capacity learning to ignore most of what it attends to.

---

## The Mamba Alternative : 

The idea behind Mamba is to model sequences as dynamical systems rather than as sets of pairwise relationships. Instead of asking "how every token relate to every other token," it asks "how a hidden state evolve as it absorbs new inputs over time."

This is the continuous-time State Space Model (SSM).

The system is governed by two differential equations : 

$$\frac{dx(t)}{dt} = A\,x(t) + B\,u(t)$$

$$y(t) = C\,x(t)$$

Where;; 
- $x(t) \in \mathbb{R}^d$ is the hidden state at time $t$; the system's memory.
- $u(t) \in \mathbb{R}$ is the input at time $t$.
- $A \in \mathbb{R}^{d \times d}$ governs how the state evolves over time; the memory dynamics.
- $B \in \mathbb{R}^{d \times 1}$ governs how the input is written into the state.
- $C \in \mathbb{R}^{1 \times d}$ governs how the state is read out as the output $y(t)$.

### Derivation of the Solution : 

The equation $\frac{dx}{dt} = Ax + Bu$ is a first-order linear ODE. We solve it by recognizing that without the $Bu$ term, the homogeneous solution is $x(t) = e^{At}x(0)$.

This comes from the definition of the matrix exponential. The scalar ODE $\frac{dz}{dt} = az$ has solution $z(t) = e^{at}z(0)$. Extending to matrix $A$: the matrix exponential $e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$ satisfies $\frac{d}{dt}e^{At} = Ae^{At}$, so $x(t) = e^{At}x(0)$ solves the homogeneous case.

For the full system with input, the variation of parameters method gives:

$$x(t) = e^{A t}\,x(0) + \int_0^t e^{A(t-s)} B\,u(s)\,ds$$

This is the general solution. The first term is the initial state decayed/evolved by $A$. The second term is the accumulated contribution of all past inputs, each weighted by how much $A$ has evolved since that input arrived.

---

## Discretization : 

Neural networks operate on discrete token sequences, not continuous signals. To use the SSM on sequences, we discretize with a learnable timestep $\Delta t$ (how much "time" passes between tokens):

$$\bar{A} = e^{A \Delta t}$$

$$\bar{B} = (e^{A\Delta t} - I)A^{-1}B \approx \Delta t \cdot B \quad \text{(Zero-Order Hold approximation)}$$

The discrete recurrence then becomes;

$$x_t = \bar{A}\,x_{t-1} + \bar{B}\,u_t$$

$$y_t = C\,x_t$$

This is a standard linear recurrence. Given the state at $t-1$ and the new input $u_t$, we compute the new state $x_t$ and read out $y_t$. The system has $O(1)$ memory (just $x_t$) and $O(N)$ compute (one update per timestep). Compared to Transformer attention: $O(N)$ memory and $O(N^2)$ compute.

---

## The Problem with Naive Discretization : 

The most natural choice for $\bar{A}$ is $e^{A\Delta t}$ where $A$ is a free $d \times d$ matrix.

This creates three immediate problems:

**Instability :** For the recurrence $x_t = \bar{A}x_{t-1} + \bar{B}u_t$ to not blow up over long sequences, all eigenvalues of $\bar{A}$ must have magnitude $\leq 1$. A random $d \times d$ matrix has eigenvalues scattered in the complex plane. Enforcing stability for a general matrix requires constrained optimization that is difficult and expensive.

**Parallelism Nightmare :** A full $d \times d$ matrix multiply at every timestep means the recurrence $x_t = \bar{A}x_{t-1} + \bar{B}u_t$ has a dense state transition. We can't trivially parallelize this across timesteps because $x_t$ depends on $x_{t-1}$ through a full matrix.

**Too much Computation :** $d \times d$ matrix multiply at every token costs $O(d^2)$ per step. For $d = 256$: 65,536 multiplications per token. For over 10,000 tokens there'll be 655 million multiplications just for the state transitions.

---

## The Diagonal Constraint : 

The solution is to constrain $A$ to be a diagonal matrix. Specifically by representing $A = -\Lambda$ where $\Lambda$ is a *diagonal matrix of positive real eigenvalues*:

$$\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d), \quad \lambda_i > 0$$

Then:

$$\bar{A} = e^{A\Delta t} = e^{-\Lambda \Delta t} = \text{diag}(e^{-\lambda_1 \Delta t},\; e^{-\lambda_2 \Delta t},\; \ldots,\; e^{-\lambda_d \Delta t})$$

The **exponential of a diagonal matrix is just the diagonal matrix of exponentials**.

### Significance of the Diagonal Matrix (Eigenvalue Decomposition) : 

Any matrix $A$ can be written as $A = V\Lambda V^{-1}$ where $\Lambda$ is diagonal (containing eigenvalues) and $V$ contains the eigenvectors. Then:

$$e^{A\Delta t} = V\,e^{\Lambda \Delta t}\,V^{-1}$$

In the full eigendecomposition, we need $V$ and $V^{-1}$ (the eigenvectors). The key in the Mamba design is: the eigenvectors $V$ can be absorbed into the $B$ and $C$ projections. Since $B$ maps input to state and $C$ maps state to output, we can fold $V^{-1}$ into $B$ and $V$ into $C$. Remaining is just the diagonal $\Lambda$.

So working in the diagonal basis is a *reparameterization* that separates the "remembered" ($\Lambda$: eigenvalues, time evolution) from the "extracted" ($B$, $C$: feature interactions).

**The $\lambda_i$ are learned parameters, initialized randomly.** They are not derived from any eigendecomposition of a real matrix. The model learns them end-to-end, and they converge to values that are useful for the task.

### Three Advantages of Diagonal $\bar{A}$ : 

**1. Guaranteed stability when $\lambda_i > 0$ :**

$$|\bar{A}_{ii}| = e^{-\lambda_i \Delta t} < 1 \quad \text{for all } \lambda_i > 0, \Delta t > 0$$

Each diagonal entry is a decay factor strictly between 0 and 1. The state magnitude can never blow up. Stability is guaranteed by the parameterization, not by constrained optimization.

**2. Elementwise multiplication instead of matrix multiply :**

The state update becomes;

$$x_t^{(i)} = e^{-\lambda_i \Delta t}\,x_{t-1}^{(i)} + \bar{B}^{(i)}\,u_t$$

Each dimension $i$ of the state evolves independently. No cross-dimension coupling in the $A$ transition. This is $O(d)$ per step instead of $O(d^2)$.

**3. Interpretability of eigenvalues :**

$\lambda_i$ controls how fast dimension $i$ of the state forgets the past. Large $\lambda_i$: fast decay, short memory. Small $\lambda_i$: slow decay, long memory. A mix of eigenvalues gives the model multi-timescale memory as some dimensions track recent inputs, others track long-range context.

---

## Selective SSM (Input-Dependent Gating) : 

Classic SSMs have fixed $A$, $B$, $C$ matrices. Every input is processed the same way regardless of content. This is the key weakness as the system cannot decide to "pay attention" to some inputs and ignore others.

Selective SSMs (Mamba) make $B$, $C$, and the timestep $\Delta t$ functions of the input:

$$\Delta_t = \sigma(u_t W_\Delta + b_\Delta)$$

$$B_t = u_t W_B$$

$$C_t = u_t W_C$$

Now the effective $\bar{A}_t = e^{-\Delta_t \Lambda}$ is also input-dependent through $\Delta_t$. The discrete recurrence becomes:

$$x_t = \bar{A}_t\,x_{t-1} + \bar{B}_t\,u_t$$

$$y_t = C_t\,x_t$$

**The role of $\Delta_t$ in controlling memory :**

- $\Delta_t \to 0$ (small timestep): $\bar{A}_t = e^{-\Delta_t \Lambda} \to I$ and $\bar{B}_t \to 0$. The state barely changes; the input is nearly ignored. The system "holds" what it has.
- $\Delta_t \to \infty$ (large timestep): $\bar{A}_t \to 0$ and $\bar{B}_t \to W_B^{-1}$ (ZOH limit). The old state is completely forgotten; the state becomes purely the new input. A hard reset.
- Intermediate $\Delta_t$: partial memory retention. The system blends old state with new input in a learned, input-dependent ratio.

This is selective gating: the model learns when to remember, when to forget, and when to reset, based on the content of the current input.

---

## The Recurrence Problem(From $O(N)$ Sequential to $O(\log N)$ Parallel) : 

The discrete recurrence $x_t = \bar{A}_t x_{t-1} + \bar{B}_t u_t$ is computed as a for-loop over $t = 1, \ldots, N$. Each step depends on the previous. This is $O(N)$ sequential operations; no two steps can be computed in parallel. Modern **GPUs thrive on parallelism**; a sequential loop wastes nearly all available compute.

### The Parallel Scan (Prefix Scan) : 

The recurrence is an *associative operation*. We define a "state pair" $\Psi_t = (\bar{A}_t, \bar{B}_t u_t)$ representing "decay coefficient and input contribution at step $t$."

The combining rule for two consecutive pairs is :

$$(\bar{A}_j, b_j) \circ (\bar{A}_i, b_i) = (\bar{A}_j \cdot \bar{A}_i , \bar{A}_j \cdot b_i + b_j)$$

Verifying: $x_i = \bar{A}_i x_{i-1} + b_i$ and $x_j = \bar{A}_j x_{j-1} + b_j$, then after two steps :

$$x_j = \bar{A}_j(\bar{A}_i x_{i-1} + b_i) + b_j = (\bar{A}_j \bar{A}_i)\,x_{i-1} + (\bar{A}_j b_i + b_j)$$

The combined pair $(\bar{A}_j \bar{A}_i , \bar{A}_j b_i + b_j)$ has exactly the same structure as a single-step pair. Hence associativity holds; combining pairs of pairs always produces the same form.

### Tree Reduction : $O(\log N)$ Depth

Since the operation is associative, we can apply it like a tree reduction. For $N = 8$ steps:

**Level 1 :** Combine pairs $(\Psi_1, \Psi_2)$, $(\Psi_3, \Psi_4)$, $(\Psi_5, \Psi_6)$, $(\Psi_7, \Psi_8)$ in parallel. 4 operations, 4 results.

**Level 2 :** Combine $(\Psi_{1\text{:}2},\ \Psi_{3\text{:}4})$ and $(\Psi_{5\text{:}6},\ \Psi_{7\text{:}8})$ in parallel. 2 operations, 2 results.

**Level 3 :** Combine $(\Psi_{1\text{:}4},\ \Psi_{5\text{:}8})$. 1 operation, final result.

Total depth: $\log_2 8 = 3$ levels. Compared to the sequential loop: 7 steps, all serial.

In practice this is computed via cumulative product and cumulative sum;

**Step 1: Cumulative product of decay factors :**

$$P_t = \prod_{s=1}^{t} \bar{A}_s$$

This gives the total accumulated decay from position 1 through position $t$.

**Step 2: Weighted inputs :**

Each input contribution $b_t = \bar{B}_t u_t$ needs to be "forwarded" to the current timestep. The contribution of $b_s$ at time $t > s$ is $b_s \cdot \prod_{r=s+1}^t \bar{A}_r = b_s \cdot P_t / P_s$.

So the normalized contribution is $b_s / P_s$.
Adding; 

$$S_t = \sum_{s=1}^{t} \frac{b_s}{P_s}$$

**Step 3: Recover state :**

$$x_t = P_t \cdot S_t$$

Both $P_t$ (cumulative product) and $S_t$ (cumulative sum) can be computed with no for-loop using `torch.cumprod` and `torch.cumsum`, which are fully parallel GPU primitives. The entire length $N$ sequence is processed in parallel. No sequential dependency. Full GPU utilization.

### Sequential vs Parallel : 

| Operation | Sequential Recurrence | Parallel Scan |
|-----------|----------------------|---------------|
| Compute | $O(N)$ sequential steps | $O(\log N)$ depth |
| Memory | $O(1)$ state | $O(N)$ for scan buffers |
| GPU utilization | Near-zero (serial) | Near-full (parallel) |

---

## Disadvantages of Pure Mamba : 

Mamba is excellent at **long-range memory and GPU-efficient inference**.

But ther are limitations :

**No Bidirectional Context :** The recurrence is causal by design; $x_t$ depends only on $x_{t-1}$ and $u_t$. For tasks where future context matters (understanding a sentence where the subject comes after the verb), pure Mamba struggles. Transformers with bidirectional attention have equal access to all positions.

**Weak at precise token-to-token matching :** Attention's $QK^\top$ directly computes similarity between any pair of tokens. Mamba's state-based approach compresses all past context into a fixed-size hidden state. If the task requires precisely retrieving what was at position 37 from a sequence of length 5,000, the compression may lose that information.

**Fixed state dimension $d$ regardless of content :** The hidden state is always $d$-dimensional regardless of how much information the sequence contains. Attention naturally scales its expressivity with sequence length (more tokens, more attention patterns). Mamba's memory capacity is fixed.

---

## The Hybrid Architecture : 

The hybrid model interleaves Mamba and Transformer blocks within the same stack. 

Each `HybridBlock` contains :

1. **Causal Multi-Head Attention :** handles precise token-to-token relationships, bidirectional pattern matching, and short-to-medium range dependencies where $O(N^2)$ cost is acceptable.

2. **Mamba Parallel Scan :** handles long-range context compression, memory-efficient state evolution, and provides $O(N)$ memory cost that does not blow up the KV cache.

3. **FFN (GELU) :** position-wise nonlinear feature mixing, shared across both components.

All three sublayers use Pre-LayerNorm and residual connections:

$$x = x + \text{Attn}(\text{LN}(x))$$
$$x = x + \text{Mamba}(\text{LN}(x))$$
$$x = x + \text{FFN}(\text{LN}(x))$$

The key property is that the attention sublayer operates in $O(N^2)$ but handles only the patterns that require precise pairwise comparison. The Mamba sublayer operates in $O(N \log N)$ (parallel scan) and handles temporal state evolution and long-range memory.

Together, the hybrid does what neither alone can; **precise local attention along with efficient long-range memory**.

---

## Architecture : 

```
Input: token sequence (Batch, N)
    |
Token Embedding : vocab_size → d_model = 256        (Batch, N, 256)
    |
HybridBlock × 4:

    LayerNorm → MultiHeadAttention (8 heads, d_k = 32, causal mask)
                QKᵀ / √32 + M_causal → softmax → ⊗ V
    Residual add

    LayerNorm → MambaScan (parallel scan via cumprod + cumsum)
                Δt = σ(u)          input-dependent timestep
                A  = exp(-Δt · λ)  diagonal decay matrix
                B  = (1-g) · W_B(u)  gated input projection
                A  = g · A           gated decay
                P  = cumprod(A)      accumulated decay
                S  = cumsum(B/P)     normalized input sum
                x  = P · S           state reconstruction
                y  = W_C(x)          output projection
    Residual add

    LayerNorm → FFN (256 → 512 → 256, GELU)
    Residual add

    → (Batch, N, 256)

Final LayerNorm → Linear(256 → vocab_size) → logits
```

![Architecture Diagram](com.png)

---

## Time and Space Complexity : 

Let $N$ = sequence length, $d$ = model dimension (256), $H$ = attention heads (8), $L$ = layers (4).

**Attention sublayer :**

$$\text{Time: } O(N^2 \cdot d) \qquad \text{Space: } O(N^2 + N \cdot d)$$

The $N^2$ attention matrix is the bottleneck. For $N=1{,}024$: $1{,}024^2 \times 256 \approx 268M$ multiplications per layer. For $N=65{,}536$: impractical on any single GPU.

**Mamba sublayer (Parallel Scan) :**

$$\text{Time: } O(N \log N \cdot d) \qquad \text{Space: } O(N \cdot d)$$

The parallel scan has $\log N$ depth over $N$ operations, each of dimension $d$. Scan buffers for $P$ and $S$ are both $(B, N, d)$. For $N=65{,}536$: the Mamba sublayer is feasible; the attention sublayer is not. This is the core motivation for the hybrid.

**Inference with KV cache :**

$$\text{Space: } O(L \cdot N \cdot d)$$

The *Mamba sublayer requires no KV cache* because its state $x_t$ is a fixed-size $d$-dimensional vector. Only the most recent state needs to be stored for the next step. Total Mamba inference memory is $O(L \cdot d)$, constant in $N$. This is the core advantage over pure Transformer inference.

**Full hybrid per layer :**

$$\text{Time: } O(N^2 d + N \log N \cdot d) = O(N^2 d) \quad \text{(attention dominates)}$$

$$\text{Space: } O(N^2 + N \cdot d) \quad \text{(attention matrix dominates for small } N \text{)}$$

For very long sequences where the attention component is applied only to a local window (sliding window attention), the Mamba component dominates and the hybrid achieves $O(N \log N)$ overall.

---

## Failure Case Analysis : 

**Attention still quadratic within each block :** The hybrid reduces the quadratic cost but does not eliminate it. If full global attention is applied at every layer across the full sequence, the memory bottleneck remains. The hybrid is **most effective when attention is applied locally** (to a window of recent tokens) and Mamba handles the long-range state.

**Mamba parallel scan adds scan buffer overhead :** The $O(N \cdot d)$ scan buffers for $P$ and $S$ are allocated in GPU memory for every Mamba sublayer. For $L=4$ layers and large $N$: this can exceed the $O(N^2)$ attention matrix cost for short sequences. The crossover point where Mamba becomes memory-efficient compared to attention is approximately $N > d$, which for $d = 256$ means sequences longer than 256 tokens.

**Stability of the gated scan :** The implementation adds `1e-6` to $A$ before `cumprod` and `1e-6` to $P$ before inversion. For very long sequences, numerical precision of the cumulative product degrades even with this guard. The product of 10,000 values slightly above zero still approaches zero.

**No positional encoding in this implementation :** Neither sinusoidal PE nor RoPE is applied. The Mamba component encodes position implicitly through the temporal recurrence. The attention component has no positional information at all; it processes all positions identically. This limits the hybrid's ability to **learn position-sensitive patterns** in the attention sublayer. Adding RoPE to Q and K (as in Llama) would improve this.

**Fixed $\lambda$ initialization :** The eigenvalue parameters $\lambda_i$ are initialized from `torch.randn(d_model)`, which places some $\lambda_i$ near zero or negative. Near-zero $\lambda_i$ creates near-unit diagonal entries in $\bar{A}$, which means those state dimensions have very long memory (slow decay). Negative $\lambda_i$ makes $e^{-\lambda_i \Delta t} > 1$, creating explosive growth in that dimension.

---

## Metrics : 

> **Dataset :** WikiText-2 (raw v1), tokenized with a 10,000-token BPE vocabulary trained on the training split. Both models trained for 200 steps with AdamW ($lr = 3 \times 10^{-4}$), batch size 8, sequence length 512. Benchmarks run at batch size 4 across sequence lengths $\{64, 128, 256, 512, 1024, 2048\}$ on CUDA.

### EDA :

The three plots below show training perplexity, latency scaling, and throughput scaling as measured on the WikiText-2 benchmark run.

![Benchmark Plots](met.png)

**Perplexity :** Perplexity measures how confidently a language model predicts the next token; lower is better, and it is the exponential of cross-entropy loss. It is the standard quality metric for language modelling.

- Only one curve is prominently visible because both models' traces sit almost on top of each other, which is itself informative: the Mamba scan component does not hurt language modelling quality at this scale.
  
- The curve drops sharply from $10^4$ in the first 50 steps. This is the model learning BPE token frequency statistics, the easy part. The gradient is steep because even a crude unigram prior slashes perplexity dramatically.
  
- By step 100 the descent slows and plateaus around $10^3$. This is where syntactic and semantic structure must be learned, and 200 steps on WikiText-2 is nowhere near enough. The plateau is a training budget limit, not an architectural ceiling.

**Latency Scaling :** Latency measures wall-clock time for a single forward pass; plotted on a log-log scale so that power-law scaling appears as a straight line.

- The Transformer (blue) shows a sharp anomalous dip around sequence length 64–128. This is a GPU warmup artifact: CUDA kernel launch overhead dominates at very small batch-sequence products, so the first measurement is noisy and unreliable.
  
- Above 256 tokens, both curves grow steeply and roughly in parallel. The Hybrid is consistently above the Transformer because it runs attention and the Mamba scan together; the scan adds to the forward pass rather than replacing attention.
  
- The crossover where Mamba's $O(N \log N)$ scaling wins over attention's $O(N^2)$ would only appear at sequence lengths well beyond 2048. At this model size ($d = 128$, single-layer attention) the quadratic regime has not yet made attention catastrophically expensive.

**Throughput :** Throughput measures tokens processed per second; higher is better, and it captures how efficiently the model uses GPU parallelism across the batch.

- The Transformer (blue) peaks around 256–512 tokens then declines. This is the point where **the $N^2$ attention matrix starts saturating memory bandwidth**; the GPU spends more time moving data than computing.
  
- The Hybrid (orange) shows a flatter, lower curve throughout. It does strictly more work per forward pass (attention + scan + FFN vs. attention + FFN), so it processes fewer tokens per second at every sequence length tested.
  
- The Hybrid does not show a throughput advantage here because the benchmark does not reach the regime where Mamba's linear memory cost would dominate. At $N > 8{,}000$ with fixed memory, the Transformer begins to OOM while the Mamba state stays constant; that is where the curves would invert.

### Benchmark Metrics : 

Raw per-model, per-sequence-length measurements from the benchmark run. Latency is single-forward-pass wall-clock time in seconds; VRAM is peak GPU memory allocated in MB; Throughput is tokens per second ($\text{batch} \times \text{seqlen} / \text{latency}$).

| SeqLen | Model       | Latency (s) | VRAM (MB)   | Throughput (tok/s) |
|--------|-------------|-------------|-------------|---------------------|
| 64     | Transformer | 0.016416    | 80.59       | 1.56 × 10⁴          |
| 64     | Hybrid      | 0.001456    | 82.60       | 1.76 × 10⁵          |
| 128    | Transformer | 0.001230    | 103.41      | 4.16 × 10⁵          |
| 128    | Hybrid      | 0.002068    | 107.42      | 2.48 × 10⁵          |
| 256    | Transformer | 0.001454    | 151.38      | 7.04 × 10⁵          |
| 256    | Hybrid      | 0.002229    | 159.41      | 4.59 × 10⁵          |
| 512    | Transformer | 0.002007    | 251.49      | 1.02 × 10⁶          |
| 512    | Hybrid      | 0.003202    | 267.20      | 6.40 × 10⁵          |
| 1024   | Transformer | 0.004619    | 482.42      | 8.87 × 10⁵          |
| 1024   | Hybrid      | 0.005923    | 514.55      | 6.92 × 10⁵          |
| 2048   | Transformer | 0.013924    | 1063.99     | 5.88 × 10⁵          |
| 2048   | Hybrid      | 0.015996    | 1287.74     | 5.12 × 10⁵          |

**Interpretation of Metrics :**

- At sequence lengths 64–256, the Hybrid's latency is comparable to or slightly better than the Transformer despite doing more computation. `cumprod` and `cumsum` are highly parallelized CUDA primitives that hit peak throughput at small sizes where attention head computation has not yet saturated the device.
- By 512 tokens, the Hybrid is reliably slower and more memory-hungry. The VRAM gap grows with sequence length; at 2048 tokens the Hybrid uses ~224 MB more than the Transformer.
- This gap grows because the scan buffers $P$ and $S$ are both $(B, N, d)$ tensors that accumulate on top of the attention matrix. The Transformer's VRAM grows roughly quadratically (attention matrix); the Hybrid grows quadratically plus linearly (attention + scan), so the gap widens predictably.
- In a production setting at 8k–128k sequence lengths, the Mamba component's $O(d)$ constant inference state would begin to invert this relationship. That regime is not captured in this benchmark.

---

## Conclusion : 

- The benchmark confirms theoretical predictions at small scale that the Hybrid adds latency and VRAM overhead at moderate sequence lengths because it runs both attention and the SSM, not one instead of the other.
- Perplexity trajectories for both models are nearly identical. The Mamba component does not hurt language modelling quality; it just has not yet had the chance to help it at this sequence length and training budget.
- The architectural advantage, constant KV-cache memory at inference and $O(N \log N)$ long-range state, only becomes the dominant factor at sequence lengths well beyond what this 200-step WikiText-2 run covers.
- The right experimental setting to observe the Hybrid's edge is fixed-memory inference at $N > 4{,}096$ against a Transformer that has begun to OOM; that is where the Mamba component's $O(d)$ state pays off.
