# Session 9: Quiz Answers

## 1. Transformer Components
Name the two main components inside each Transformer block and briefly describe what each does.

**Answer:**
1. **Multi-Head Self-Attention**: Allows each position to gather information from all other positions (respecting causal masking). Multiple heads let the model attend to different types of patterns simultaneously.

2. **Feed-Forward Network (FFN)**: A position-wise neural network (same network applied to each position independently) that processes the attended information. Think of it as giving each position time to "think" about the information it gathered.

Both are wrapped with residual connections and layer normalization.

## 2. Positional Encoding
Why do Transformers need positional encoding? What would happen without it?

**Answer:**
Attention is **permutation-invariant** – it treats the input as a set, not a sequence. The attention computation between positions doesn't depend on their actual positions in the sequence.

Without positional encoding:
- "The cat sat on the mat" would be processed identically to "mat the on sat cat The"
- The model couldn't learn word order or grammatical structure
- Language modeling would fail because predicting the next word requires knowing word order

Positional encoding adds position information directly to the embeddings so the model can distinguish positions.

## 3. Causal Masking in Transformers
How is causal masking implemented in our code? Find the relevant line and explain what it does.

**Answer:**
```python
scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
```

This line:
1. Takes `self.mask`, which is a lower-triangular matrix of ones (1s below and on diagonal, 0s above)
2. For positions where mask is 0 (future positions), fills the attention score with `-inf`
3. After softmax, `-inf` becomes 0, so future positions have zero attention weight

The mask is created as:
```python
mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
```

## 4. Residual Connections
In the code, point to where residual connections are implemented. Why are they written as `x = x + layer(x)` rather than just `x = layer(x)`?

**Answer:**
In `TransformerBlock.forward()`:
```python
x = x + self.attn(self.ln1(x))  # Residual around attention
x = x + self.ffn(self.ln2(x))   # Residual around FFN
```

Writing `x = x + layer(x)` instead of `x = layer(x)`:
1. **Gradient flow**: During backpropagation, the gradient can flow directly through the addition (∂(x+f(x))/∂x = 1 + ...), providing a "gradient highway"
2. **Learning identity**: If a layer should do nothing, it just needs to output zeros
3. **Stable training**: Prevents values from exploding/vanishing through many layers
4. **Enables depth**: Makes it possible to train networks with 100+ layers

## 5. Multi-Head Attention
If we have d_model=64 and n_heads=4, what is the dimension of each head? Why might using multiple heads be better than one big attention operation?

**Answer:**
Each head has dimension: `head_dim = d_model // n_heads = 64 // 4 = 16`

Multiple heads are better because:
1. **Diverse attention patterns**: Each head can learn to attend to different types of relationships (e.g., subject-verb, adjective-noun, positional patterns)
2. **Parallel processing**: All heads compute simultaneously
3. **Richer representation**: The concatenated output captures multiple perspectives
4. **Regularization**: Having separate subspaces prevents the model from putting all information in one pattern

It's like having multiple experts, each looking at the text from a different angle, then combining their insights.

## 6. Code Modification Exercise
Modify the `MiniTransformer` to print the attention weights from the first head of the first layer during generation. (Hint: you'll need to modify `CausalSelfAttention` to optionally return attention weights)

**Answer:**
```python
class CausalSelfAttention(nn.Module):
    # ... (same __init__)
    
    def forward(self, x, return_attn=False):
        # ... (same attention computation until attn_weights)
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        if return_attn:
            return self.out_proj(out), attn_weights
        return self.out_proj(out)

# Then in TransformerBlock, pass return_attn parameter
# And in MiniTransformer.generate(), access and print the weights
```

## 7. Scaling Discussion
Our mini transformer has ~12K parameters. GPT-3 has 175 billion. What aspects of the architecture would you scale up to reach that size?

**Answer:**
To scale from ~12K to 175B parameters:

1. **d_model**: Increase from 32 to 12,288 (GPT-3)
2. **n_layers**: Increase from 2 to 96 (GPT-3)
3. **n_heads**: Increase from 4 to 96 (GPT-3)
4. **d_ff**: Increases with d_model (typically 4× d_model)
5. **vocab_size**: Use larger vocabulary (GPT-3 uses ~50K BPE tokens)
6. **max_seq_len**: Increase context window (2048 for GPT-3)

The beauty of Transformers is that this scaling is straightforward – just increase these numbers and train on more data. The architecture itself remains the same!
