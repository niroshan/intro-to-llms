# Session 9: Quiz

## 1. Transformer Components
Name the two main components inside each Transformer block and briefly describe what each does.

## 2. Positional Encoding
Why do Transformers need positional encoding? What would happen without it?

## 3. Causal Masking in Transformers
How is causal masking implemented in our code? Find the relevant line and explain what it does.

## 4. Residual Connections
In the code, point to where residual connections are implemented. Why are they written as `x = x + layer(x)` rather than just `x = layer(x)`?

## 5. Multi-Head Attention
If we have d_model=64 and n_heads=4, what is the dimension of each head? Why might using multiple heads be better than one big attention operation?

## 6. Code Modification Exercise
Modify the `MiniTransformer` to print the attention weights from the first head of the first layer during generation. (Hint: you'll need to modify `CausalSelfAttention` to optionally return attention weights)

## 7. Scaling Discussion
Our mini transformer has ~12K parameters. GPT-3 has 175 billion. What aspects of the architecture would you scale up to reach that size?
