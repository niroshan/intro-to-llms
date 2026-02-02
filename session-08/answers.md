# Session 8: Quiz Answers

## 1. QKV Understanding
In the Query-Key-Value framework, explain:
a) What does the Query represent?
b) What does the dot product between Q and K compute?
c) Why do we apply softmax to the scores?

**Answer:**
a) The Query represents "what information am I looking for?" – it's the current position asking a question about what it needs from other positions.

b) The dot product Q·K computes a similarity score between the query and each key. Higher dot products indicate the key "matches" or is relevant to what the query is looking for. Mathematically, it measures how aligned the two vectors are.

c) Softmax converts the raw scores into a probability distribution that sums to 1. This ensures we get a weighted average of values (not just a sum), and it makes the weights interpretable as "how much attention to pay" to each position.

## 2. Scaling Factor
Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention? What would happen if we didn't?

**Answer:** 
When the dimension $d_k$ is large, dot products can become very large in magnitude. This pushes the softmax function into regions where it has extremely small gradients (near 0 or 1 outputs), making learning difficult.

For example, if $d_k = 512$ and vectors have elements around 1, the dot product could be around 512. After softmax, one position might get weight 0.9999 and others nearly 0, with almost no gradient flowing to change this.

Dividing by $\sqrt{d_k}$ keeps the variance of the scores roughly constant regardless of dimension, ensuring softmax operates in a region with healthy gradients.

## 3. Causal Masking
Why do we need causal masking for language models? What would happen if we didn't use it during training?

**Answer:**
Causal masking prevents each position from "seeing" future tokens. This is essential because:

1. **Training consistency**: During training, we have the full sequence available, but we're training the model to predict the next token. If position 3 could attend to position 4, it would learn to "cheat" by just copying the answer.

2. **Generation reality**: During generation, future tokens don't exist yet. If we trained without masking, the model would depend on information that won't be available at generation time.

Without causal masking, the model would achieve unrealistically low training loss but fail completely at generating text because it never learned to predict without future context.

## 4. Attention vs RNN
List two advantages of attention over RNNs for processing sequences.

**Answer:**
1. **Parallelization**: Attention can compute relationships between all pairs of positions simultaneously (in parallel), while RNNs must process positions sequentially. This makes attention much faster to train on modern GPUs.

2. **Direct long-range connections**: In attention, any two positions can interact directly in a single step. In RNNs, information must flow through all intermediate hidden states, where it can be lost or diluted (vanishing gradient). Attention's direct connections make it easier to learn long-range dependencies.

## 5. Interpreting Attention Weights
Given the sentence "The dog chased the cat", if the word "chased" has high attention weights on both "dog" and "cat", what might this indicate about what the model has learned?

**Answer:**
This suggests the model has learned that "chased" is semantically connected to both its subject ("dog" – who is doing the chasing) and its object ("cat" – what is being chased). The verb is attending to its arguments.

This is evidence that attention can learn grammatical and semantic relationships without being explicitly programmed with linguistic rules. The model discovered through training that verbs need to "look at" their subjects and objects to make good predictions.

This interpretability is one of attention's strengths – we can inspect the weights to understand (somewhat) what the model is doing.

## 6. Code Exercise
Modify the `simple_self_attention` function to also return the raw scores (before softmax). Then print both the scores and the weights for a 3-position sequence. Observe how softmax "sharpens" the distribution.

**Answer:**
```python
def simple_self_attention_with_scores(x):
    scores = torch.matmul(x, x.transpose(-2, -1))
    d_k = x.size(-1)
    scaled_scores = scores / (d_k ** 0.5)
    weights = F.softmax(scaled_scores, dim=-1)
    output = torch.matmul(weights, x)
    return output, weights, scaled_scores  # Return scores too

# Test
x = torch.randn(1, 3, 4)
output, weights, scores = simple_self_attention_with_scores(x)
print("Raw scaled scores:\n", scores.squeeze())
print("\nAfter softmax (weights):\n", weights.squeeze())
# Notice how softmax pushes values toward 0 or 1
# Higher scores become larger weights, lower scores become smaller
```
