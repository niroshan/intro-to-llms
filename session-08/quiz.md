# Session 8: Quiz

## 1. QKV Understanding
In the Query-Key-Value framework, explain:
a) What does the Query represent?
b) What does the dot product between Q and K compute?
c) Why do we apply softmax to the scores?

## 2. Scaling Factor
Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention? What would happen if we didn't?

## 3. Causal Masking
Why do we need causal masking for language models? What would happen if we didn't use it during training?

## 4. Attention vs RNN
List two advantages of attention over RNNs for processing sequences.

## 5. Interpreting Attention Weights
Given the sentence "The dog chased the cat", if the word "chased" has high attention weights on both "dog" and "cat", what might this indicate about what the model has learned?

## 6. Code Exercise
Modify the `simple_self_attention` function to also return the raw scores (before softmax). Then print both the scores and the weights for a 3-position sequence. Observe how softmax "sharpens" the distribution.
