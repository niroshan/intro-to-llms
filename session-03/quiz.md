# Session 3: Quiz

## 1. Vocabulary and Embeddings
In our neural model, we used an embedding size of 16 for a small vocabulary. What might be a reasonable embedding size for a larger vocabulary (say 10,000 words)? And what is the intuition for choosing embedding dimensions?

## 2. Understanding Training
Why do we set `optimizer.zero_grad()` before computing the outputs each epoch in the code? What would happen if we omitted that?

## 3. Prediction Check
According to our training data, after the word "cat", the next word was "sat" once and "ate" once. In an ideal scenario, what probability should a well-trained neural model assign to "sat" vs "ate" after "cat"? If your trained model instead shows a strong preference for one over the other (like 70/30), what could be a reason?

## 4. Generalization Scenario
Suppose we train a neural network on a large corpus and it has seen the sentence "I went to Paris". It has also seen many sentences like "I went to [other cities]" and "I traveled to Paris". However, it never saw "I traveled to Paris" exactly. Would a neural language model likely assign a non-zero probability to "I traveled to Paris" as a whole (specifically "traveled" after "I")? Why or why not?

## 5. Looking Ahead
We discussed RNNs coming next for handling longer sequences. Can you think of a limitation of the fixed-window neural network (like our bigram NN) when it comes to context length? How does an RNN address it at a high level?
