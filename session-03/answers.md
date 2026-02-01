# Session 3: Quiz Answers

## 1. Vocabulary and Embeddings
In our neural model, we used an embedding size of 16 for a small vocabulary. What might be a reasonable embedding size for a larger vocabulary (say 10,000 words)? And what is the intuition for choosing embedding dimensions?

**Answer:** For 10,000 words, common embedding sizes might be 50, 100, 300, or even 512 dimensions – it really depends on the complexity of relationships we expect the model to capture and available data. Intuition: The embedding dimension is like how many features we allow the model to use to characterize a word. Too low (like 2 or 5) and it might not encode enough nuance (words will collide or not capture multiple facets). Too high (like 1000) and it might overfit or be inefficient if we don't have enough data to justify those parameters. Historically, popular pre-trained embeddings (like Word2Vec, GloVe) used 50-300 dimensions for large corpora. So something in the low hundreds is often reasonable for 10k vocabulary. In summary: maybe around 100 or 300 could be a good guess. More dimensions allow capturing more fine-grained distinctions but require more data to train.

## 2. Understanding Training
Why do we set `optimizer.zero_grad()` before computing the outputs each epoch in the code? What would happen if we omitted that?

**Answer:** `optimizer.zero_grad()` clears the gradients from the previous iteration. In PyTorch, by default, gradients accumulate on each parameter (like summing over multiple backward passes) unless reset. If we omitted `zero_grad()`, then each epoch's gradient would add on top of the previous epoch's gradient, which is not what we want – it would effectively apply outdated gradients and mess up the update. We want each weight update to be based solely on the current batch's error. So we zero them to start fresh each time (or each batch). Forgetting to zero would cause incorrect, compounded updates and likely divergence or weird training behavior.

## 3. Prediction Check
According to our training data, after the word "cat", the next word was "sat" once and "ate" once. In an ideal scenario, what probability should a well-trained neural model assign to "sat" vs "ate" after "cat"? If your trained model instead shows a strong preference for one over the other (like 70/30), what could be a reason?

**Answer:** Ideally, since "cat sat" and "cat ate" occurred equally often in the training, we'd expect P(sat|cat) ≈ P(ate|cat) ≈ 0.5 each (50/50). If the model shows 70/30 or some skew, possible reasons:
1. The model might not have fully converged or might have gotten stuck in a local optimum due to randomness or insufficient training (though 100 epochs for such tiny data is usually enough).
2. The network architecture or hyperparameters might introduce a bias. For example, maybe the initialization or the way the non-linearity works caused it to favor one output if not perfectly symmetric.
3. The training algorithm (SGD) might have overshot a bit or not perfectly balanced those probabilities, but as long as it's close, it's fine. Slight imbalances can happen because with limited precision and small data, it might not land exactly on 0.5.
4. Another possibility: if "sat" or "ate" has different overall frequency, but here each appeared once after "cat". If one had appeared in another context too, that could sway the embedding.

So, basically small training irregularity or local minima, but theoretically it should be 0.5/0.5.

## 4. Generalization Scenario
Suppose we train a neural network on a large corpus and it has seen the sentence "I went to Paris". It has also seen many sentences like "I went to [other cities]" and "I traveled to Paris". However, it never saw "I traveled to Paris" exactly. Would a neural language model likely assign a non-zero probability to "I traveled to Paris" as a whole (specifically "traveled" after "I")? Why or why not?

**Answer:** Yes, a neural language model would likely assign a reasonable non-zero probability to the phrase "I traveled to Paris" even if it wasn't seen verbatim. The model has seen "I went to Paris" and many instances of "I traveled to X" or "I traveled to London/New York/etc." From this it probably learned:
1. "I [past tense verb] to [Location]" is a common structure.
2. "traveled" is similar in context to "went" (both involve going somewhere).
3. "Paris" is a location that can follow "to".

So it would give a decent probability to "traveled" after "I" (especially when followed by "to Paris" context continuing) because it generalizes the concept that "I [verb] to [place]" is likely and "traveled" is a plausible verb there. In contrast, an N-gram model that never saw "I traveled" would give "traveled" a zero probability after "I". The neural model's embedding for "traveled" might be near "went" or "journeyed" etc., so it knows it fits in that slot even if exact sequence wasn't seen. This highlights the generalization power.

## 5. Looking Ahead
We discussed RNNs coming next for handling longer sequences. Can you think of a limitation of the fixed-window neural network (like our bigram NN) when it comes to context length? How does an RNN address it at a high level?

**Answer:** The fixed-window neural network (even if we extend to trigrams, 5-grams, etc.) has a limitation that it can only look at a **fixed number of words of context**. If something important happened earlier than that window, the model can't directly use it. For example, a fixed 2-word context model can't capture a dependency that spans 5 words. We could increase the window size, but there's a practical limit – the number of parameters and required data grows massively, and it's still fixed-length. Also, you don't know how large a window is enough; some dependencies in language can be arbitrarily long (like referring back to the subject at the start of a long sentence).

An **RNN (Recurrent Neural Network)** addresses this by not having a fixed window. Instead, it processes input word by word, and maintains a **hidden state** that carries information along as it goes through the sequence. Think of it like reading a sentence and remembering context in a "memory" vector that gets updated at each word. In theory, an RNN can carry forward information indefinitely (or as long as needed) because it recurrently uses its previous state combined with the new input to form a new state. This means earlier words can influence later output because their effect is preserved in the state. At a high level, RNNs introduce a kind of memory of arbitrary length, which fixed windows cannot. (We will find out basic RNNs have trouble with *very* long sequences due to things like vanishing gradients, but they conceptually can handle sequences of varying length – which is their big advantage.)
