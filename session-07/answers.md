# Session 7: Quiz Answers

## 1. Gate Understanding
Explain in your own words what the "forget gate" does in an LSTM. Why is it called "forget"?

**Answer:** The forget gate decides what information to remove or "forget" from the cell state. It outputs values between 0 and 1 for each element in the cell state. A value near 0 means "forget this information" (multiply by ~0, erasing it), while a value near 1 means "keep this information" (multiply by ~1, preserving it). It's called "forget" because when it outputs low values, it causes the LSTM to forget/discard that part of its memory.

## 2. Cell State vs Hidden State
An LSTM has both a cell state ($C_t$) and a hidden state ($h_t$). What's the difference in their roles?

**Answer:** 
- **Cell state ($C_t$)**: The "long-term memory" – it's the conveyor belt that carries information through time. It can maintain information with minimal transformation if the gates allow.
- **Hidden state ($h_t$)**: The "working memory" or output – it's what gets sent to the next layer and is used for predictions. It's derived from the cell state but filtered through the output gate.

Think of it this way: the cell state stores everything you might need to remember, while the hidden state is what you're actively thinking about right now.

## 3. Why Gates Help Gradients
How do gates help solve the vanishing gradient problem?

**Answer:** Gates help because:
1. The cell state update is **additive** ($C_t = f_t \odot C_{t-1} + ...$) rather than multiplicative. This means gradients can flow through the addition operation without being multiplied by small factors at each step.
2. When the forget gate is near 1 and the input gate is near 0, the cell state passes through almost unchanged ($C_t \approx C_{t-1}$), providing a direct path for gradients.
3. The gates themselves are learned, so the network can learn to keep the gradient-carrying paths open when long-term dependencies need to be learned.

## 4. LSTM vs GRU
What's the main structural difference between LSTM and GRU? When might you choose one over the other?

**Answer:** 
- **LSTM**: Has 3 gates (forget, input, output) and separate cell state + hidden state
- **GRU**: Has 2 gates (reset, update) and only one combined state

Choose **LSTM** when:
- You have lots of data and computational resources
- The task has very long-term dependencies
- You want more fine-grained control over memory

Choose **GRU** when:
- You want faster training (fewer parameters)
- You have limited data (simpler model may generalize better)
- The dependencies aren't extremely long-range

In practice, they often perform similarly, so try both and see what works for your specific task.

## 5. Practical Observation
In the code example, both RNN and LSTM might achieve similar low loss on the training sequence. Why might LSTM still be preferred even when both can "memorize" the training data?

**Answer:** Even if both achieve low training loss:
1. **Generalization**: LSTM may generalize better to new, unseen sequences because it has learned more robust representations
2. **Convergence speed**: LSTM often reaches low loss faster, especially for sequences with repeated patterns (like "twinkle twinkle")
3. **Longer sequences**: If we made the sequence longer, the RNN's performance would degrade while LSTM would remain stable
4. **Robustness**: LSTM's memory is more reliable – it won't "drift" as much when generating long outputs
5. **Future scaling**: If you later want to train on longer, more complex text, the LSTM architecture is already suited for it
