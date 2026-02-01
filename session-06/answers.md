# Session 6: Quiz Answers

## 1. Understanding Hidden State
In the code, why do we call `h = h.detach()` at the end of each epoch's training loop? What could happen if we didn't?

**Answer:** We detach the hidden state to break the computation graph between epochs. If we didn't, PyTorch would try to backpropagate gradients from the next epoch through the hidden state into the previous epoch's computations, which doesn't make sense and would double count or accumulate gradients incorrectly. Essentially, without detaching, the model would treat the sequence as continuing from epoch to epoch, which is not what we want (each epoch is a fresh training pass). Detaching ensures that each epoch's gradient calculations start anew with respect to the model parameters.

## 2. Observation
During generation, we used `torch.multinomial` to sample the next character instead of always taking `torch.argmax` (the most likely character). If our model has learned the training sequence almost perfectly, what's the difference between these two approaches in output?

**Answer:** If the model learned the sequence almost perfectly, at each step the correct next character likely has probability near 1.0. In that case, whether we sample or take argmax, we will get the same result (the correct next character) because it dominates. However, with more data or uncertainty, using multinomial adds randomness – it can produce different continuations according to probability distribution, not always the single highest probability sequence. Argmax would always give the single most likely next char (deterministic), which in our trivial case is the training sequence.

## 3. RNN vs. Feed-forward
Consider a situation with the phrase: "The cat, which had been sleeping on the mat all afternoon, suddenly **woke** up." A bigram model might predict "sat" or something weird after "suddenly" because it only sees "suddenly ___". How could an RNN handle this better?

**Answer:** An RNN will process the whole sentence up to "suddenly", carrying along context. The subject "cat" and the verb "had been sleeping" are earlier. By the time it gets to "suddenly ___", the RNN's hidden state could contain the information that the cat was sleeping. So an RNN has a chance to correctly predict "woke" because it knows the cat was asleep. A bigram can't use "sleeping" information because it's more than one word away.

## 4. Vanishing Gradient Practical
If you made an RNN read a 100-word sentence and then asked it to recall the very first word, why is this hard for a basic RNN?

**Answer:** Because of the **vanishing gradient problem**. The influence of the first word tends to diminish with each step. When training, the error signal has to backpropagate through 100 time steps. At each step, gradients get multiplied by factors often < 1, causing them to shrink exponentially. By the time they reach weights related to word1, they're almost zero – meaning the model doesn't learn to preserve that info for long distances.

## 5. Application
Name a real-world application where RNNs (or their improved versions) were traditionally used, and briefly describe how the sequential nature is crucial there.

**Answer:** **Machine Translation**. Traditionally, an RNN encoder-decoder model (often using LSTMs) was used. The encoder RNN reads a sentence in English word by word and compresses its meaning into a final hidden state. Then a decoder RNN starts generating a sentence in French, one word at a time, using that encoded information. The sequential nature is crucial because language is sequential: the meaning of a sentence is in the sequence of words. The RNN had to remember the whole source sentence to produce the correct translation.
