# Session 6: Quiz

## 1. Understanding Hidden State
In the code, why do we call `h = h.detach()` at the end of each epoch's training loop? What could happen if we didn't?

## 2. Observation
During generation, we used `torch.multinomial` to sample the next character instead of always taking `torch.argmax` (the most likely character). If our model has learned the training sequence almost perfectly, what's the difference between these two approaches in output?

## 3. RNN vs. Feed-forward
Consider a situation with the phrase: "The cat, which had been sleeping on the mat all afternoon, suddenly **woke** up." A bigram model might predict "sat" or something weird after "suddenly" because it only sees "suddenly ___". How could an RNN handle this better?

## 4. Vanishing Gradient Practical
If you made an RNN read a 100-word sentence and then asked it to recall the very first word, why is this hard for a basic RNN?

## 5. Application
Name a real-world application where RNNs (or their improved versions) were traditionally used, and briefly describe how the sequential nature is crucial there.
