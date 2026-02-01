# Session 5: Quiz

## 1. Understanding Hidden State
In your own words, explain what the hidden state $h_t$ represents in an RNN. Why is it useful for language modeling?

## 2. RNN vs Fixed-Window
A fixed-window neural network with window size 3 looks at the last 3 words. An RNN processes words one by one with a hidden state. What happens if important context is 5 words back? Compare how each model handles this.

## 3. Vanishing Gradient Thought
If you made an RNN read a 100-word sentence and then asked it to recall the very first word, why is this hard for a basic RNN?

## 4. Practical Application
Consider the phrase: "The cat, which had been sleeping on the mat all afternoon, suddenly **woke** up." A bigram model might predict "sat" or something weird after "suddenly" because it only sees "suddenly ___". How could an RNN handle this better?

## 5. Looking Forward
We mentioned LSTMs help with long-term memory. Based on what you know about the vanishing gradient problem, what do you think LSTMs might do differently? (Just speculation based on what we've learned.)
