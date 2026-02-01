# Session 5: Quiz Answers

## 1. Understanding Hidden State
In your own words, explain what the hidden state $h_t$ represents in an RNN. Why is it useful for language modeling?

**Answer:** The hidden state $h_t$ is a vector that acts as the RNN's "memory" – it's a compressed summary of all the information the network has seen so far in the sequence (from word 1 up to word t). It's useful for language modeling because it allows the model to consider context beyond just the immediately previous word. The hidden state can encode things like the subject of a sentence, the topic being discussed, whether we're inside a quote, etc. This information can then influence predictions about what word comes next, enabling more coherent text generation than fixed-window models.

## 2. RNN vs Fixed-Window
A fixed-window neural network with window size 3 looks at the last 3 words. An RNN processes words one by one with a hidden state. What happens if important context is 5 words back? Compare how each model handles this.

**Answer:** 
- **Fixed-window (size 3):** Cannot directly see or use information from 5 words back – it's outside the window. The model is blind to that context, no matter how important it is. It would have to guess based only on the last 3 words.
- **RNN:** When it processed the important word 5 steps ago, it updated the hidden state to (hopefully) include relevant information. That information is then carried forward through all subsequent steps. By the time we're 5 words later, the hidden state still contains some representation of that earlier context (though it may be degraded due to vanishing gradients). The RNN at least has a *mechanism* to use distant information, even if imperfectly.

## 3. Vanishing Gradient Thought
If you made an RNN read a 100-word sentence and then asked it to recall the very first word, why is this hard for a basic RNN?

**Answer:** Because of the **vanishing gradient problem**. As the RNN processes 100 words, the influence of the first word on the hidden state tends to diminish with each step, especially if there's no reinforcement of that info later. When training, the error signal that would adjust weights to remember word1 until the end has to backpropagate through 100 time steps. At each step, gradients can get multiplied by factors (like derivative of tanh etc.), often < 1, causing them to shrink exponentially as they go back through 100 steps. By the time they reach weights related to word1, they're almost zero – meaning the model doesn't learn to preserve that info. So the network effectively "forgets" or can't carry specific info for that long. Without special mechanisms, it's hard for a basic RNN to carry something unchanged for 100 steps.

## 4. Practical Application
Consider the phrase: "The cat, which had been sleeping on the mat all afternoon, suddenly **woke** up." A bigram model might predict "sat" or something weird after "suddenly" because it only sees "suddenly ___". How could an RNN handle this better?

**Answer:** An RNN will process the whole sentence up to "suddenly", carrying along context. In this sentence, the subject "cat" and the verb "had been sleeping" are earlier. By the time it gets to "suddenly ___ up", the RNN's hidden state could contain the information that the cat was sleeping, so something related to waking is likely. The word "suddenly" alone doesn't tell you what comes after (bigram might guess randomly), but the RNN's memory includes that the cat had been sleeping. So an RNN has a chance to correctly predict "woke" because it knows the cat was asleep (so waking is a logical next event). Essentially, the RNN uses long-range dependency: it remembers the cat's state (sleeping) and when "suddenly" arrives, it can combine that knowledge to predict "woke".

## 5. Looking Forward
We mentioned LSTMs help with long-term memory. Based on what you know about the vanishing gradient problem, what do you think LSTMs might do differently? (Just speculation based on what we've learned.)

**Answer:** LSTMs probably have some mechanism to prevent gradients from vanishing over many steps. They might:
- Have a way to pass information more directly without as many transformations that shrink gradients
- Include "gates" that can decide what information to keep or throw away, rather than always mixing everything together
- Maintain a separate "memory cell" that can hold information without constant modification

(This is indeed what LSTMs do – they have a cell state that can carry information unchanged, plus input/forget/output gates that regulate information flow. We'll learn the details in the next session!)
