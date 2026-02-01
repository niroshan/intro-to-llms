# Session 1: Quiz

## 1. Concept Check
In a bigram model, what information do we use to predict the next word?

a. The entire sentence so far.
b. Only the immediately preceding word.
c. The topic of the text.

## 2. Probability Calculation
Suppose in a corpus, the word "London" appears 100 times. Out of those 100 occurrences, 30 times it's followed by "is", 20 times by "was", 5 times by "in", and the rest are all other words (each less frequent).

**(a)** What is P("is"|"London") according to the bigram model?

**(b)** If "London" is never followed by the word "sunny" in the corpus, what probability will the model assign to P("sunny"|"London")?

## 3. Practical Thinking
Why might a bigram model confuse the two sentences "The dog **chased** the cat" and "The dog **scared** the cat"? (Hint: think about what context the model uses for the word "the" before "cat".)

## 4. Data Exploration
Take a small sample text (a paragraph from a book or article) and manually write down the bigrams it contains. (i) List five bigram pairs from the text. (ii) Identify if any word in that text always follows a particular word (like in our tiny corpus, "the" always followed by "cat"). What does that tell you about the text or the model?

## 5. Critical Thinking
What do you think will happen if we use a trigram model (N=3) instead of a bigram? What's one benefit and one drawback of using trigrams?
