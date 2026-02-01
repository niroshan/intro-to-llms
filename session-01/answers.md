# Session 1: Quiz Answers

## 1. Concept Check
In a bigram model, what information do we use to predict the next word?

a. The entire sentence so far.
b. Only the immediately preceding word.
c. The topic of the text.

**Answer:** b. Only the immediately preceding word (that's the Markov assumption for bigrams).

## 2. Probability Calculation
Suppose in a corpus, the word "London" appears 100 times. Out of those 100 occurrences, 30 times it's followed by "is", 20 times by "was", 5 times by "in", and the rest are all other words (each less frequent).

**(a)** What is P("is"|"London") according to the bigram model?

**(b)** If "London" is never followed by the word "sunny" in the corpus, what probability will the model assign to P("sunny"|"London")?

**Answer:**
(a) P(is|London) = 30/100 = 0.3 (30%).
(b) It will assign 0, since that pair wasn't seen (assuming no smoothing).

## 3. Practical Thinking
Why might a bigram model confuse the two sentences "The dog **chased** the cat" and "The dog **scared** the cat"? (Hint: think about what context the model uses for the word "the" before "cat".)

**Answer:** In both sentences, by the time the model is at the word "the" before "cat", the immediate previous word is "scared" in one case and "chased" in the other. A bigram model only looks one word back, so it considers P(cat|the) without remembering whether "the" came after "chased" or "scared". Actually, the model would be predicting "cat" after "the" (based on how often "the cat" appears overall) and **does not** take into account the verb two words back. Thus, it treats the contexts the same and might miss differences in meaning or grammar that depend on the verb.

## 4. Data Exploration
Take a small sample text (a paragraph from a book or article) and manually write down the bigrams it contains. (i) List five bigram pairs from the text. (ii) Identify if any word in that text always follows a particular word (like in our tiny corpus, "the" always followed by "cat"). What does that tell you about the text or the model?

**Answer:** *(Open-ended, but expected outcome: Students list actual bigrams from their sample. They might notice, for example, common pairs like ("Mr.", "Holmes") if the text was Sherlock Holmes, or ("New", "York"). If a word always follows another in the sample, it likely means the sample is small or that phrase is a fixed expression. The student should note that the model would then deterministically predict that follower after the given word.)*

## 5. Critical Thinking
What do you think will happen if we use a trigram model (N=3) instead of a bigram? What's one benefit and one drawback of using trigrams?

**Answer:** A trigram model uses the last 2 words of context instead of 1. Benefit: it captures more context, so it can distinguish situations that a bigram would confuse (it knows "scared the cat" vs "chased the cat" because the two-word contexts "scared the" vs "chased the" are different). It can generate more coherent phrases because it conditions on a broader history. Drawback: It needs significantly more data to get good statistics for every possible pair of context words. There are more possible 2-word combinations than single words, so data sparsity becomes an even bigger issue – many trigrams might never be seen. Also, the model is larger (more storage needed for counts) and slower to train if done naively. Essentially, higher N means more context but exponentially more data required.
