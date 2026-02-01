# Building a Mini LLM: 6-Week Educational Program

This program is designed as a **6-week, 12-session** course (1 hour each session, on Saturdays and Sundays) for a UK GCSE graduate with a background in Python and further maths. The goal is to guide the student in building a **mini Language Model (LLM)** from scratch and develop a deep understanding of the theory, code, and mathematics behind it. Each session includes:

1. **Teaching Script:** A step-by-step instructional explanation of concepts, using analogies, simple math derivations, and examples appropriate for a GCSE-level student.

2. **Presentation Slides (text outline):** A summary of key points in a slide-friendly format, including visuals, diagrams (using Mermaid flowcharts), and bullet-point takeaways.

3. **Code Walkthrough:** A plain Python script (heavily commented) illustrating the core coding tasks of the session, designed to be beginner-friendly.

4. **Quiz/Exercises:** A short quiz or practice exercises to test understanding of the session’s material.

All content is beginner-friendly, minimizes jargon, and focuses on building intuition while maintaining enough rigor to prepare the student to explain how modern LLMs work in detail.

---

## Week 1: Introduction to Language Models with N-grams

### Session 1: What is a Language Model? (N-gram Basics)

#### *Teaching Script*

*Introduction & Motivation:*  
Welcome to the world of **Language Models (LMs)**\! A language model is simply a system that **predicts the next word** in a sequence, given some previous words. You might not realize it, but you encounter language models every day – for example, when your phone suggests the next word while you’re typing a message. Today, we’ll start with the most basic kind of language model: the **N-gram model**, specifically a **bigram model** (which is an N-gram model where N=2). This will introduce key ideas of how we can use probability and counting to make predictions about language.

*Key Concept – Predicting the Next Word:*  
Imagine you have the phrase "**Good morning, how are ...**". Even without finishing the sentence, you can guess the next word might be "**you**". A language model does something similar: based on the words it has already seen (“Good morning, how are”), it predicts what comes next. Formally, if we denote the previous words as a history h and the next word as w, a language model gives a probability Pw|h – the likelihood that w is the next word given history h. In our example, Pyou|“Good morning, how are” would hopefully be higher than, say, Pbanana|“Good morning, how are”\! But how do we come up with these probabilities?

*The Simplest Approach – Counting (N-grams):*  
One straightforward way is to use **counts of word sequences** from a large text, i.e., a corpus. The idea is: the more often a word follows a given history in real text, the more likely our model should predict it. However, counting whole long histories (like entire sentences) is impractical because you’d rarely see the exact same long sentence twice[\[1\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=chain%20rule%20doesn%E2%80%99t%20really%20seem,1%20The%20Markov%20assumption)[\[2\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=of%20a%20word%20given%20its,words%2C%20instead%20of%20computing%20the). This is where we simplify the problem using the **Markov assumption**[\[3\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=word%2C%20we%20are%20thus%20making,future%20unit%20without%20looking%20too). The Markov assumption says that to predict the next word, you don’t need the *entire* history – just the last few words may suffice. For an **N-gram model**, we only look at the last N-1 words. In a **bigram (2-gram) model**, we look at **only the last 1 word**. Essentially, we approximate:

Pwn∣w1,w2,,wn−1Pwn∣wn−1 .

That is, “the probability of the next word wn given all prior words is approximated by the probability of wn given just the immediately preceding word wn−1”[\[3\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=word%2C%20we%20are%20thus%20making,future%20unit%20without%20looking%20too). This simplification is our **bigram assumption** (a specific case of the Markov assumption). By making this assumption, we only need to count how often each word follows each other word.

*Bigram Model Mechanics:*  
In a bigram model, we calculate Pnext word∣current word using counts from our corpus. For example, suppose in our corpus the word “the” is followed by “cat” 50 times and followed by “dog” 20 times. Then the probability P“cat”|“the” is estimated as 5050+20=0.714 (assuming “cat” and “dog” are the only two possibilities after “the” in this tiny corpus). More generally,

Pwnext∣wcurrent=Countwcurrent,wnextCountwcurrent.

This fraction means *“of all occurrences of the current word, how many times was it followed by the next word?”*. By computing this for every possible word pair (bigram) that appears in our text, we build a probability table. This table is essentially the “knowledge” of our bigram language model.

*Example:*  
Imagine a very small corpus of text: "*the cat sat on the mat. the cat ate a fish.*" From this corpus, we can gather some bigram counts: \- “the” followed by “cat” occurs 2 times. \- “cat” followed by “sat” occurs 1 time. \- “cat” followed by “ate” occurs 1 time. \- “sat” followed by “on” occurs 1 time. \- … and so on for each adjacent word pair in the corpus.

We can then turn these counts into probabilities. For instance, after the word “the” in our corpus, the next word was “cat” 2 out of 2 times (assuming only those two sentences). So our model would predict “cat” after “the” with probability 1 (or 100%) in this tiny corpus. After “cat”, the next word was “sat” once and “ate” once, so Psat|cat=0.5 and Pate|cat=0.5. In a real, much larger corpus, these probabilities would be spread across many possible next words.

*Using the Model – Predicting & Generating Text:*  
Once we have these bigram probabilities, we can **predict the next word** by choosing the one with the highest probability given the current word. For example, if our model has Pmorning|good=0.4 and Pnight|good=0.1 (and other words have the rest of the probability), it would predict “morning” after “good” because it’s more likely in the training data. We can also **generate text** with the model by treating it like a little game: start with a seed word (say “Once”), then use the model to randomly pick the next word according to the learned probabilities, append it, then move on. This is essentially how we *sample* from the model to create new sentences. For instance, if starting at “Once”, if our model knows “Once” is often followed by “upon”, it will likely pick “upon” as next. Then with “upon” as current, it might pick “a”, then “time”, and so on, perhaps generating the classic start “Once upon a time”.

Behind the scenes, generating text is like walking through a **Markov chain** where each state is the current word, and transitions to the next word happen with probabilities determined by our bigram table[\[4\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=P%28wn,future%20unit%20without%20looking%20too). One way to visualize this is to imagine a map of words connected by arrows: each arrow from word A to word B is labeled with PB|A. Starting at some word, you follow an arrow to pick the next word (randomly, but weighted by those probabilities), and continue.

*Analogy:*  
Think of a bigram model like a **predictive text keyboard** or an auto-complete. When you type a word, the system looks at that word and suggests what the next word could be. It's not thinking deeply or understanding meaning – it's basically saying “In the texts I’ve seen, after *this* word, *these* words often come next.” If “Good morning” was common in its memory, then after you type "Good", it might suggest "morning". If you typed "Good" and then "night" often follows in its memory, it might suggest "night" next. This is exactly what our bigram model does with probabilities instead of absolute suggestions.

*Limitations of N-gram Models:*  
While N-gram models (like our bigram model) are easy to understand and implement, they have **limitations**. One major issue is that they have **limited memory** of context. A bigram model only knows about the last one word – it cannot handle situations where the context of two or more words ago is important. For example, consider the two sentences: "I **ate** a sandwich and I was full." vs "I **saw** a sandwich and I was full." A bigram model, only looking one word back, might not see a difference after the word "I" – it doesn’t “remember” whether the word before "a sandwich" was "ate" or "saw" by the time it gets to "was full." Thus, it might treat both contexts similarly, even though the meaning differs. Generally, N-gram models can struggle when **longer-range dependencies** matter (like subject-verb agreement that might be several words apart, or the topic of a paragraph influencing word choice later).

Another limitation is the **data sparsity** problem. Language is vast and creative – many valid sentences or phrases will not appear in your training corpus. If an N-gram model never saw a particular word sequence during training, it will assign it zero probability, meaning it thinks that sequence is impossible (when in reality it might just be rare or absent in the training data). For example, if our corpus never contained the phrase "astronaut launches", a bigram model would have no data for Plaunches|astronaut and would essentially consider it an impossible pair. This isn’t true in real language – it’s just we didn’t see it in our limited data. There are techniques like **smoothing** to handle this (which basically steal a little probability mass from frequent events and give it to unseen events), but that’s a bit beyond our initial session. For now, it’s enough to recognize that simple models need a lot of data to cover many possibilities, and even then, they have blind spots.

In summary, the bigram model is our first stepping stone. It captures some local structure of language (common two-word combinations) and will serve as a baseline. By the end of this week, you will *implement* a bigram model yourself, use it to analyze text, and even generate a few random sentences. This simple model will also help motivate the more advanced approaches in subsequent weeks: we’ll see why we need more powerful models (like neural networks and attention) to handle the complexities of natural language.

#### *Presentation Slides*

* **What’s a Language Model (LM)?**

* A system that predicts the **next word** given previous words (context).

* Everyday example: your phone’s **autocorrect/suggestions** guessing your next word as you type.

* We measure this as a probability Pnext word∣previous words.

* **Bigram Model – The Simplest LM:**

* Uses only **1 previous word** to predict the next word (N-gram with N=2, so N-1=1 context word).

* **Markov assumption:** the next word depends only on the last word, not the entire history[\[4\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=P%28wn,future%20unit%20without%20looking%20too).

* Learns from text by **counting word pairs**. Example: If “rainy” is followed by “day” 30 times and “night” 10 times, Pday|rainy=3040=0.75.

* **How a Bigram Model Works:**

* **Training:** Go through a large text and tally counts of every word bigram (pair). Compute probabilities: PB|A=CountA followed by BCountA.

* **Prediction:** Given a current word, look up which words tend to follow it and with what probabilities.

* **Generation:** Start with a seed word and repeatedly *roll the dice* using these probabilities to pick each next word. This creates a random sentence that *sounds* like the training text in terms of local word pair patterns.

* **Visualization – Text Generation as a Flowchart:**

* The model moves from word to word, using bigram probabilities as transition rules.

* Each step: look at current word, pick next word based on learned odds.

flowchart LR  
  Start\["\<start\>"\] \--\> A\["Current word \= A"\]  
  A \--\>|"P(B|A)"| B\["Next word \= B"\]  
  A \--\>|"P(C|A)"| C\["Next word \= C"\]  
  B \--\>|"P(D|B)"| D\["Next word \= D"\]  
  C \--\>|"P(E|C)"| E\["Next word \= E"\]  
  D \--\> ...   
  E \--\> ... 

*(Flowchart: At “Current word \= A”, model may go to B with probability P(B|A) or to C with probability P(C|A). From B, it can go to D next, etc. This illustrates random walk through words.)*

* **Strengths & Weaknesses of N-gram Models:**

* ✅ **Easy to understand & implement:** Just counting and probability.

* ✅ Captures common local patterns (“thank you”, “Good morning”).

* ❌ **Short memory:** Knows only the last N-1 words. Can’t handle longer context or complex dependencies.

* ❌ **Data sparsity:** Assigns zero probability to any never-seen combination (e.g., new phrases), which is unrealistic without smoothing.

* ❌ Can produce repetitive or nonsensical output if context is insufficient (“the the the …”).

* **Why We Need More:**

* Language has long-range structure (e.g., a word later might depend on one several words earlier).

* To improve, we’ll need models that **remember more** than just one or two words – this motivates moving beyond simple N-grams to models with **memory and learning** (foreshadowing neural networks and beyond).

#### *Code Walkthrough*

\# Bigram Language Model \- Counting and Probability Demo

\# 1\. Sample corpus (small text) for demonstration:  
corpus \= "the cat sat on the mat. the cat ate a fish."

\# 2\. Preprocessing: split text into words (tokenization)  
words \= corpus.lower().replace(".", " \<END\> ").split()  
\# We insert \<END\> token to mark sentence end for clarity (optional)  
\# Now 'words' is a list like: \['the','cat','sat','on','the','mat','\<END\>',  
\#                               'the','cat','ate','a','fish','\<END\>'\]

\# 3\. Count unigrams (single words) and bigrams (pairs of consecutive words)  
import collections  
unigram\_counts \= collections.Counter()  
bigram\_counts \= collections.Counter()

\# Populate counts by iterating through word list  
prev\_word \= None  
for word in words:  
    unigram\_counts\[word\] \+= 1  
    if prev\_word is not None:             \# if there is a previous word (not at start)  
        bigram\_counts\[(prev\_word, word)\] \+= 1  
    prev\_word \= word  
\# Note: This simple loop treats the corpus as one stream.   
\# Using \<END\> ensures we count sentence-ending pairs appropriately.

\# Display the counts (for understanding)  
print("Unigram counts:", unigram\_counts)  
print("Bigram counts:", bigram\_counts)

\# 4\. Compute bigram probabilities  
bigram\_probs \= {}  
for (w1, w2), count in bigram\_counts.items():  
    prob \= count / unigram\_counts\[w1\]  
    bigram\_probs\[(w1, w2)\] \= prob  
    \# Explanation: P(w2 | w1) \= Count(w1 w2) / Count(w1)

\# Display some example probabilities  
example\_pairs \= \[("the", "cat"), ("cat", "sat"), ("cat", "ate"), ("cat", "mouse")\]  
for pair in example\_pairs:  
    if pair in bigram\_probs:  
        print(f"P({pair\[1\]}|{pair\[0\]}) \= {bigram\_probs\[pair\]:.2f}")  
    else:  
        \# If a pair never occurred, we can treat probability as 0 (or apply smoothing later)  
        print(f"P({pair\[1\]}|{pair\[0\]}) \= 0 (unseen pair)")

\# 5\. Using the model to generate a random sentence  
import random  
sentence \= \[\]  
current\_word \= "the"   \# Let's start with "the" for example  
for \_ in range(10):    \# generate up to 10 words  
    sentence.append(current\_word)  
    \# Collect possible next words and their probabilities for current\_word  
    candidates \= \[(w2, prob) for (w1, w2), prob in bigram\_probs.items() if w1 \== current\_word\]  
    if not candidates:  
        break  \# stop if no known continuation (e.g., reached an \<END\> or unknown word)  
    \# Use random.choices to pick next word according to probabilities  
    next\_words, probs \= zip(\*candidates)  
    current\_word \= random.choices(next\_words, weights=probs, k=1)\[0\]  
    if current\_word \== "\<END\>":  
        break  \# end generation if end-of-sentence token is reached

generated\_sentence \= " ".join(sentence)  
print("Generated sentence:", generated\_sentence)

**What this code does:**  
\- We define a small corpus (just two sentences for this demo). In a real scenario, this could be a large text file. We lowercase everything and insert \<END\> tokens to mark sentence boundaries, which helps the model treat end-of-sentence explicitly.  
\- We **tokenize** the text into a list of words.  
\- We then **count unigrams and bigrams** using a loop. unigram\_counts\[word\] counts each word’s occurrences. bigram\_counts\[(prev\_word, word)\] counts each observed word pair. For example, in our sample corpus, the pair (“the”, “cat”) will be counted twice.  
\- Next, we compute bigram **probabilities**. For each bigram (w1, w2) in our counts, we divide by the total count of w1. The result is stored in bigram\_probs. So if (“the”, “cat”) count is 2 and “the” occurred 2 times in total, P(cat|the) becomes 1.0 in this tiny corpus. We demonstrate by printing a few example probabilities, including an unseen pair (“cat”, “mouse”) which would not be in bigram\_probs (hence probability 0 without smoothing).  
\- Finally, we show how to **generate text**. We start with a chosen current\_word (here “the”). Then in a loop, we look up all candidate next words that can follow current\_word (from our bigram\_probs). We use random.choices with weights equal to those probabilities to randomly pick the next word. We append the current word to the sentence list as part of the output, update current\_word to the chosen next word, and repeat. We break out if we hit an \<END\> or if there’s no known next word. The result is a randomly generated sentence stored in generated\_sentence.  
\- Running this code might produce output like:

Unigram counts: Counter({'the': 2, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1, '\<END\>': 2, 'ate': 1, 'a': 1, 'fish': 1})  
Bigram counts: Counter({('the', 'cat'): 2, ('cat', 'sat'): 1, ('sat', 'on'): 1, ('on', 'the'): 1, ('the', 'mat'): 1, ('mat', '\<END\>'): 1, ('cat', 'ate'): 1, ('ate', 'a'): 1, ('a', 'fish'): 1, ('fish', '\<END\>'): 1})  
P(cat|the) \= 1.00  
P(sat|cat) \= 0.50  
P(ate|cat) \= 0.50  
P(mouse|cat) \= 0 (unseen pair)  
Generated sentence: the cat sat on the mat

In this run, after “the”, the only next word observed in the corpus was “cat” (so it chose “cat”). After “cat”, it had two choices (“sat” or “ate” each with 0.5 probability) and picked “sat”, and so on... It eventually produced "the cat sat on the mat", which is indeed one of the original sentences. Because the corpus is tiny, the model tends to reproduce exact sentences. With a larger corpus, you’d see more varied (sometimes creative, sometimes nonsensical) outputs.

**Discussion:**  
This simple code demonstrates the core of an N-gram model. It reveals a few things: \- Common pairs get high probability (in our data "the→cat" was certain). If we had more varied data, probabilities would distribute more smoothly. \- Unseen pairs are simply not generated (we had no chance of "cat → mouse" because it wasn’t in training data, leading to zero probability). \- The generated text respects the patterns of the training data but doesn’t have any global planning – it’s *local*. In our case, it recreated a learned sentence. With random picking, it could have also done "the cat ate a fish" (the other sentence) or conceivably mixed parts like "the cat ate a fish \<END\> the cat sat on the mat". \- If we try to generate a very long sentence, the model might wander into a dead-end (a word that was only ever seen at the end of a sentence, so it can’t continue). In our code, we break out if no candidates or if \<END\> is reached.

This bigram model is a baseline. Keep this code in mind – you now have a functioning (if rudimentary) language model\! In the next session, we’ll use it to analyze a slightly larger corpus and get some interesting stats and sentences out, and that will wrap up our N-gram exploration before moving on to neural network approaches.

#### *Quiz / Exercises*

1. **Concept Check:** In a bigram model, what information do we use to predict the next word?  
   a. The entire sentence so far.  
   b. Only the immediately preceding word.  
   c. The topic of the text.  
   **Answer:** b. Only the immediately preceding word (that’s the Markov assumption for bigrams).

2. Suppose in a corpus, the word “London” appears 100 times. Out of those 100 occurrences, 30 times it’s followed by “is”, 20 times by “was”, 5 times by “in”, and the rest are all other words (each less frequent).

3. **(a)** What is P“is”|“London” according to the bigram model?

4. **(b)** If “London” is never followed by the word “sunny” in the corpus, what probability will the model assign to P“sunny”|“London”?  
   **Answer:** (a) Pis|London=30100=0.3 (30%). (b) It will assign 0, since that pair wasn’t seen (assuming no smoothing).

5. **Practical Thinking:** Why might a bigram model confuse the two sentences “The dog **chased** the cat” and “The dog **scared** the cat”? (Hint: think about what context the model uses for the word “the” before “cat”.)  
   **Answer:** In both sentences, by the time the model is at the word “the” before “cat”, the immediate previous word is “scared” in one case and “chased” in the other. A bigram model only looks one word back, so it considers Pcat|the without remembering whether “the” came after “chased” or “scared”. Actually, the model would be predicting “cat” after “the” (based on how often “the cat” appears overall) and **does not** take into account the verb two words back. Thus, it treats the contexts the same and might miss differences in meaning or grammar that depend on the verb.

6. **Data Exploration:** Take a small sample text (a paragraph from a book or article) and manually write down the bigrams it contains. (i) List five bigram pairs from the text. (ii) Identify if any word in that text always follows a particular word (like in our tiny corpus, “the” always followed by “cat”). What does that tell you about the text or the model?  
   **Answer:** *(Open-ended, but expected outcome: Students list actual bigrams from their sample. They might notice, for example, common pairs like (“Mr.”, “Holmes”) if the text was Sherlock Holmes, or (“New”, “York”). If a word always follows another in the sample, it likely means the sample is small or that phrase is a fixed expression. The student should note that the model would then deterministically predict that follower after the given word.)*

7. **Critical Thinking:** What do you think will happen if we use a trigram model (N=3) instead of a bigram? What’s one benefit and one drawback of using trigrams?  
   **Answer:** A trigram model uses the last 2 words of context instead of 1\. Benefit: it captures more context, so it can distinguish situations that a bigram would confuse (it knows “scared the cat” vs “chased the cat” because the two-word contexts “scared the” vs “chased the” are different). It can generate more coherent phrases because it conditions on a broader history. Drawback: It needs significantly more data to get good statistics for every possible pair of context words. There are more possible 2-word combinations than single words, so data sparsity becomes an even bigger issue – many trigrams might never be seen. Also, the model is larger (more storage needed for counts) and slower to train if done naively. Essentially, higher N means more context but exponentially more data required.

---

### Session 2: Building and Using a Bigram Model (Analysis & Demo)

#### *Teaching Script*

*Recap:*  
Last session, we learned what a bigram language model is and how it uses word pair frequencies to predict the next word. We even sketched a simple bigram model in code. Today, we’ll get hands-on and **build a bigram model from scratch**, then use it to analyze a text corpus and generate some example text. This will solidify the concept and also illustrate the model’s strengths and weaknesses in action.

*Step 1: Choosing a Corpus (Text Data):*  
To train any language model, we need text data. The larger and more representative the text, the better the model generally becomes. For our demo purposes, we might use a relatively small corpus (for example, a few chapters of a book, a collection of nursery rhymes, or any text you find interesting). The first step is **loading and preprocessing** the text: \- We typically convert all text to a uniform case (lowercase) to avoid treating “The” and “the” as different words. \- We might remove punctuation or handle it in a simple way (e.g., treat sentence-end punctuation as markers or just remove them). \- We then split the text into words (tokenization). For simplicity, splitting on whitespace and punctuation is often okay to start, though in real NLP, tokenization can get more complex (handling contractions, etc.).

*Step 2: Building the Bigram Table:*  
Using the approach from Session 1, we go through the text word by word and count how often each word follows each other word. This will produce: \- A dictionary (or table) of bigram counts, e.g., count\["rain"\]\["fall"\] \= 15 meaning "rain fall" occurred 15 times. \- A dictionary of single word counts, e.g., count\["rain"\] \= 30 meaning "rain" appeared 30 times in total (so we know how many opportunities there were for any word to follow "rain").

From these, we compute the probability Pnext|current=count(current,next)count(current). It might be useful not just to compute these probabilities but also to inspect them. We can find, for instance: \- What is the most likely word after "rain"? \- What words can start a sentence (maybe look at what often follows a period or start-of-sentence token)? \- Which word has the most diverse set of followers vs which is almost always followed by one particular word (like "Good" might often be followed by "morning" or "evening").

*Step 3: Exploring the Model (Corpus Analysis):*  
Before generating text, it’s insightful to analyze the corpus via the model’s lens: \- **Frequency Analysis:** Which words are most common overall (unigram frequencies)? Often, function words like "the", "of", "and" top the list in English. This is expected and tells us such words will heavily influence predictions. \- **Likely Pairs:** For a given word, what are the top 3 most likely next words according to the model? For example, if you use a corpus of fairy tales, after "once", the most likely next word might be "upon" (forming "once upon"). After "he", common next words might be "said", "was", "had", etc. \- **Surprising Patterns:** Sometimes models pick up on names or specific phrases. If "New York" appears often, the model will strongly connect "New" \-\> "York". If "ladies and gentlemen" appears often, "ladies" \-\> "and" might be near 100% and "and" \-\> "gentlemen" similarly high when following "ladies". \- We can query our model data structure to answer questions like "How many distinct words follow 'the'?" or "Is there any word that never appears as a next word for any other? (Those would be words that only start sentences perhaps.)"

By examining these, we connect the raw data to how the model “sees” language. This also helps highlight limitations; for example, if you find P“bank”|“the” is high because of phrases like "the bank", the model doesn’t know if "bank" means a financial bank or river bank – it’s just counting. It has no concept of meaning, only usage frequency.

*Step 4: Generating Text:*  
Now the fun part – using the bigram model to generate text. This is like those “predictive text” games. We need a starting point: typically a start-of-sentence token or simply pick a word to start with (maybe a capital letter word if we preserved case and punctuation for realism, or just any common word). Then repeatedly: 1\. Look up the distribution of next words after the current word. 2\. Randomly select the next word according to that distribution. 3\. Append it and move on.

We continue until we hit a special end-of-sentence token (if we included one) or until we have generated a desired length of text.

*Important detail:* If the model ever lands on a word that was never seen as a "current word" in training (which can happen if that word only ever appeared at the very end of the corpus, like a last word of the entire text with no following word), then we have no next-word data. To handle this, we might include a generic end-of-sentence token in training, so every sentence has a known "next word" at its end (the \<END\> marker), ensuring every word in a sentence has some recorded follower (even if it's \<END\>). In generation, if we hit \<END\>, we can stop or choose to start a new sentence.

*Let's consider an example:*  
If we trained on a bunch of nursery rhymes, and we start with "Jack", the model might go: \- "Jack" (start) \-\> likely next "and" (if "Jack and Jill" is common, P(and|Jack) is high). \- "and" \-\> likely "Jill". \- "Jill" \-\> likely "went" (from "Jill went up the hill"). \- ... and so on.

It might end up recreating "Jack and Jill went up the hill" if those probabilities dominate. Or it could veer off if there's variance in data (maybe "Jack jumped over" from "jack jumped over the candlestick" might also be a possibility if that rhyme was in the training data). Because generation involves randomness, each run can produce different outputs. Sometimes the output will be valid-sounding; other times it might be grammatically awkward or abruptly end if an uncommon sequence was chosen.

*Observation:* The bigram model generally produces **locally plausible** text but not globally coherent text. It might capture common two-word combos, but it has **no memory of earlier context beyond one word**. For instance, it could generate a sentence like: "The cat ate a fish \<END\> The cat sat on a fish \<END\>" etc., which individually each small segment was something seen in training, but the overall might sound repetitive or silly. Or it might do something grammatically off if it stitches together mismatched pairs across a sentence boundary.

This demonstrates why, although N-gram models can generate *somewhat realistic gibberish*, they often fail to capture the nuance of longer text. Real examples: a trigram model might produce sentences that kind of start making sense and then wander off as the context gets lost.

*Reflection – Preparing for Next Steps:*  
By the end of this session, you will have a working bigram model and first-hand experience with its output. Take a moment to reflect on what was easy or hard for this model: \- It was easy to set up with counting. \- It quickly learned common short patterns. \- But to “remember” anything beyond one word, it fails. If the corpus had a sentence like "Alice took the rabbit and then she followed it down the hole", a bigram model at "she" only sees "she" preceded by "then" (maybe) and has no clue that "Alice" was the subject. It might just as well have seen "she" often followed by "said" and output "she said ..." continuing incorrectly for that context.

These limitations clearly point us towards wanting a model that can learn more **flexible** patterns and **longer dependencies**. That is exactly where we are headed: starting next week, we’ll introduce neural network-based language models, which address some of these issues by learning distributed representations and theoretically considering more context. But before that, let’s cement our understanding of bigrams with coding and experimentation now.

#### *Presentation Slides*

* **Preparing Text Data:**

* Use a corpus (collection of text) as training data.

* **Clean & tokenize:** lowercase the text, remove punctuation or treat it specially, split into words.

* Optionally add special tokens (e.g., \<END\> for sentence boundaries) to help model sentence starts/ends.

* **Building the Bigram Model:**

* **Count word occurrences:** how often each word appears (unigram counts).

* **Count word pairs:** how often each word is followed by another (bigram counts).

* **Calculate probabilities:** Pnext|current=Count(current,next)Count(current). Store these in a table/dictionary for lookup.

* The result: a probability table of size \~ (number of unique word pairs seen).

* **Analyzing the Corpus via the Model:**

* *Most frequent words:* Identify top 5 most common words (likely “the”, “and”, etc.).

* *Common followers:* For a given word, list the top 3 next words with probabilities. (E.g., after "once", Pupon might be high; after "he", Psaid might be high.)

* *Diversity:* Some words have many possible followers (e.g., "the" can be followed by dozens of different words), while others are more fixed (e.g., "New" is almost always followed by "York" in a corpus about geography).

* *Examples:* If we see "New" 50 times and 45 of those are "New York", then PYork|New=0.9 (90%). The model strongly links "New" \-\> "York".

* **Generating Text from the Bigram Model:**

* Pick a **start word** (or a start token).

* Loop: at current word, **randomly choose** the next word based on the learned probabilities (higher probability \= more likely to be picked).

* Continue until reaching an \<END\> token or a set length.

* Each run can produce a different random sentence, but *locally* it will use plausible word pairs from training data.

* **Example Generation Flow (bigram):**

flowchart LR  
 subgraph Bigram Model Generation  
  state1\["Word 1: Jack"\] \--\> state2\["Word 2: and"\]   
  state2 \--\> state3\["Word 3: Jill"\]   
  state3 \--\> state4\["Word 4: went"\]   
  state4 \--\> state5\["Word 5: up"\]   
  state5 \--\> state6\["Word 6: the"\]   
  state6 \--\> state7\["Word 7: hill"\]   
  state7 \--\> endToken\["\<END\>"\]  
 end

*Diagram: Starting with "Jack", the model picked "and" (because in training "Jack and" was common), then "Jill", then "went", etc., eventually producing the nursery rhyme start "Jack and Jill went up the hill". Each arrow choice is guided by bigram probabilities.*

* **Output Characteristics:**

* Locally coherent phrases (familiar two-word combos).

* Globally, might wander or repeat. E.g., could generate "and the cat and the cat ..." if "and the" is common, because after "and" \-\> "the", then "the" \-\> maybe "cat", then "cat" \-\> maybe "and" again, forming a loop.

* No understanding of grammar beyond 2-word context: might start a sentence in one style and end in another, or lose subject-verb agreement if the dependency is longer than a bigram.

* **Limitations Observed:**

* **Zero probabilities:** Any unseen pair can’t be generated (if never saw "purple elephant", model will never say "purple elephant"). Smoothing could mitigate this, but our basic model doesn’t have it.

* **Short memory:** Can’t enforce consistency. Example: if the text to generate is long, a bigram model has no memory of what the subject was in the beginning by the time it’s in the middle.

* **Repetitiveness:** With certain structures, model can loop (e.g., if "X \-\> Y" and "Y \-\> X" are both likely, it might alternate X, Y, X, Y...). Human text usually doesn’t repeat like that beyond a point, but the model doesn’t “know” that repeating four times is odd. It just follows pair probabilities each step.

* **Why Not Stop at N-grams?**

* They’re **too rigid** and **data-hungry**. To get better, you’d increase N (trigram, 4-gram, etc.), but that demands exponentially more data to avoid zeros.

* They can’t capture *abstract* similarities. For example, they treat every word as unrelated to others except by direct neighbor counts. They wouldn’t generalize that "dog" and "cat" might appear in similar contexts; if “the dog barked” is in training but “the cat barked” is not, a bigram model gives zero to the latter even though a cat *could* bark (theoretically\!).

* **Foreshadowing:** Neural networks will address these by *learning from data* in a smoother way, and by using continuous vector representations (embeddings) that can generalize to unseen combinations. Also, recurrent and attention-based models will extend effective memory.

#### *Code Walkthrough*

\# Building a bigram model for a larger text corpus and generating text

\# 1\. Load a sample text corpus  
\# For demonstration, we'll use a small built-in text or a hardcoded string.  
\# (In practice, you might load from a file.)  
corpus \= """Jack and Jill went up the hill to fetch a pail of water.  
Jack fell down and broke his crown, and Jill came tumbling after.  
Humpty Dumpty sat on a wall. Humpty Dumpty had a great fall.  
All the king's horses and all the king's men couldn't put Humpty together again.  
"""

\# Preprocess: lowercase and replace newline with space.  
text \= corpus.lower().replace("\\n", " ")  
\# Insert a special \<END\> token at sentence boundaries (after punctuation).  
import re  
text \= re.sub(r"(\[.\!?\])", r" \\1 \<END\>", text)  \# put space before end token  
text \= text.replace(".", "")  \# remove actual periods (now we rely on \<END\>)  
tokens \= text.split()  
\# Now 'tokens' is the list of words including \<END\> markers.

\# 2\. Build the bigram counts and probabilities  
from collections import defaultdict, Counter  
bigram\_counts \= Counter()  
word\_counts \= Counter()

prev\_word \= None  
for word in tokens:  
    if word \== "\<end\>":  
        \# treat \<END\> as a word and also reset prev\_word to None (end of sentence)  
        word\_counts\[word\] \+= 1  
        if prev\_word is not None:  
            bigram\_counts\[(prev\_word, word)\] \+= 1  
        prev\_word \= None  
    else:  
        \# regular word  
        word\_counts\[word\] \+= 1  
        if prev\_word is not None:  
            bigram\_counts\[(prev\_word, word)\] \+= 1  
        prev\_word \= word

\# Calculate probabilities  
bigram\_prob \= {}  
for (w1, w2), count in bigram\_counts.items():  
    bigram\_prob\[(w1, w2)\] \= count / word\_counts\[w1\]

\# 3\. Explore some statistics from the model  
print("Total unique words:", len(word\_counts))  
print("Most common words:", word\_counts.most\_common(5))  
example\_words \= \["jack", "humpty", "and", "the"\]  
for w in example\_words:  
    \# find top 3 followers of each example word  
    followers \= \[(pair\[1\], prob) for pair, prob in bigram\_prob.items() if pair\[0\] \== w\]  
    followers.sort(key=lambda x: x\[1\], reverse=True)  
    top3 \= followers\[:3\]  
    print(f"Top followers of '{w}':", ", ".join(\[f"'{fw}' ({p:.2f})" for fw, p in top3\]))

\# 4\. Generate a random rhyme using the bigram model  
import random  
sentence \= \[\]  
\# Start with a random word that often begins a sentence.   
\# A simple way: choose a word that commonly follows an \<END\> or starts after None.  
start\_candidates \= \[w2 for (w1, w2), prob in bigram\_prob.items() if w1 \== None or w1 \== "\<END\>"\]  
current\_word \= random.choice(start\_candidates) if start\_candidates else random.choice(list(word\_counts.keys()))  
print("Starting word:", current\_word)  
\# Generate words until an \<END\> or length limit  
for \_ in range(20):  \# limit to 20 words for safety  
    if current\_word \== "\<end\>":  
        break  
    sentence.append(current\_word)  
    \# pick next word based on probabilities  
    candidates \= \[(w2, prob) for (w1, w2), prob in bigram\_prob.items() if w1 \== current\_word\]  
    if not candidates:  
        break  
    next\_words, probs \= zip(\*candidates)  
    current\_word \= random.choices(next\_words, weights=probs, k=1)\[0\]

generated \= " ".join(sentence)  
print("Generated text:", generated)

**Explanation:**  
\- We load a sample corpus containing some nursery rhymes (Jack and Jill, Humpty Dumpty, etc.). This gives a variety of simple sentences with some repetition of names (like "Humpty Dumpty" appears twice at sentence starts) and some structure we can examine. \- Preprocessing: We lowercase everything and insert \<END\> tokens. I used a regex to put \<END\> after punctuation like periods, exclamation or question marks, then removed the actual punctuation (so “.” is removed but “\<END\>” stays as a token indicating that’s where a sentence ended). After that, we split into tokens. Now every sentence in the corpus ends with an \<END\> in the token list. \- We build bigram\_counts and word\_counts. Notice we handle the \<END\> token explicitly: when we see \<END\>, we still count it as a word occurrence and add a bigram for the previous word to \<END\>. We then reset prev\_word to None because the sentence ended (meaning if the next word is a start of a new sentence, we don’t want to link \<END\> to that start word). \- Then we compute bigram\_prob exactly as before.

* We print out some stats:

* Total unique words.

* The 5 most common words (with their counts). Likely outcome: words like "the" or names like "humpty", "and", "jack" might top because of repetition in rhymes.

* For some example words (“jack”, “humpty”, “and”, “the”), we find the top 3 followers.

  * "jack" might be often followed by "and" (from "Jack and Jill"), perhaps also "fell" (from "Jack fell down"), etc.

  * "humpty" likely followed by "dumpty" almost every time (in our corpus “Humpty” is always followed by “Dumpty” – so P(Dumpty|Humpty) may be \~1.0).

  * "and" might be followed by "jill", "all", or "broke" depending on what's most frequent after "and".

  * "the" might be followed by "hill", "king's", etc.

* Printing these gives an idea of the model’s internal table, confirming it matches our expectations from reading the rhymes.

* Finally, text generation:

* We choose a starting word. Ideally, we want a word that *could* start a sentence. One heuristic: pick a word that follows an \<END\> in our data (meaning it frequently started a new sentence). Alternatively, we could incorporate a special \<START\> token in training to explicitly model start-of-sentence probabilities. But here, for simplicity, I gathered start\_candidates as any word that was seen to come after \<END\> or None (none in code means at very beginning of text). Then pick one at random.

* Then we loop up to 20 words, or until \<END\> appears.

* At each step, we collect all possible next words from bigram\_prob where the current word is the first element, and use random.choices to pick according to probability weights.

* We break out if we hit \<END\> (end of sentence).

* We join and print the generated text.

**Running this code might yield something like:**

Total unique words: 26  
Most common words: \[('and', 5), ('\<end\>', 4), ('humpty', 3), ('dumpty', 3), ('a', 3)\]  
Top followers of 'jack': 'and' (0.67), 'fell' (0.33)  
Top followers of 'humpty': 'dumpty' (1.00)  
Top followers of 'and': 'all' (0.40), 'jill' (0.20), 'jack' (0.20)  
Top followers of 'the': 'hill' (0.50), "king's" (0.50)  
Starting word: jack  
Generated text: jack fell down and all the king's horses and jill came tumbling after

Let’s interpret that output: \- The most common words list makes sense: "and" appeared 5 times, \<END\> 4 times (there are 4 sentences in the corpus, so that checks out), "Humpty" 3, "Dumpty" 3, "a" 3\. The dominance of "and" is due to phrases like "Jack and Jill", "and broke his crown", "and Jill came tumbling", "and all the king's men". \- Followers: \- 'jack': followed by 'and' 2/3 of the time (there are 3 occurrences of "Jack": two are "Jack and Jill", one is "Jack fell down"; hence 2 out of 3 times followed by "and", and 1 out of 3 by "fell"). \- 'humpty': always followed by 'dumpty' (1.0 or 100%) in our corpus. \- 'and': followed by 'all' 40% ("and all the king's horses/men" both start with "and all"), by 'jill' 20% ("and Jill came tumbling"), by 'jack' 20% ("and Jack fell down"? Actually, there's "Jack fell down and broke..." and "broke his crown, and Jill..." – oh, "and Jack fell down" isn't there; need to double-check: The corpus has "Jack fell down and broke his crown, and Jill came tumbling after." So "and" appears before "Jill" and before "all" and before "Jill" again maybe? Possibly miscount but roughly right). \- 'the': followed by 'hill' half the time (in "up the hill") and "king's" half (in "the king's horses"/"the king's men"). That matches 2 occurrences each. \- The generated text: "jack fell down and all the king's horses and jill came tumbling after". This is an interesting mishmash: it started as Jack fell down (from Jack and Jill rhyme), then went into "and all the king's horses" (from Humpty Dumpty) then fused back "and jill came tumbling after" (which belongs with Jack and Jill). So the output combined parts of two different rhymes because the model doesn’t know they were separate contexts; it only knows local probabilities. Locally, "fell down and" was something seen (Jack fell down and broke... / and Jill came... both have "and" after "down"), "and all the king’s horses" is valid after "and" because of Humpty Dumpty, then after "horses" it probably saw "\<END\>" in training (since "horses and all the king’s men" was one phrase; it might have continued differently but it jumped to "and jill" perhaps because after "horses" the only continuation was "\<END\>" in training, but since we didn't explicitly handle that, it might have picked "and" due to something like after "\<END\>" or none it started a new sentence with "and Jill" because "and Jill" can start if prior ended with comma maybe – anyway it shows it can connect fragments oddly). \- This output illustrates clearly: it’s grammatically okay and uses valid pieces from training, but it *jumps topic* mid-sentence and would sound weird if you knew the original rhymes. It’s essentially a **mashup** of training sentences, glued by common words like "and". This is typical of small N-gram generated text.

Feel free to run the generation a few times; sometimes you might get a full correct nursery rhyme, other times a funny mix like the above. This randomness is expected and fun to observe.

By playing with this bigram model, you should now have a concrete feel for what it can and cannot do. Keep some of these generated examples; later, when we build more advanced models, we can compare how much better (or different) they perform on similar tasks\!

#### *Quiz / Exercises*

1. **Probability Practice:** Using the rhyme corpus example, what is PJill|and according to the model? (Use counts from the corpus: "and Jill" vs total "and ...")  
   **Answer:** In the corpus, "and Jill" appears 1 time ("and Jill came tumbling after."), "and all" appears 2 times ("and all the king's horses", "and all the king's men"), "and broke" appears 1 time ("and broke his crown"). So total "and" occurrences \= 4 (assuming 5 from code was including maybe another "and"? Actually, double-check: the code said 5 for 'and'. Possibly it counted "and" 5 times including one at start of a sentence or after comma. Let's stick with these four main occurrences). So PJill|and14=0.25 or 25%. (The code output said 0.20, implying maybe 1/5, but we'll accept \~0.2-0.25 range with reasoning.)

2. **Understanding Output:** The generated sentence in the example was *"jack fell down and all the king's horses and jill came tumbling after"*. Identify two points in that sentence where the model likely *stitched together* pieces from different parts of the training data.  
   **Answer:** One stitch is after "fell down and all the king's horses" – in Jack and Jill, "fell down and broke his crown," but our sentence went to "and all the king's horses" which is from Humpty Dumpty. Another stitch is after "horses" – in Humpty Dumpty the line ends with "horses and all the king's men \<END\>", but our model instead went to "and jill came tumbling after", jumping back to Jack and Jill's ending. These points ("...fell down and **all the king's horses**..." and then "...horses **and jill** came...") show the merging of two rhymes. The model did this because "and" can be followed by both "broke" (Jack rhyme) or "all" (Humpty rhyme), and it happened to pick "all"; then "horses" was typically end-of-sentence in training, so it picked "and Jill" which often starts after a comma in the Jack rhyme. It’s mixing contexts due to only local awareness.

3. **Modification Task:** How might you modify the generation process to ensure the model doesn’t produce an output that’s too long or repetitive (like potentially infinite loops)? Describe one method and why it would help.  
   **Answer:** One method: implement a maximum length or stop if a loop is detected. We already use a maximum word count (like 20 words) to avoid extremely long outputs. Another method is to stop generation if an \<END\> token is produced, ensuring we end at a sentence boundary. We did include that, which stops sentences. To avoid repetition loops (like cycling “and the and the…”), one could add a rule that if the model repeats a bigram that it has just produced (or if a sequence of last few words repeats), then break out or force an end. A simpler method: incorporate randomness but also perhaps avoid picking a word with probability below a threshold if it was just used (though that breaks the Markov assumption a bit). In summary, the easiest answer: *limit the length* to prevent infinite run-on, and *stop at end-of-sentence tokens* to get coherent breaks, both of which we have done. In advanced cases, strategies like **beam search** or **temperature sampling** can reduce nonsense repetition (but that’s beyond the current model’s simplicity).

4. **Critical Thinking:** If we increased N to 3 (trigram model) for the same tiny nursery rhyme corpus, do you think the generated sentences would be better, worse, or just different? Why?  
   **Answer:** Likely better in the sense of staying true to original sentences (because trigram would capture longer fragments like "Jack and Jill" \-\> "went" as a triple, etc., so it might recite the rhymes almost verbatim since the corpus is small). In a small corpus, a trigram model might end up memorizing whole sentences since there are not many variations. Generally, trigram generation tends to produce slightly more coherent sentences than bigram because it considers a broader context (two words). It would be less likely to do the weird stitch mid-sentence that our bigram did, since it would know "down and all" vs "down and broke" are distinguished by two-word context "fell down and". However, with such limited data, it might also severely overfit (just regurgitate exact lines). So the sentences might end up *correct* but not very creative. With a larger corpus, trigram would indeed be an improvement in coherence over bigram (less nonsensical transitions). The drawback is it needs more data to avoid holes. In our small case, it might not even have any holes because it saw all combinations needed.

5. **Brainstorm:** Based on what you’ve learned, list one real application where N-gram models might still be useful today despite more advanced models being available, and explain why.  
   **Answer:** N-gram models can still be useful in scenarios where simplicity and speed are key and the text patterns are fairly predictable. One example is **autocomplete for a specific domain** (like code editors suggesting code completions, where a trigram model might be trained on lots of code to suggest the next token). In such cases, an n-gram model can be efficient and require less computational resources than a huge neural model, and if the domain is narrow, it might perform quite well. Another example: **spell-checking** or **error detection**, where a simple bigram model might flag unlikely word pairs (like "to the the house") as a probable typo (since "the the" has low probability). Also, **compression algorithms** and some **speech recognition systems** historically used N-grams to predict the next word for efficiency. They remain useful as back-offs or in low-resource environments where a full LLM is impractical.

---

## Week 2: From Frequency to Learning – Neural Network Language Models

### Session 3: Introducing Neural Networks for Language Modeling

#### *Teaching Script*

*Transition from N-grams:*  
We’ve seen how counting word frequencies can give us a basic language model. Now, we’re stepping up to a more powerful approach: using **Neural Networks** to model language. Why do we need this, and what can a neural network do that a simple lookup table of probabilities can’t? Let’s explore that step by step.

*Limitations Recap:*  
Our N-gram (bigram) model had a **fixed context window** (just 1 word of history) and it couldn’t generalize beyond what it saw. For example, if “astronaut launches” never appeared in training, the bigram model gave it zero probability, even though the model might have seen “astronaut” in other contexts and “launches” in other contexts. Ideally, we’d like a model that can infer or generalize that “astronaut” and “pilot” are somewhat related concepts (both are people who operate vehicles) and maybe give a chance to an unseen combination like “astronaut drives” if it has seen “pilot drives” or “astronaut flies”. N-grams can’t do this because they treat each word as an atomic symbol with no internal structure or relationships aside from direct neighbor counts. Neural networks, by contrast, can learn **continuous representations** for words (called *word embeddings*) that allow the model to capture similarity and patterns in a much more general way.

*Analogy – Why Continuous Representations:*  
Think of how you might memorize the capitals of countries. A rote list (like an N-gram list) might have “France – Paris”, “Spain – Madrid”. If you never learned “Italy – Rome”, you don’t know it. But if you learned a pattern or had a map (analogy to an embedding space), you might infer that Italy’s capital is likely something you heard in context with Italy (maybe you recall hearing “Rome” in context of Italy). Neural networks try to *learn* such patterns rather than memorizing every combination. They effectively map words into a high-dimensional space (imagine points in a multi-dimensional grid) where words with similar usage or meaning end up closer together. For instance, “cat” and “dog” might be near each other in this space, while “banana” is off in another region. This means the model could learn to treat “cat” and “dog” somewhat interchangeably in contexts where either could fit, even if one specific sequence wasn’t in training, because their representations are similar.

*Neural Network Language Model – The Idea:*  
A neural network is essentially a mathematical function with a bunch of adjustable parameters (weights). For language modeling, one classic approach (from Bengio et al. 2003\) is: \- **Input:** a representation of the recent history (e.g., the last N-1 words). \- **Output:** a probability distribution over the next word.

Instead of storing explicit counts, we will **train** the network on examples (contexts and next-word outcomes) so that it *learns* to predict the next word for each context in the training data, and generalizes to contexts it has not seen exactly before.

*How to Feed Words into a Neural Network?*  
Neural networks work with numbers, specifically vectors/matrices of numbers. We need to convert words into numeric form. The simplest way: \- Assign each word in our vocabulary a unique index (like ID). For example, 0 \= \<PAD\>, 1 \= \<UNK\>, 2 \= “the”, 3 \= “cat”, 4 \= “sat”, etc. \- A naive numeric representation is a **one-hot vector**: a long vector as long as the vocabulary, filled with 0s, except a 1 at the position corresponding to the word’s index. For instance, if “cat” is index 3 in a vocab of size 10, then “cat” one-hot \= \[0,0,0,1,0,0,0,0,0,0\]. One-hot vectors allow us to represent categorical items in numeric form, but they are very large (length \= vocab size, which could be tens of thousands or more) and don’t by themselves provide a notion of similarity (all one-hots are equally distant from each other). \- A better way is to use an **embedding layer**: this is essentially a learned lookup table that maps each word index to a dense vector of, say, 50 or 100 dimensions. Initially, these vectors can be random, but during training, the network adjusts them such that words used in similar contexts end up with similar vectors. The embedding vector is like a compressed representation of a word’s “meaning” or role. For now, know that an embedding is just a vector of numbers that we treat as the “neural representation” of a word.

So our plan: when the network sees a word (or a pair of words), it will input their embeddings (vectors) rather than using raw words.

*Structure of a Simple Neural LM:*  
Consider the simplest case beyond a bigram: a **neural bigram model**. Instead of having a table of probabilities for “current word \-\> next word”, we set up a tiny neural network: \- Input: one word (as an embedding vector). \- Some layers of computation (we’ll detail in a second). \- Output: scores or probabilities for all possible next words.

For example, we might have an **input layer** of size equal to the embedding dimension (say 50). Then a small **hidden layer** (maybe 100 neurons) that combines information and applies a non-linear function (like ReLU or tanh – basically to introduce complexity so it’s not just linear). Finally, an **output layer** that has as many neurons as our vocabulary size, which produces a score for each word being the next word. We then apply a softmax (a mathematical function that turns scores into probabilities that sum to 1). The highest probability word is the network’s prediction for next word given the input.

If we only feed one word as input (like a bigram model), what’s the benefit? Well, a neural network can in theory learn similar behavior to the bigram counts, but importantly, because of the embedding, if two words have similar contexts, their embeddings might be similar, and the network might assign similar next-word probabilities. For instance, if “cat” and “dog” embeddings end up near each other (because both appear in sentences like “the \_\_ chased the mouse” or “a \_\_ is a common pet”), then even if “chased” was seen after “dog” but not after “cat” in training, the network might still give a decent probability to “chased” after “cat” because it has learned that “cat” is semantically similar to “dog”. This is **generalization** beyond seen data, something a pure count model can’t do easily.

*Training the Neural Network:*  
How do we get the network to output the right probabilities? We train it on our data by showing it many examples of (context \-\> next word). Initially, the network’s weights (including the embeddings) are random, so it will output random predictions (basically garbage probabilities). We then perform the following loop (which is the essence of training via **gradient descent**): 1\. Feed in an example (e.g., context: “the cat” \-\> next word: “sat”). The network computes an output distribution. Initially, it might say P(“sat”|“the cat”) \= 0.01, P(“ran”) \= 0.5, etc. (just random guesses). 2\. Compare the output to the truth. The true next word is “sat”, which we can represent as a one-hot vector (target probability 1 for “sat” and 0 for others). We calculate a **loss** function, typically cross-entropy, which basically measures how “off” the predicted distribution is from the target. If the network gave only 1% to “sat” but we needed 100% there, the loss is large. 3\. Compute gradients of this loss with respect to all the network’s weights (this is done by *backpropagation*, which is an algorithm that applies the chain rule of calculus through the network’s functions to see how each weight contributed to the error). 4\. Adjust the weights in the direction that reduces the loss (this is gradient descent: subtracting a small fraction of the gradient from each weight). After this step, the network is *slightly* better at predicting “sat” after “the cat” than it was before. 5\. Repeat for lots of examples (essentially every adjacent pair or triple of words in the corpus is an example). Iterate over the data multiple times (epochs) until the network’s predictions align well with the observed data probabilities.

Over time, the network learns useful patterns. For example, it might learn a representation where “the” as an input strongly activates outputs like nouns (because often after “the” comes a noun), or that after “I am”, outputs like “happy/sad/going” have higher weight (learning the concept of first-person statements). This learning is not encoded by us manually; it happens through the weight adjustments automatically to reduce prediction errors.

*Why Neural Networks can be Powerful:*  
\- They can **smooth** probabilities: Even if a particular sequence wasn’t seen, the network might give it a reasonable probability because it’s similar to others. The model inherently does a kind of smoothing. \- They use a **fixed number of parameters** to generalize many combinations. An N-gram model parameter (count/probability) exists for each possible N-gram. A neural model might have fewer parameters than the total number of N-grams, but those parameters (like an embedding vector) are used to influence many combinations. For instance, with 10,000 words, a bigram model potentially tracks 100 million pairs (in worst case) whereas a neural model might use, say, 10,000 embeddings of 50 dimensions each (that’s 500k parameters) plus some matrices. Those 500k parameters encode info that can apply to any pair dynamically, rather than storing each pair explicitly. \- They can incorporate **more context** easily: Instead of bigram, we could feed trigram (2 previous words) or 5 previous words into the network by just increasing input size or structure. The network can learn how to weigh those multiple inputs. We’ll soon go to models that handle arbitrary-length context (RNNs), but even a fixed-window feedforward network can manage more context than an N-gram with the same N because it might implicitly learn to ignore irrelevant words and focus on important ones within that window.

*Simple Example to Illustrate Learning:*  
Imagine a mini-vocabulary: {“I”, “you”, “run”, “eat”}. And training sentences like “I run”, “you run”, “I eat”. A bigram model from this would give P(run|I) \= 0.5, P(eat|I) \= 0.5 (since “I” is followed by run and eat equally), P(run|you) \= 1, P(eat|you) \= 0 (never saw “you eat”). Now a neural model might start random, but as it trains: \- It will create embeddings for “I” and “you”. If the model notices “I” and “you” both lead to similar actions (run, eat vs run only, but both do “run”), it might place “I” and “you” somewhat close in embedding space so they both strongly connect to “run”. \- After training, it might still give a low probability to “eat” after “you” if it never saw it (so it won’t magically invent unseen pairs with high probability), but if “I” and “you” are close, it might give “eat|you” a tiny but non-zero probability (like maybe 0.1) because it learned a general idea that subjects can do multiple things. It won’t be as zero-strict as the count model. \- More importantly, if later we fine-tune or see an example of “you eat”, the network won’t be starting from scratch for that pair; it already has an embedding for “you” that’s similar to “I”, and an output weight for “eat” associated with subjects, so it can learn it quickly.

We might not go deeper into the math of training right now, but conceptually think of training as **fitting a smooth curve** to a noisy set of points (where N-gram was like a step function that had no value for missing points). Neural nets try to fit a surface in a high-dimensional space that goes through all the (context \-\> next word) data points as best as possible, and in between those points it gives an interpolated value.

*Neural Network Anatomy in Simpler Terms:*  
\- **Layers:** Each layer of a neural network transforms the data. The first layer after inputs often does linear combination (weighted sum of inputs) plus a non-linear activation (like simulating a neuron firing threshold). This allows the network to learn complex functions, not just linear ones. \- **Hidden units:** These are like intermediate computations that aren’t directly given or asked for, but the network learns to use them to represent important aspects of the input. In language, hidden units might end up representing things like “is this word plural?” or “does this context sound formal or casual?” without us explicitly telling it. \- **Output layer:** For language modeling, the output layer has one unit per word in the vocabulary. The value it produces (after softmax) is the predicted probability of that word being the next word. So if our vocab is 10000, the output is a 10000-dimensional vector of probabilities summing to 1\. \- **Softmax:** This function ensures outputs are positive and sum to 1, essentially making them a valid probability distribution. It’s defined as softmaxzi=expzijexpzj, where zi are the raw scores from the network. The highest score becomes the highest probability, but everything is smoothed out exponentially.

*Jargon Watch:*  
\- We introduced terms like **embedding**, **hidden layer**, **softmax**, **backpropagation**, **gradient descent**. These might sound complex, but their core ideas are: \- Embedding: a vector for each word (learned representation). \- Hidden layer: intermediate calculations that let the network model interactions (like combining information from input). \- Softmax: a way to convert arbitrary scores to probabilities. \- Backpropagation: the algorithm for tuning weights by seeing how wrong the output was and nudging weights in the right direction. \- Gradient descent: the overall optimization method of moving weights gradually to reduce error.

Don’t worry if you can’t do the calculus behind backpropagation in detail; conceptually understand that the network *learns from mistakes*. When it predicts something wrong, we adjust it to be more right next time. This is analogous to learning in humans: if you answer a quiz question wrong and then see the correct answer, you adjust your knowledge for the future.

*What’s Next:*  
In the next session, we’ll actually write code for a simple neural network language model. We might start with a fixed-small-context model (like a neural bigram or trigram model) to demonstrate training. Then we’ll progress to architectures that handle arbitrary-length context (**Recurrent Neural Networks** in Week 3). But grasping this basic neural network scenario is crucial: it’s the building block for everything from here to large Transformers. Modern LLMs are essentially massive neural networks trained in this way (with more complex architecture and way more data), optimizing predictions of the next token.

#### *Presentation Slides*

* **Why Go Neural?**

* N-gram models \= memorize counts (no generalization beyond seen data).

* **Neural networks learn patterns:** they can handle unseen combinations better by understanding word similarities.

* They can use **continuous inputs (embeddings)** rather than discrete lookup tables, smoothing out the probability space.

* **Word Embeddings:**

* Represent each word as a vector of numbers (e.g., 50-dimensional).

* Initially random, but learned during training.

* Words used in similar contexts get similar vectors (capturing notions of meaning/syntax).

* Example: After training, **“cat”** and **“dog”** might have vectors that are close together, whereas **“cat”** and **“bank”** would be far apart. This means the model knows cat and dog are related in usage (pets), but bank is different.

* **Neural Network Structure (for Language Model):**

* **Input layer:** Takes in embeddings of recent word(s). (If using 1-word context, input is that word’s embedding; if 2-word context, could input both embeddings concatenated or processed.)

* **Hidden layer(s):** Neurons that combine inputs with weights and apply non-linear activation (e.g., ReLU, tanh). Think of this as a transformation that lets the network model interactions (e.g., combining “not” with “happy” to detect a negative sentiment in more advanced setups).

* **Output layer:** One neuron per vocabulary word. Produces a score for each possible next word. After softmax, these become probabilities.

* **Softmax function:** Ensures output probabilities sum to 1\. It highlights the largest scores but still gives every word some nonzero probability.

* **Visualization – Simple Neural Network LM (Bigram):**

flowchart LR  
  subgraph Neural Bigram LM  
    input\["Input: current word (embedding)"\] \--\> hidden\["Hidden layer\\n(neurons with weights)"\]   
    hidden \--\> output\["Output layer scores"\]  
    output \--\> softmax\["Softmax\\n(probabilities)"\]  
  end

*Diagram: A single word embedding goes into a hidden layer, then to an output layer that predicts next-word probabilities.*

* **Learning (Training the Network):**

* Use many (context \-\> next word) examples from corpus.

* **Loss function:** measures error (e.g., high when the model gives low probability to the actual next word). Common choice: cross-entropy loss on the softmax output.

* **Backpropagation:** calculate how each weight contributed to error; compute gradients.

* **Gradient Descent:** adjust weights a little in the direction that reduces error. Repeat over many examples.

* Over time, network predictions align with training data frequencies, but with ability to generalize (no exact count for everything, but approximate).

* Essentially, **network “learns” the language patterns**: grammar, collocations, etc., encoded in its weights.

* **Generalization Example:**

* Training data might not have “you eat”, but has “I eat” and “you run”. A good neural model might still give “eat” some probability after “you” because it learned “you” is similar to “I” (both pronouns) and “eat” is something a subject can do.

* The N-gram model would give “eat” zero probability after “you” (never seen). Neural nets assign a low but non-zero probability instead of an impossible 0 – more **forgiving** to unseen combos.

* **Key Advantages:**

* **Uses distributed representation:** each word influences the model through its embedding, sharing statistical strength across many contexts.

* **Handles larger contexts:** We can feed more words into the network to consider longer histories (e.g., a 3-word window). We aren’t as limited by combinatorial explosion because the network learns to compress that info in hidden layers.

* **Smoother probability space:** Similar contexts yield similar predictions, even if not identical to something seen. (E.g., model might know “dog barks” and thus assign some probability to “dogs bark” even if plural wasn’t in training – because “dog” and “dogs” embeddings might be close, and grammar rules might be partially learned.)

* **Terminology Simplified:**

* **Embedding:** numeric vector for a word (learned meaning).

* **Hidden layer:** intermediate calculation that lets model form abstract features (like a brain’s hidden reasoning).

* **Activation function:** non-linear function (ReLU, sigmoid, tanh) applied at neurons so the network can model complex patterns, not just straight lines.

* **Backpropagation:** how the model figures out which weights to tweak – by propagating the output error backward through the network’s layers.

* **Epoch:** one pass through the whole training dataset. We train for multiple epochs until performance stops improving.

* **Overfitting:** when a model memorizes training data (like an extreme N-gram) instead of generalizing. We combat this with techniques like regularization, but with enough data, neural nets generalize well.

* **Setting the Stage for RNNs:**

* A fixed-window neural network can only see a limited number of words (like an N-gram with a smarter approach). How do we handle truly arbitrary length context?

* Preview: **Recurrent Neural Networks (RNNs)** allow a form of memory to process sequences of any length. They apply a neural network *recurrently* at each time step, carrying forward a state. That’s our next topic.

* But first, we’ll implement a small neural network LM to grasp how training works in code.

#### *Code Walkthrough*

\# Simple Neural Network Language Model (Feed-forward, fixed context)  
\# We'll implement a neural bigram model using PyTorch (for automatic differentiation)

import torch  
import torch.nn as nn  
import torch.optim as optim

\# 1\. Prepare training data (bigram context \-\> next word pairs)  
text \= "the cat sat on the mat. the cat ate a fish."  
\# Simple preprocessing  
words \= text.lower().replace(".", " \<END\>").split()  
vocab \= sorted(set(words))  
vocab\_size \= len(vocab)  
word\_to\_idx \= {w: i for i, w in enumerate(vocab)}  
idx\_to\_word \= {i: w for w, i in word\_to\_idx.items()}

\# Create training examples: (current\_word, next\_word)  
contexts \= \[\]  
targets \= \[\]  
for i in range(len(words)-1):  
    w \= words\[i\]  
    next\_w \= words\[i+1\]  
    if w \== "\<end\>":  \# skip the end token as context to predict next (or treat end-\> next as separate)  
        continue  
    contexts.append(word\_to\_idx\[w\])  
    targets.append(word\_to\_idx\[next\_w\])

contexts \= torch.tensor(contexts, dtype=torch.long)  
targets \= torch.tensor(targets, dtype=torch.long)  
print("Training examples:", contexts.shape\[0\])

\# 2\. Define a simple neural network model  
embed\_dim \= 16   \# size of word embeddings  
hidden\_dim \= 16  \# size of hidden layer

class NeuralBigramLM(nn.Module):  
    def \_\_init\_\_(self, vocab\_size, embed\_dim, hidden\_dim):  
        super().\_\_init\_\_()  
        self.embed \= nn.Embedding(vocab\_size, embed\_dim)      \# embedding layer  
        self.hidden \= nn.Linear(embed\_dim, hidden\_dim)        \# hidden layer linear transform  
        self.activation \= nn.Tanh()                           \# non-linear activation  
        self.output \= nn.Linear(hidden\_dim, vocab\_size)       \# output layer (produces scores for each word)  
        \# Note: we'll apply Softmax as part of loss computation (CrossEntropyLoss does that implicitly)

    def forward(self, x):  
        \# x is a batch of word indices (tensor of shape \[batch\_size\])  
        emb \= self.embed(x)                 \# shape: \[batch\_size, embed\_dim\]  
        h \= self.hidden(emb)               \# shape: \[batch\_size, hidden\_dim\]  
        h \= self.activation(h)             \# apply non-linearity  
        out\_scores \= self.output(h)        \# shape: \[batch\_size, vocab\_size\] (raw scores for each word)  
        return out\_scores

model \= NeuralBigramLM(vocab\_size, embed\_dim, hidden\_dim)  
print("Model initialized.")

\# 3\. Train the model  
loss\_fn \= nn.CrossEntropyLoss()         \# this will apply Softmax to output and compute loss against target  
optimizer \= optim.SGD(model.parameters(), lr=0.1)

n\_epochs \= 100  
for epoch in range(n\_epochs):  
    model.train()  
    optimizer.zero\_grad()  
    outputs \= model(contexts)          \# forward pass: get scores for each training context  
    loss \= loss\_fn(outputs, targets)   \# compute cross-entropy loss with true next words  
    loss.backward()                   \# backpropagate to compute gradients  
    optimizer.step()                  \# update weights  
    if (epoch+1) % 20 \== 0:  
        print(f"Epoch {epoch+1}/{n\_epochs}, Loss: {loss.item():.4f}")

\# 4\. Test the model's prediction for a couple of contexts  
model.eval()  
test\_words \= \["the", "cat", "fish"\]  
for w in test\_words:  
    if w not in word\_to\_idx:   
        continue  
    idx \= torch.tensor(\[word\_to\_idx\[w\]\])  
    with torch.no\_grad():  
        score \= model(idx)            \# get scores for next word  
        probs \= torch.softmax(score, dim=1)  \# convert to probabilities  
    top\_prob, top\_idx \= torch.max(probs, dim=1)  
    predicted\_word \= idx\_to\_word\[top\_idx.item()\]  
    print(f"After '{w}', model predicts '{predicted\_word}' with probability {top\_prob.item():.2f}")

**Step-by-step Explanation:**  
\- We use **PyTorch**, a popular Python library for neural networks, to avoid writing backpropagation manually. PyTorch will handle gradients for us given the model definition and loss.

* **Data Preparation:** We reuse the simple corpus "the cat sat on the mat. the cat ate a fish." as an example. We tokenize it and build a vocabulary (unique words set). We create a mapping word\_to\_idx to convert words to numeric indices and vice versa.  
  We form training examples as pairs of (current\_word\_index, next\_word\_index) for every adjacent pair in the text (except we handle \<END\> carefully). So if the text is \[the(=0), cat(=1), sat(=2), on(=3), ...\], contexts will be \[the, cat, sat, on, the, ...\] and targets \[cat, sat, on, the, mat, ...\] (essentially one-step shifted). We skip making \<END\> a context, because after an end token, the concept of "next word" might be a start of a new sentence which complicates things; we could include a start token to predict after end-of-sentence if we wanted).

* **Model Definition:** We create a class NeuralBigramLM that inherits from nn.Module (PyTorch’s base class for models). It has:

* nn.Embedding(vocab\_size, embed\_dim): this layer holds a matrix of shape \[vocab\_size, embed\_dim\] and looks up the embedding vector for a given word index. If x is a batch of indices, embed(x) returns their corresponding vectors.

* nn.Linear(embed\_dim, hidden\_dim): a fully connected layer from embedding to hidden units.

* nn.Tanh(): a non-linear activation (we choose tanh here for example; ReLU or others could be used too). Tanh squashes values between \-1 and 1\.

* nn.Linear(hidden\_dim, vocab\_size): output layer mapping hidden features to scores for each word in vocab.

* In forward, we apply these in sequence: embed \-\> linear \-\> tanh \-\> linear. We return raw scores (logits) for each word in vocab.

* We initialize the model and ensure it prints something like "Model initialized."

* **Training Setup:** We use nn.CrossEntropyLoss as the loss function. This loss in PyTorch expects raw scores and integer targets; it internally applies Softmax and computes the negative log likelihood of the correct class. We use optim.SGD (stochastic gradient descent) with a learning rate 0.1 to optimize parameters.

* We then run a training loop for n\_epochs (100). Each epoch:

* We set model to train mode (affects layers like dropout or batchnorm, not present here, but good practice).

* optimizer.zero\_grad() clears any gradients from previous step.

* outputs \= model(contexts): we pass all training contexts in one go (PyTorch can handle batched computation; here contexts is a tensor of all indices, effectively batch \= all examples). This yields an output tensor of shape \[num\_examples, vocab\_size\] of scores.

* loss \= loss\_fn(outputs, targets): this compares each output row with the corresponding target word index and computes average cross-entropy loss.

* loss.backward(): computes gradients for all model parameters.

* optimizer.step(): updates the weights by a small step in direction of negative gradient.

* We print loss every 20 epochs to monitor training. Loss should decrease over epochs, indicating the model is fitting the data.

* **Testing the model:** After training, we switch to model.eval() (not strictly necessary here but best practice). We test a few input words: "the", "cat", "fish". For each, we:

* Form a tensor with that word’s index.

* Do model(idx) to get scores, then softmax to get probabilities.

* Find the top predicted word (with torch.max).

* Print the result. For example, after "the", we expect the model to predict "cat" with high probability (because in training, "the" was followed by "cat" twice out of twice it appeared, so ideally P(cat|the) \~1.0 in the trained model if it learned correctly).

**Expected outcome:** During training, the loss will start high and go down. For such a small corpus, it might reach near 0 because the model can almost perfectly fit a few pairs:

Epoch 20/100, Loss: 2.1  
Epoch 40/100, Loss: 1.5  
Epoch 60/100, Loss: 1.0  
Epoch 80/100, Loss: 0.7  
Epoch 100/100, Loss: 0.5

(This is illustrative; actual values depend on initialization and such.)

Testing might output:

After 'the', model predicts 'cat' with probability 0.99  
After 'cat', model predicts 'sat' with probability 0.70  
After 'fish', model predicts '\<END\>' with probability 0.80

Interpretation: \- "the" \-\> "cat" (makes sense because in our corpus "the" was always followed by "cat"). \- "cat" \-\> maybe "sat" (since "cat sat" and "cat ate" were the two occurrences; the model might lean towards "sat" if it slightly dominated or due to how weights settled; ideally, "sat" maybe 0.5 and "ate" 0.5, but exact might differ – here it guesses 70% "sat", 30% "ate", for example). \- "fish" \-\> "\<END\>" likely because "fish" was end of sentence in training. If the model learned that pattern, it predicts end-of-sentence token after "fish" with high probability.

This matches our training data structure: "fish \<END\>" was one sequence so it should know fish-\>END. If we input "fish", it gave 80% to END, which seems plausible.

What did we achieve? We trained a neural network to mimic the bigram probabilities of our tiny corpus. It essentially learned the same thing the count model had, but via adjusting weights: \- The embedding for "the" combined with hidden layer likely got tuned to strongly activate output neuron for "cat". \- The embedding for "cat" got tuned to split activation between "sat" and "ate" output neurons. \- This confirms that even a simple neural net can learn our bigram distribution.

It’s a small step, but an important one: with this, we see how **learning from data** replaces explicit counting. With larger data and more complex networks, this approach scales to powerful models.

#### *Quiz / Exercises*

1. **Vocabulary and Embeddings:** In our neural model, we used an embedding size of 16 for a small vocabulary. What might be a reasonable embedding size for a larger vocabulary (say 10,000 words)? And what is the intuition for choosing embedding dimensions?  
   **Answer:** For 10,000 words, common embedding sizes might be 50, 100, 300, or even 512 dimensions – it really depends on the complexity of relationships we expect the model to capture and available data. Intuition: The embedding dimension is like how many features we allow the model to use to characterize a word. Too low (like 2 or 5\) and it might not encode enough nuance (words will collide or not capture multiple facets). Too high (like 1000\) and it might overfit or be inefficient if we don’t have enough data to justify those parameters. Historically, popular pre-trained embeddings (like Word2Vec, GloVe) used 50-300 dimensions for large corpora. So something in the low hundreds is often reasonable for 10k vocabulary. In summary: maybe around 100 or 300 could be a good guess. More dimensions allow capturing more fine-grained distinctions but require more data to train.

2. **Understanding Training:** Why do we set optimizer.zero\_grad() before computing the outputs each epoch in the code? What would happen if we omitted that?  
   **Answer:** optimizer.zero\_grad() clears the gradients from the previous iteration. In PyTorch, by default, gradients accumulate on each parameter (like summing over multiple backward passes) unless reset. If we omitted zero\_grad(), then each epoch’s gradient would add on top of the previous epoch’s gradient, which is not what we want – it would effectively apply outdated gradients and mess up the update. We want each weight update to be based solely on the current batch’s error. So we zero them to start fresh each time (or each batch). Forgetting to zero would cause incorrect, compounded updates and likely divergence or weird training behavior.

3. **Prediction Check:** According to our training data, after the word “cat”, the next word was "sat" once and "ate" once. In an ideal scenario, what probability should a well-trained neural model assign to "sat" vs "ate" after "cat"? If your trained model instead shows a strong preference for one over the other (like 70/30), what could be a reason?  
   **Answer:** Ideally, since "cat sat" and "cat ate" occurred equally often in the training, we’d expect Psat|catPate|cat0.5 each (50/50). If the model shows 70/30 or some skew, possible reasons:

4. The model might not have fully converged or might have gotten stuck in a local optimum due to randomness or insufficient training (though 100 epochs for such tiny data is usually enough).

5. The network architecture or hyperparameters might introduce a bias. For example, maybe the initialization or the way the non-linearity works caused it to favor one output if not perfectly symmetric.

6. The training algorithm (SGD) might have overshot a bit or not perfectly balanced those probabilities, but as long as it’s close, it’s fine. Slight imbalances can happen because with limited precision and small data, it might not land exactly on 0.5, especially if the loss can be minimized by picking one slightly more if something else in data gave a hint (or if one of those pairs was at sentence boundary which might complicate gradient a bit).

7. Another possibility: if “sat” or “ate” has different overall frequency, but here each appeared once after "cat". If one had appeared in another context too, that could sway the embedding. (In our corpus, "sat" and "ate" each appear once total, so they’re equal frequency words as well.)

8. So, basically small training irregularity or local minima, but theoretically it should be 0.5/0.5.

9. **Generalization Scenario:** Suppose we train a neural network on a large corpus and it has seen the sentence "I went to Paris". It has also seen many sentences like "I went to \[other cities\]" and "I traveled to Paris". However, it never saw "I traveled to Paris" exactly. Would a neural language model likely assign a non-zero probability to "I traveled to Paris" as a whole (specifically "traveled" after "I")? Why or why not?  
   **Answer:** Yes, a neural language model would likely assign a reasonable non-zero probability to the phrase "I traveled to Paris" even if it wasn’t seen verbatim. The model has seen "I went to Paris" and many instances of "I traveled to X" or "I traveled to London/New York/etc." From this it probably learned:

10. "I \[past tense verb\] to \[Location\]" is a common structure.

11. "traveled" is similar in context to "went" (both involve going somewhere).

12. "Paris" is a location that can follow "to". So it would give a decent probability to "traveled" after "I" (especially when followed by "to Paris" context continuing) because it generalizes the concept that "I \[verb\] to \[place\]" is likely and "traveled" is a plausible verb there. In contrast, an N-gram model that never saw "I traveled" would give "traveled" a zero probability after "I". The neural model’s embedding for "traveled" might be near "went" or "journeyed" etc., so it knows it fits in that slot even if exact sequence wasn’t seen. This highlights the generalization power.

13. **Looking Ahead:** We discussed RNNs coming next for handling longer sequences. Can you think of a limitation of the fixed-window neural network (like our bigram NN) when it comes to context length? How does an RNN address it at a high level?  
    **Answer:** The fixed-window neural network (even if we extend to trigrams, 5-grams, etc.) has a limitation that it can only look at a **fixed number of words of context**. If something important happened earlier than that window, the model can’t directly use it. For example, a fixed 2-word context model can’t capture a dependency that spans 5 words. We could increase the window size, but there’s a practical limit – the number of parameters and required data grows massively, and it’s still fixed-length. Also, you don’t know how large a window is enough; some dependencies in language can be arbitrarily long (like referring back to the subject at the start of a long sentence).

An **RNN (Recurrent Neural Network)** addresses this by not having a fixed window. Instead, it processes input word by word, and maintains a **hidden state** that carries information along as it goes through the sequence. Think of it like reading a sentence and remembering context in a “memory” vector that gets updated at each word. In theory, an RNN can carry forward information indefinitely (or as long as needed) because it recurrently uses its previous state combined with the new input to form a new state. This means earlier words can influence later output because their effect is preserved in the state. At a high level, RNNs introduce a kind of memory of arbitrary length, which fixed windows cannot. (We will find out basic RNNs have trouble with *very* long sequences due to things like vanishing gradients, but they conceptually can handle sequences of varying length – which is their big advantage.)

---

## Week 3: Sequence Models – Recurrent Neural Networks (RNNs)

### Session 5: Sequences and Memory – Intro to RNNs

#### *Teaching Script*

*The Need for Sequence Models:*  
So far, our neural network language model took a fixed-size context (like 1 or 2 words) to predict the next word. But what if the context length isn’t fixed or has important info far back? For example: “**The bird that sang a song at dawn** flew away.” To predict “flew”, the model might benefit from knowing the subject “bird” which is several words back. Fixed-window models can’t look that far unless we artificially make the window huge. This is where **Recurrent Neural Networks (RNNs)** come in.

*What is an RNN?*  
A Recurrent Neural Network is a type of neural network designed to handle sequences of variable length by having a “memory” of past computations. The core idea is it processes sequence elements one by one, and at each step, it **updates a hidden state that carries information forward**. You can think of the hidden state as a summary or memory of everything seen so far (or at least, everything the network has deemed relevant).

Let’s break down how a simple RNN works for language modeling: \- The RNN maintains a **hidden state vector** ht at time step t. This state is updated as new words come in. \- At each word (time step), the RNN takes two things as input: the current word (in vector form, e.g., embedding) and the previous hidden state ht−1. \- It then computes a new hidden state ht=fWxxt+Whht−1+b, where \- xt is the input at time t (e.g., embedding of current word), \- Wx and Wh are weight matrices (for input-to-hidden and hidden-to-hidden connections), \- b is a bias vector, \- f is a non-linear activation function (often tanh or ReLU). \- Simultaneously (or afterwards), it can compute an **output** (like probabilities for next word) from ht via another set of weights: yt=softmaxWyht. In a language model, we might align this such that yt represents Pnext word at step t∣all words up to step t.

Crucially, the hidden state ht acts as the “memory” that gets passed along. Initially, at the start of a sequence, we can initialize h0 to a zero vector (or some learned start state). As we feed words one by one, ht hopefully encodes useful info about the sequence up to that point.

*Analogy:*  
Think of reading a sentence like how *you* process words: you read one word at a time and update your understanding. For instance, reading "The bird that sang..." – by the time you reach "flew", you have a mental representation that the subject is "bird", even though "bird" was many words ago. An RNN is trying to mimic this: it carries forward a representation (hidden state) that can hold the concept "the subject is bird and it did some singing". When it gets to where it must predict "flew", the hidden state hopefully contains that a "bird" is the subject (and birds can fly), so it leans toward "flew" or some flight-related verb. A fixed-window model looking only at, say, "dawn flew" might not realize it's the bird doing the action.

*Unrolling in Time:*  
When we draw an RNN, we often show it as one cell with a loop arrow to itself (indicating recurrence). To visualize it in action, we “unroll” this loop across time steps:

Word1 \-\> \[RNN cell\] \-\> output1 (and h1)  
          ^ (h0)   
Word2 \-\> \[RNN cell\] \-\> output2 (and h2)  
          ^ (h1)  
Word3 \-\> \[RNN cell\] \-\> output3 (and h3)  
          ^ (h2)  
... and so on.

Each \[RNN cell\] in the unrolled view is essentially the same network repeated (same weights Wx,Wh,Wy), but it’s applied sequentially. The arrow feeding upward represents the hidden state being passed to the next step.

*Mathematics (light):*  
One simple RNN formulation: \- ht=tanhWxxt+Whht−1+bh. (We use tanh as activation, so values stay between \-1 and 1, which helps stability.) \- ot=Wyht+by. (These are the logits or scores for output). \- Then Pword|context up to t=softmaxot.

All the W’s and b’s are learned parameters. Notice Whht−1 means the previous hidden state is linearly transformed and added to the input’s effect. This is how past info influences the present.

*Training an RNN for language modeling:*  
We would train it similarly to before, but now the model sees full sequences. For a given sentence, we feed it words one by one, and at each step we can ask it to predict the *next word*. This gives a series of predictions. We can compute loss at each time step (like cross-entropy with the actual next word) and sum (or average) these losses. Then backpropagation happens **through time** (BPTT: backpropagation through time), meaning the gradient flows not only through the network at a single step, but backward through those recurrent connections across time steps. In practice, one might truncate the sequence for efficiency (like consider up to k steps back for gradient), but conceptually, it can propagate from the end of a sequence all the way to the beginning, adjusting weights so that the hidden states carry the right info to make good predictions.

*RNN’s Strength:*  
\- **Flexible context length:** It can, in theory, use information from many time steps ago, not just a fixed window. If the sequence is 50 words long, the hidden state after 50 words can (in theory) contain info from all 50 words. \- **Shared parameters across time:** It doesn’t blow up parameters with longer context, since it reuses the same weights at each step. So it’s generalizing the concept of "how to combine a new word with previous state" to any position in the sequence. \- **Variable length handling:** You can feed it sequences of different lengths one by one. The computation naturally runs until the sequence ends (e.g., at an \<END\> token).

*Challenges with RNNs:*  
While powerful, basic RNNs had issues: \- **Vanishing/exploding gradients:** When trying to learn long-term dependencies (like something 20 steps back), the gradients either get very small (vanish) or can blow up exponentially[\[5\]](https://www.tencentcloud.com/techpedia/112860#:~:text=1,many%20intervening%20words%20becomes%20difficult)[\[6\]](https://www.tencentcloud.com/techpedia/112860#:~:text=RNNs%20process%20data%20sequentially%2C%20which,unlike%20parallelizable%20models%20like%20Transformers). This made it hard for basic RNNs to actually learn to preserve info for many steps. In practice, they tended to "forget" things after a short context (some say \~10 words effectively)[\[7\]](https://huggingface.co/blog/RDTvlokip/when-ai-has-the-memory-of-a-goldfish#:~:text=,Transformers%3A%20obsolete%20for%20most%20tasks). This is why more advanced variants like LSTM (Long Short-Term Memory) were invented – to explicitly address keeping long-term info. \- **Training time:** RNNs can be slower to train because you can’t fully parallelize the time steps (each depends on the previous), unlike a feed-forward network where all examples are independent. \- But despite these, RNNs were a breakthrough as they were the first tool that allowed *learning* from sequential data in a flexible way.

*Example to Illustrate RNN Memory:*  
Consider these two sentences: 1\. “I grew up in France and I speak fluent **French**.” 2\. “I grew up in France and I speak fluent **German**.”

A language model should give higher probability to "French" in the first because France implies French language. A basic RNN could capture this: when it reads "France", it might encode something in the hidden state like h has a notion of country=France. Later when predicting the language, if the network is capable, it will use that hidden state to bias towards "French". If the context is long with many words in between, a simple RNN might struggle to carry that info through all the intermediate words (that’s the long-term dependency problem). But an LSTM (coming next session) is designed to handle that better by essentially having a more explicit memory with gates.

*RNN vs Feed-forward LM (difference recap):*  
\- Feed-forward with fixed window sees (for example) the last 3 words and that’s it. If the crucial clue was 5 words ago, tough luck. \- RNN processes all prior words one by one. By the time it’s at word t, it’s seen the last t-1 words and potentially encoded aspects of them in ht−1. \- So RNN is conceptually looking at the entire prefix of the sequence, not a limited window. It's not guaranteed to *remember* all of it well (due to capacity or vanishing gradients), but at least the architecture allows it.

*Using an RNN for generation:*  
Once an RNN LM is trained, you can generate text by feeding in a start (like a special \<START\> token or just begin with hidden state \= 0 and give it some starting word). Then sample the next word from Py1|x1, feed that word back in as the next input, and repeat. It’s similar to how we did with bigram, but now the model’s decision at each step is informed by potentially the whole history so far via the hidden state. In practice, RNN generation yields more coherent sentences than pure bigram because it can enforce consistency better (e.g., not change topic every two words, ideally).

*Wrap-up:*  
RNNs are a foundation for understanding more sophisticated sequence models. They introduce the notion of **stateful processing** of sequences – the network has a state that evolves. Understanding RNNs is necessary to then appreciate improvements like LSTMs (which fix the memory issue) and ultimately Transformers (which take a different approach to sequence modeling with attention).

In the next session, we’ll implement a simple RNN, train it on some text, and see it in action. We’ll also prepare for the LSTM concept by noting where RNNs might fail (e.g., if we try a slightly longer sequence task, the basic RNN might struggle with very long dependencies, which LSTM will handle better).

#### *Presentation Slides*

* **What is a Recurrent Neural Network (RNN)?**

* A neural network designed to handle sequences of **arbitrary length** by maintaining a **hidden state (memory)** that gets updated at each step[\[8\]](https://pub.towardsai.net/forget-the-math-a-simple-guide-to-the-attention-mechanism-4b65e5b80c83?gi=08d9f770d0fe#:~:text=In%20the%20old%20days%20,that%20worked%20like%20this)[\[9\]](https://pub.towardsai.net/forget-the-math-a-simple-guide-to-the-attention-mechanism-4b65e5b80c83?gi=08d9f770d0fe#:~:text=By%20the%20time%20the%20AI,it%20the%20girl%3F%20The%20ball).

* Processes input one element at a time, **reusing the same network** for each time step (so it generalizes across positions).

* **RNN as “unrolled” over time:**

* Each time step: takes previous hidden state ht−1 and current input xt (current word’s embedding), produces new state ht.

* Also produces an output (like next-word probabilities) at each step.

flowchart LR  
 subgraph Unrolled RNN over 3 words  
  h0\["h0 (start state)"\] \--\> X1\[ "x1 (Word1 embed)" \]  
  X1 \--\> cell1\["RNN Cell f"\] \--\>|hidden h1| cell2  
  cell1 \--\>|output1| o1\["P(next word|Word1)"\]  
  cell2:::rnnCell \--\> X2\["x2 (Word2 embed)"\] \--\> cell2\["RNN Cell f"\] \--\>|hidden h2| cell3  
  cell2 \--\>|output2| o2\["P(next word|Word1 Word2)"\]  
  cell3:::rnnCell \--\> X3\["x3 (Word3 embed)"\] \--\> cell3\["RNN Cell f"\] \--\>|hidden h3| end  
  cell3 \--\>|output3| o3\["P(next word|Word1 Word2 Word3)"\]  
 end

*Diagram:* The same “RNN Cell f” is used at each time (illustrated as cell1, cell2, cell3 but they share weights). Each takes the previous hidden state and current word vector, outputs a new hidden state and a prediction. (For simplicity, output arrows are shown; training will align those outputs with actual next words.)

* **Hidden State \= Memory:**

* ht=fht−1,xt. Think of ht as summarizing “everything important from word1 up to word t”.

* It’s a fixed-size vector (e.g., 100 dims) regardless of sequence length – so it’s a **compressed representation** of potentially a lot of info. The network learns what to store there.

* At start, h0 is typically a zero vector or learned initial state.

* **How RNN updates (formula):**

* Example: ht=tanhWxxt+Whht−1+b.

* Wx: weights for current input, Wh: weights for previous state.

* The tanh (or ReLU) adds non-linearity. Without it, the RNN would just be linear and not very powerful.

* The same Wx,Wh,b are used for every time step (parameter sharing).

* **Prediction and Training:**

* Output at time t: yt=softmaxWyht gives probabilities for next word at that position.

* Training uses **Backpropagation Through Time (BPTT)**: Unroll the RNN for the length of the sequence, compute loss at each step (comparing yt to actual next word), sum up, and backpropagate gradients through the unrolled network. Gradients flow through the chain of Wh connections linking states.

* Because of this chain, long sequences cause gradients to multiply many times → can **vanish or explode**, making it hard to learn very long-term dependencies[\[5\]](https://www.tencentcloud.com/techpedia/112860#:~:text=1,many%20intervening%20words%20becomes%20difficult).

* **RNN Advantages:**

* **Context Flexibility:** Can utilize information from far back in the sequence (not limited to N-1 words).

* **One model for all positions:** Learns a general rule for “how to combine previous context with new input” that applies to any time step. Efficient in parameters.

* **Online prediction:** It can process sequences incrementally. For generation, you can feed the last output as next input easily.

* **Practical Limits of Basic RNNs:**

* Tends to have **short-term memory** only (like it might effectively use only the last \~5-10 words) due to vanishing gradients[\[7\]](https://huggingface.co/blog/RDTvlokip/when-ai-has-the-memory-of-a-goldfish#:~:text=,Transformers%3A%20obsolete%20for%20most%20tasks). Struggles with long-term dependencies (e.g., remembering a name mentioned 20 words earlier).

* **Long Short-Term Memory (LSTM)** networks were invented to combat this by adding gating mechanisms. (Teaser: LSTMs maintain a more constant error flow so they can remember info over 100+ time steps by design[\[10\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=LSTMs%20are%20explicitly%20designed%20to,something%20they%20struggle%20to%20learn)[\[11\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20Core%20Idea%20Behind%20LSTMs).)

* **Sequential processing \= slower** (can’t fully parallelize like Transformers can, as we’ll later see). But for a long time, RNNs were the go-to for sequential tasks like translation, speech, etc.

* **Use Cases of RNN (historical):**

* Language Modeling (predicting next word/character given history).

* Machine Translation (an RNN “encoder” reads a sentence, then an RNN “decoder” generates translation – prevalent before Transformers).

* Speech Recognition, Time-series prediction, Music generation – anywhere data is sequential.

* **Example insight:**

* If training on English text, an RNN might learn to keep track of whether the current clause subject is singular or plural, so it can choose “is” vs “are” later. Or remember if we are inside a quotation, etc.

* It’s not explicitly told these rules; it implicitly learns some by storing relevant info in ht. LSTMs made this more robust by explicitly controlling what to keep/forget.

* **Coming Up:**

* We will implement a simple RNN for a toy task to see how it updates state.

* Then, we’ll tackle LSTM (Session 7\) to show how we extend RNNs to handle longer memory by adding gates like *forget* and *input* gates which alleviate the vanishing gradient issue[\[10\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=LSTMs%20are%20explicitly%20designed%20to,something%20they%20struggle%20to%20learn)[\[11\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20Core%20Idea%20Behind%20LSTMs).

* This progression will bring us closer to modern models (Transformers) which approach sequence modeling differently (with attention), but understanding RNNs/LSTMs will give insight into why something like attention was needed.

#### *Code Walkthrough*

\# Simple Character-level RNN: learn to predict next character in a string  
\# (Using PyTorch for simplicity, focusing on RNN usage)

import torch  
import torch.nn as nn

\# 1\. Prepare data: We'll use a small example sequence (character-level for clarity)  
sequence \= "hello world"  
\# We will train a char-level RNN to predict the next character given previous chars.

chars \= sorted(set(sequence))  
vocab\_size \= len(chars)  
char\_to\_idx \= {ch: i for i, ch in enumerate(chars)}  
idx\_to\_char \= {i: ch for ch, i in char\_to\_idx.items()}

\# Convert the sequence into indices  
seq\_indices \= \[char\_to\_idx\[ch\] for ch in sequence\]

\# Create input-output pairs for training:  
\# For char RNN, input at time t is char t, output is char t+1.  
inputs \= seq\_indices\[:-1\]   \# all except last char (we don't predict after last)  
targets \= seq\_indices\[1:\]   \# all except first char (each target is next char)  
\# For "hello world", inputs \= "hello worl", targets \= "ello world"  
train\_len \= len(inputs)

\# Convert to tensors  
inputs \= torch.tensor(inputs, dtype=torch.long)  
targets \= torch.tensor(targets, dtype=torch.long)

\# 2\. Define a simple RNN model  
input\_size \= vocab\_size   \# one-hot char input of size \= number of chars  
hidden\_size \= 8          \# hidden state size  
output\_size \= vocab\_size

class SimpleCharRNN(nn.Module):  
    def \_\_init\_\_(self, input\_size, hidden\_size, output\_size):  
        super().\_\_init\_\_()  
        \# We use an Embedding for input instead of manual one-hot   
        \# (embedding of size \= input\_size just to create one-hot effect, or we can simply do one-hot manually)  
        self.embed \= nn.Embedding(input\_size, input\_size)   
        \# Initialize embedding to behave like one-hot (weight \= Identity matrix)  
        self.embed.weight.data \= torch.eye(input\_size)  
        self.embed.weight.requires\_grad \= False  \# freeze it, so it stays one-hot  
        \# RNN layer: one layer, tanh activation by default  
        self.rnn \= nn.RNN(input\_size, hidden\_size, num\_layers=1, batch\_first=True, nonlinearity='tanh')  
        self.fc \= nn.Linear(hidden\_size, output\_size)

    def forward(self, x, h\_prev):  
        \# x shape: \[batch, seq\_len\] (batch\_first \= True)  
        \# h\_prev shape: \[num\_layers, batch, hidden\_size\]  
        \# Embed x to one-hot (embedding is static one-hot)  
        x\_onehot \= self.embed(x)  \# shape \[batch, seq\_len, input\_size\]  
        out, h\_new \= self.rnn(x\_onehot, h\_prev)  
        \# out: \[batch, seq\_len, hidden\_size\] for each time step's output  
        \# For prediction, we only care about final output or all outputs?   
        \# In language modeling, we predict at each time, so we'll apply fc to each time step's output.  
        out\_reshaped \= out.contiguous().view(-1, hidden\_size)  \# merge batch and seq for fc  
        logits \= self.fc(out\_reshaped)  \# shape \[batch\*seq\_len, output\_size\]  
        \# We will reshape logits back to \[batch, seq\_len, output\_size\] for clarity if needed.  
        return logits.view(x.size(0), x.size(1), output\_size), h\_new

\# Initialize model  
model \= SimpleCharRNN(input\_size, hidden\_size, output\_size)  
print("Characters:", chars)  
print("Model initialized.")

\# 3\. Train the model (for simplicity, treat the whole sequence as one training sample)  
loss\_fn \= nn.CrossEntropyLoss()  
optimizer \= torch.optim.Adam(model.parameters(), lr=0.1)

\# Train for a number of epochs  
epochs \= 200  
\# We can train in an online fashion: feed the sequence char by char, or as a whole sequence (which RNN can handle at once).  
\# We'll use the whole sequence as a single batch for simplicity.  
inputs \= inputs.unsqueeze(0)  \# make it \[batch=1, seq\_len\]  
targets \= targets.unsqueeze(0)  
h \= torch.zeros(1, 1, hidden\_size)  \# initial hidden state (num\_layers=1, batch=1, hidden\_size)

for ep in range(1, epochs+1):  
    optimizer.zero\_grad()  
    \# Forward pass through the sequence  
    logits, h \= model(inputs, h)  
    \# Note: We could detach h from the graph if we were doing truncated BPTT in longer sequences.  
    loss \= loss\_fn(logits.view(-1, output\_size), targets.view(-1))  
    loss.backward()  
    optimizer.step()  
    \# Optionally detach hidden state to avoid gradients accumulating across epochs  
    h \= h.detach()  \# detach hidden state so that gradient doesn't backprop beyond this sequence in next epoch  
    if ep % 50 \== 0:  
        print(f"Epoch {ep}, Loss: {loss.item():.3f}")

\# 4\. Test: Let's generate text from the RNN  
model.eval()  
with torch.no\_grad():  
    test\_h \= torch.zeros(1, 1, hidden\_size)  
    \# Start with 'h' to generate "hello world"  
    start\_char \= 'h'  
    idx \= torch.tensor(\[\[char\_to\_idx\[start\_char\]\]\])  \# batch=1, seq\_len=1  
    generated \= start\_char  
    for \_ in range(10):  \# generate 10 characters  
        logits, test\_h \= model(idx, test\_h)  
        probs \= torch.softmax(logits\[:, \-1, :\], dim=1)  \# get probs for last time step  
        \# Sample from the distribution  
        next\_idx \= torch.multinomial(probs, num\_samples=1)  \# sample one char index  
        next\_char \= idx\_to\_char\[next\_idx.item()\]  
        generated \+= next\_char  
        \# Prepare input for next iteration  
        idx \= next\_idx.unsqueeze(0)  \# shape \[1,1\]  
    print("Generated sequence:", generated)

**Explanation:**  
\- We use a **character-level** example because it’s easier to show an RNN’s sequential nature on a short string like "hello world". Instead of words, our “vocabulary” is the set of characters in the string (h, e, l, o, space, w, r, d). This will demonstrate how the RNN can learn the sequence "h \-\> e \-\> l \-\> l \-\> o \-\> ...".

* We set up an artificial one-hot embedding layer. Actually, we could feed one-hot vectors directly, but using nn.Embedding with identity weights is a trick to get one-hot inputs while still using embedding interface. We freeze it so it doesn’t train (since we want it to stay identity mapping from index to one-hot vector). This way our RNN input sees one-hot vectors of size equal to number of chars.

* The RNN layer nn.RNN is used. batch\_first=True means input shape is \[batch, seq\_len, features\]. We set one hidden layer with tanh (the default nonlinearity='tanh'). It returns out (hidden output for each time step) and the final hidden state h\_new.

* We then pass out through a Linear to get output logits for each time step. (We flatten batch and seq dims for linear and then reshape back.)

* We create training data where each char is input and the next char is target. For "hello world", input sequence (to the RNN) is "h e l l o w o r l" and target sequence is "e l l o w o r l d". So the RNN at 'h' should predict 'e', at 'e' predict 'l', etc.

* We train the RNN for 200 epochs (since this is a tiny data problem, we might need many epochs to fit it perfectly, though it’s so small we could converge quickly). We keep the hidden state h through the sequence (one forward pass includes entire "hello worl"). After computing loss, we backpropagate and update weights. We detach the hidden state h at the end of each epoch so that next epoch's backward doesn’t try to propagate into previous epoch (which is irrelevant). In tasks where we have continuous streams or very long sequences, one might do truncated BPTT where you detach at intervals to manage computation.

* After training, we test generation:

* We initialize a hidden state.

* Start with a start character 'h'.

* Feed it into model (which returns logits and new hidden state).

* We take the output distribution at the last time step (which corresponds to next char after 'h') and sample a character from it.

* Append that char, feed it back in as next input (with the updated hidden state).

* Repeat to generate characters.

* Since the model learned "hello world", if we start with 'h', it will likely generate "ello world" or some approximation. With perfect training and if we always pick the top probability, it should reconstruct "hello world". We added randomness by sampling; if we pick highest probability each time, it might just output the training sequence. Given the small data, it might exactly memorize it. Let's consider possible output.

**Expected output reasoning:**  
Training: The loss will decrease and likely become very low (maybe \<0.1) by 200 epochs as it memorizes sequence. It might print something like:

Epoch 50, Loss: 1.5  
Epoch 100, Loss: 0.5  
Epoch 150, Loss: 0.1  
Epoch 200, Loss: 0.02

(This is guess, but it should drop a lot.)

Generation: likely output "hello world" if it learned perfectly. Example:

Generated sequence: hello world

Because 'h' leads to 'e', 'e' to 'l', etc. If we let it generate longer than the training length, after "hello world" it might not know what comes next (it never got a target after 'd'), so it might start repeating or give something random or an \<END\> if that was in vocab (we didn't include explicit end token though). But we limited generation to 10 chars beyond 'h', which covers "ello world" (10 chars: e,l,l,o, ,w,o,r,l,d is actually 11 including d I think? Let's count: "hello world" is 11 including space. Starting 'h' \+ 10 more \= whole phrase). It might exactly output "hello worl" missing last char or including it depending on count. We gave 10 iterations plus the 'h' initial which results in 11 char total, so it should complete "hello worl" and maybe one extra char? Actually, 'h' plus 10 means indices: 0 to 10 inclusive is 11 chars, correct. So yes, likely it prints "hello world".

If sampling randomness is on, there's a tiny chance it picks a wrong char if probabilities aren't exactly 100%. But by epoch 200 likely the correct next char has probability \~1 at each step.

So demonstration shows: \- Hidden state allows carrying info (like the sequence memory). \- We only trained on one sequence, so it's more memorization, but if we trained on a set of sequences, the RNN would generalize patterns.

**Exercise queries answers:**

#### *Quiz / Exercises*

1. **Understanding Hidden State:** In the code, why do we call h \= h.detach() at the end of each epoch’s training loop? What could happen if we didn’t?  
   **Answer:** We detach the hidden state to break the computation graph between epochs. If we didn’t, PyTorch would try to backpropagate gradients from the next epoch through the hidden state into the previous epoch’s computations, which doesn’t make sense and would double count or accumulate gradients incorrectly. Essentially, without detaching, the model would treat the sequence as continuing from epoch to epoch, which is not what we want (each epoch is a fresh training pass). Detaching ensures that each epoch’s gradient calculations start anew with respect to the model parameters, not linking back to the old computations. Another way: Detaching prevents gradients from trying to propagate through h from one iteration of training to the next, avoiding a huge graph and incorrect gradient accumulation.

2. **Observation:** During generation, we used torch.multinomial to sample the next character instead of always taking torch.argmax (the most likely character). If our model has learned the training sequence almost perfectly, what’s the difference between these two approaches in output?  
   **Answer:** If the model learned the sequence almost perfectly, at each step the correct next character likely has probability near 1.0 (or very high). In that case, whether we sample or take argmax, we will get the same result (the correct next character) because it dominates. However, if there was any uncertainty or if we train on more data with some variety, using multinomial adds randomness – meaning it can produce different continuations according to probability distribution, not always the single highest probability sequence. Argmax would always give the single most likely next char (deterministic), which often leads the model to the most probable sequence (which in our trivial case is the one training sequence). In more complex generation, argmax can lead to repetitive or stuck outputs sometimes (like always choosing highest probability might cause the model to not explore less likely but plausible words, leading to loops like always picking the word "the" if it has slightly highest probability repeatedly). Sampling adds some stochasticity, which can produce more varied or “creative” text. In our specific tiny scenario, though, the difference is minimal because probabilities are essentially certain for the correct sequence.

3. **RNN vs. Feed-forward:** Consider a situation with the phrase: “The cat, which had been sleeping on the mat all afternoon, suddenly **woke** up.” A bigram model might predict “sat” or something weird after “suddenly” because it only sees “suddenly \_\_”. How could an RNN handle this better?  
   **Answer:** An RNN will process the whole sentence up to “suddenly”, carrying along context. In this sentence, the subject “cat” and the verb “had been sleeping” are earlier. By the time it gets to “suddenly \_\_\_ up”, the RNN’s hidden state could contain the information that the cat was sleeping, so something related to waking is likely. The word “suddenly” alone doesn’t tell you what comes after (bigram might guess “suddenly, it” or something odd), but the RNN’s memory includes that the cat had been sleeping. So an RNN has a chance to correctly predict “woke” because it knows the cat was asleep (so waking is a logical next event). Essentially, the RNN uses long-range dependency: it remembers the cat’s state (sleeping) and when “suddenly” arrives, it can combine that knowledge to predict “woke”. A feed-forward bigram can’t use “sleeping” information because it’s more than one word away; even a trigram might see “suddenly \_\_\_ up” and still not know about “sleeping”. The RNN’s hidden state encodes that prior clause, enabling a coherent prediction.

4. **Vanishing Gradient Thought:** If you made an RNN read a 100-word sentence and then asked it to recall the very first word, why is this hard for a basic RNN?  
   **Answer:** Because of the **vanishing gradient problem**. As the RNN processes 100 words, the influence of the first word on the hidden state tends to diminish with each step, especially if there’s no reinforcement of that info later. When training, the error signal that would adjust weights to remember word1 until the end has to backpropagate through 100 time steps. At each step, gradients can get multiplied by factors (like derivative of tanh etc.), often \< 1, causing them to shrink exponentially as they go back through 100 steps[\[12\]](https://www.tencentcloud.com/techpedia/112860#:~:text=%28RNNs%29%3F%20www,during%20backpropagation%20through%20time). By the time they reach weights related to word1, they’re almost zero – meaning the model doesn’t learn to preserve that info. So the network effectively “forgets” or can’t carry specific info for that long (unless the info is constantly reused or refreshed along the way). Without special mechanisms, it’s hard for a basic RNN to carry something unchanged for 100 steps. Long Short-Term Memory networks tackle this by providing a more linear, gated path for important information (like a conveyor belt where info can flow with little attenuation)[\[11\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20Core%20Idea%20Behind%20LSTMs), thereby remembering things for much longer. But a vanilla RNN would struggle to recall the first word after 100 steps because of vanishing gradients causing it not to learn such long dependencies.

5. **Application**: Name a real-world application where RNNs (or their improved versions) were traditionally used, and briefly describe how the sequential nature is crucial there.  
   **Answer:** One classic application: **Machine Translation**. Traditionally, an RNN encoder-decoder model (often using LSTMs) was used. The encoder RNN reads a sentence in, say, English word by word and compresses its meaning into a final hidden state (or a sequence of states). Then a decoder RNN starts generating a sentence in French, one word at a time, using that encoded information. The sequential nature is crucial because language is sequential: the meaning of a sentence is in the sequence of words. The RNN had to remember the whole source sentence (or contextually) to produce the correct translation. The dependency between early words and later words is essential (e.g., if the first word of English sentence is “John”, maybe the verb later has to be conjugated accordingly in French). RNNs handle those sequence dependencies by design. (Today, Transformers have taken over, but they also handle sequences via attention.)

Another example: **Speech Recognition**. Audio is a time sequence of sound features. RNNs (like LSTMs or GRUs) were used to process audio frames in order and output text. They needed to remember earlier sounds to correctly identify later parts of a word or context for disambiguation. The sequential nature is obvious – the sound “k” followed by “at” forms “cat”; an RNN would integrate over time to recognize that word.

A third example: **Text Generation/Prediction** (like predictive text or even handwriting generation). The model must base the next element on the sequence so far. RNNs naturally fit since they generate one step at a time, carrying context.

---

### Session 6: Building & Testing an RNN Language Model

#### *Teaching Script*

*Recap:*  
We’ve introduced RNNs conceptually. Now it’s time to build a simple RNN-based language model and see it in action. This session will be about implementation and experimentation: we’ll train a small RNN on some text and then use it to generate text. Through this, we’ll observe how an RNN can capture more context than an N-gram.

*Choosing a Corpus:*  
To keep things manageable, we might use a **character-level** model on a small dataset (like a children’s rhyme or a paragraph of text). Character-level means the RNN will treat each character as a token. This way, we can see it learn basics like spelling or simple structure. Alternatively, we can use a word-level model on a tiny corpus, but a word-level RNN with a very small dataset might not be very interesting (since it would just memorize a few sentences). Character-level on something like "Twinkle twinkle little star..." might show the network learning that pattern.

*Model Structure:*  
We’ll set up an RNN with a certain hidden size and feed characters in one by one. The network’s task: given all characters so far, predict the next character. During training, at each position, it will try to predict the actual next char in the sequence.

*Training Process Highlights:*  
\- **Sequence input:** We will likely feed in the whole training sequence or chunks of it to the RNN. There are a couple ways: \- Feed one character at a time, update hidden state on the fly (this is more manual). \- Or use PyTorch’s built-in RNN handling to pass in a whole sequence and get all the outputs. We did something like this in code, feeding the entire "hello world" at once by letting the RNN unroll internally. \- **Loss calculation:** We accumulate the prediction loss at each time step. For example, if the sequence is "HELLO", the model gets 'H' and should predict 'E', gets 'E' predict 'L', etc. We sum the loss for predicting each of 'E','L','L','O'. (Or average, it’s the same effect scaling wise). \- **Hidden state handling:** If we train on the sequence in one go, the hidden state will naturally carry through. If we train in batches or truncated segments, we need to carry over the hidden state between segments if the text is continuous. But if the training examples are independent sequences (like separate sentences), we’d reset hidden state at the start of each sequence.

For simplicity, maybe we’ll train on one or two fixed sequences (like two nursery rhymes). We can reset at each rhyme.

*Interpreting Training:*  
As it trains, initially it will predict mostly nonsense or very broad (e.g., often predicting a common character like space or 'e' regardless of input). As it learns, it should start to produce the right next letters more often. We can monitor the loss or even sample from the model during training to see improvement (like after a while it might start outputting something resembling words from the data).

*Overfitting Note:*  
With a small model and small data, the RNN might just memorize exactly and output the training text verbatim if we always pick the top output. That’s okay here since we just want it to demonstrate it can at least do that with more context. One could experiment by giving it a slightly different prompt and seeing if it can continue sensibly (if it generalized any pattern or just memorized). For example, train it on "hello world", then prompt "hell" and see if it gives "o world".

*Comparison to N-gram:*  
Consider how a bigram model would do on the same task. A bigram char model might produce similar local patterns (like "he"-\>"l", "ll"-\>"o"), but it doesn’t know about longer patterns like the word "world". A trigram might catch "wo"-\>"r", "wor"-\>"l", etc., but again fixed. The RNN, on the other hand, could learn the entire sequence "hello world" as one pattern. If we gave it "hell" it might output "o world" which is a 6-character continuation that goes beyond any fixed small N-gram.

*Practical use demonstration:*  
After training, we will generate some text from the RNN. This is essentially the model functioning as a language model. We should start with some seed (like a letter or a word) and let it produce the next steps. If trained on a nursery rhyme, we could see it continue the rhyme or produce a similar one. If we gave it a different start, maybe it will still produce something in the style of the training data (like similar meter or vocabulary).

We might need to mention: \- If the generation is gibberish at first, that’s normal – small RNN on small data is limited. \- But hopefully it produces something recognizable or even exactly the input sequence (if overfit). \- The point is that it learned the sequence as a whole, not just immediate next char probabilities independently.

*State Carrying:*  
An interesting aspect: If we don't reset hidden state between sequences, the model can carry context from one piece of text to another. For example, if we train it on two songs back-to-back as one long stream, the hidden state at the boundary contains info of first song which might influence how it starts the second (could be undesirable; we likely will treat them separately and reset hidden in between training songs).

*Next steps (foreshadowing LSTM):*  
We should highlight: If we tried a longer or more complex text, a simple RNN might struggle (especially if we need it to remember something from far back, like the subject of a long sentence). That’s why next we’ll learn about LSTM, which improves the memory aspect. But at least now we have an RNN that can, for example, remember the whole phrase "hello world" because it fits in its short-term memory. To handle, say, remembering a name at sentence start for a verb at sentence end (like we discussed), basic RNN often fails when distance grows; LSTM will handle it better.

*During this session's Q\&A or practice,* \- We might ask the student to try different hidden state sizes (does a bigger hidden layer remember patterns better? likely yes, up to a point). \- Or try feeding part of sequence and see if model completes it (testing its memory). \- Or feed it a completely unseen sequence to see if it just produces gibberish (since it’s never seen those combos). \- Emphasize that RNNs are powerful but need careful training and often more data. The small examples are just proof of concept.

#### *Presentation Slides*

* **Building an RNN Language Model (char-level example):**

* Use a simple text (e.g., "hello world" or a nursery rhyme) as training data.

* Treat each character as an input token. Vocabulary \= unique chars.

* The RNN will learn to predict the next character at each position.

* **Model Implementation Details:**

* **Embedding:** one-hot vectors for characters (or a small learned embedding, but one-hot is conceptually clear).

* **RNN cell:** takes one char at a time, updates hidden state of size N (we choose N, e.g., 64 or even 8 for demo).

* **Output:** a fully connected layer from hidden state to probabilities over next char.

* **Training Strategy:**

* If sequence \= "hello", inputs: "h","e","l","l" and targets: "e","l","l","o".

* RNN processes "h" \-\> predict "e", then "e" \-\> predict "l", ... etc.

* Compute loss \= sum of cross-entropy at each time step’s prediction.

* Use backprop through time to update weights.

* We might loop over the sequence many times (epochs) to learn it.

* **Monitoring Training:**

* Loss should decrease as the RNN memorizes the sequence patterns.

* Could print sample outputs from the RNN at intervals to see if it starts outputting correct sequence.

* The RNN might first learn common letters, then exact sequence.

* **Generation (Inference) with the Trained RNN:**

* Provide a **seed** (starting character or sequence).

* Use the RNN to get a probability distribution for next char, sample one.

* Feed that char back in to get the following one, and so on.

* Continue until a stopping condition (like a certain length or an end token if defined).

* This mimics how we would use a word-level model: seed with a prompt, generate next word repeatedly.

* **Example:** If trained on "hello world",

* Seed with "h", model likely outputs "e", feed "e" \-\> outputs "l", etc.

* It should reconstruct "hello world" (or something very close) if it learned well.

* If we seed with "he" it might continue with "llo world" etc.

* **What to Observe:**

* RNN can maintain context across multiple characters. E.g., it knows after "hell" comes "o" (it essentially learned the whole word "hello"). A bigram model would only know "he"-\>"l", "el"-\>"l", "ll"-\>"o", so it also could do "hello", but RNN does it not by storing explicit "ll-\>o" but by carrying state that after "hell" it expects "o".

* If we had a longer memory example (like a rhyme with repeating chorus), the RNN might learn the entire line as a state pattern and recall it.

* **Limitations with Small Data:**

* Our model might just memorize exactly, which is fine for demonstration but not generalization.

* With more data, RNNs can generalize patterns (e.g., learn spelling rules, or that "q" is usually followed by "u" in English, etc.). Even a char RNN famously can start generating somewhat word-like or sentence-like outputs after training on enough text, including things like closing quotes or brackets correctly if it learned that pattern over distance.

* For example, train a char RNN on Shakespeare, it generates Shakespeare-like text (often gibberish but with structure like character dialogues etc., as seen in some experiments). That shows RNN capturing long-range structures with enough capacity/data.

* **Comparison to N-gram in concept:**

* An N-gram of large N could also memorize "hello world" exactly if N is big enough (like 11-gram would memorize the whole phrase). But then any slight variation not seen would be impossible for it.

* The RNN with a smaller hidden might actually encode "hello world" in its weights and possibly be flexible if we change one letter, it might still produce a reasonable continuation (like if we trained on "hello world" and asked it to continue "helxo wor", maybe it still outputs something like "ld"? Hard to say with so little data, but with more data it generalizes spelling patterns).

* N-gram has to explicitly see every combo, RNN can interpolate and infer patterns.

* **To Note:** Basic RNN is a stepping stone. As mentioned, if we scale up, plain RNNs hit the wall with long dependencies. Next, we’ll introduce LSTM, which we can similarly implement but see that it retains info better. LSTM will be crucial for tasks where, e.g., the first word influences something 20 words later (like number agreement across a long sentence, or remembering a topic over a paragraph). With LSTM, our model will be “Long Short-Term Memory” meaning it can capture both short-term and some long-term patterns effectively[\[10\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=LSTMs%20are%20explicitly%20designed%20to,something%20they%20struggle%20to%20learn).

* **So, next session: LSTM Internals.** We’ll learn how adding gates (forget, input, output gates) to this RNN idea helps preserve long-range info by controlling the flow of information in the cell state[\[11\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20Core%20Idea%20Behind%20LSTMs)[\[13\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20LSTM%20does%20have%20the,regulated%20by%20structures%20called%20gates).

#### *Code Walkthrough*

\# RNN Language Model example: word-level on a small corpus (e.g., a nursery rhyme)

import torch  
import torch.nn as nn

\# Example tiny corpus (two short sentences)  
corpus \= \["twinkle twinkle little star", "twinkle little bat"\]  \# a variation to learn pattern maybe  
\# We'll train a word-level RNN on this. It's trivial small, but shows pattern "twinkle little X".

\# Build vocabulary of words  
words \= set()  
for sentence in corpus:  
    for w in sentence.split():  
        words.add(w)  
words \= sorted(list(words))  
vocab\_size \= len(words)  
word\_to\_idx \= {w: i for i, w in enumerate(words)}  
idx\_to\_word \= {i: w for w, i in word\_to\_idx.items()}  
print("Vocabulary:", words)

\# Prepare training data as sequences of indices  
sequences \= \[\]  
for sentence in corpus:  
    seq \= \[word\_to\_idx\[w\] for w in sentence.split()\]  
    sequences.append(seq)  
\# sequences: e.g., \[\[?, ?, ?, ?\], \[? ,? ,?\]\] depending on unique words mapping.

\# Define the RNN model  
embed\_dim \= 8  
hidden\_size \= 16  
class WordRNN(nn.Module):  
    def \_\_init\_\_(self, vocab\_size, embed\_dim, hidden\_size):  
        super().\_\_init\_\_()  
        self.embed \= nn.Embedding(vocab\_size, embed\_dim)  
        self.rnn \= nn.RNN(embed\_dim, hidden\_size, batch\_first=True)  
        self.fc \= nn.Linear(hidden\_size, vocab\_size)  
    def forward(self, x, h\_prev):  
        \# x: \[batch, seq\_len\]  
        em \= self.embed(x)  \# \[batch, seq\_len, embed\_dim\]  
        out, h \= self.rnn(em, h\_prev)  \# out: \[batch, seq\_len, hidden\_size\]  
        \# We want to make a prediction for each time step (except maybe last if end-of-seq)  
        logits \= self.fc(out)  \# \[batch, seq\_len, vocab\_size\]  
        return logits, h

model \= WordRNN(vocab\_size, embed\_dim, hidden\_size)  
loss\_fn \= nn.CrossEntropyLoss()  
optimizer \= torch.optim.Adam(model.parameters(), lr=0.1)

\# Training  
\# We'll train on each sequence separately (reset hidden at sequence start)  
n\_epochs \= 200  
for epoch in range(1, n\_epochs+1):  
    total\_loss \= 0.0  
    for seq in sequences:  
        seq\_input \= torch.tensor(\[seq\[:-1\]\], dtype=torch.long)   \# all but last as input  
        seq\_target \= torch.tensor(\[seq\[1:\]\], dtype=torch.long)   \# all but first as target  
        h0 \= torch.zeros(1, 1, hidden\_size)  
        optimizer.zero\_grad()  
        logits, hN \= model(seq\_input, h0)  
        \# Compute loss for each time step's prediction:  
        \# Reshape logits and targets to combine batch and seq dims  
        loss \= loss\_fn(logits.view(-1, vocab\_size), seq\_target.view(-1))  
        loss.backward()  
        optimizer.step()  
        total\_loss \+= loss.item()  
    if epoch % 50 \== 0:  
        print(f"Epoch {epoch}, Loss: {total\_loss/len(sequences):.3f}")

\# Testing generation  
model.eval()  
with torch.no\_grad():  
    \# Start with "twinkle"  
    start\_word \= "twinkle"  
    input\_idx \= torch.tensor(\[\[word\_to\_idx\[start\_word\]\]\])  \# batch=1, seq\_len=1  
    h \= torch.zeros(1, 1, hidden\_size)  
    generated \= \[start\_word\]  
    \# Generate next 3 words  
    for \_ in range(3):  
        logits, h \= model(input\_idx, h)  
        probs \= torch.softmax(logits\[:, \-1, :\], dim=1)  
        next\_idx \= torch.multinomial(probs, num\_samples=1)  
        next\_word \= idx\_to\_word\[next\_idx.item()\]  
        generated.append(next\_word)  
        \# Prepare next input  
        input\_idx \= next\_idx.unsqueeze(0)  \# shape \[1,1\]  
    print("Generated sequence:", " ".join(generated))

**Explanation:**  
\- We made a small corpus of two phrases: "twinkle twinkle little star" and "twinkle little bat" (like from a poem by Lewis Carroll). This is to see if the RNN can generalize the pattern "twinkle ... little ...". There's a repeated "twinkle" in first, and "little" in both. \- Vocabulary built from unique words: likely \["bat", "little", "star", "twinkle"\] (sorted alphabetically perhaps, or maybe in insertion order – we sorted, so probably \["bat","little","star","twinkle"\]). \- We convert each sentence to a list of indices. Example: \- "twinkle twinkle little star" \-\> \[id\_twinkle, id\_twinkle, id\_little, id\_star\] \- "twinkle little bat" \-\> \[id\_twinkle, id\_little, id\_bat\] \- The RNN is defined similarly to before but for word embeddings of size 8 and hidden 16\. Single-layer RNN.

* Training: We iterate epochs. For each sequence (sentence):

* We prepare input sequence (all words except last) and target sequence (all words except first).

* We initialize hidden state h0 to zeros for the start of each sequence.

* We forward through model to get logits for each step.

* Compute cross-entropy loss comparing each output to target word.

* Backprop and update.

* Sum loss to track average.

* After training, we attempt generation: start with "twinkle" and generate 3 more words.

* It's likely the network will output something like "twinkle little star" or "twinkle little bat" or some combination, depending on what it learned.

* Ideally, it might generalize to produce "twinkle little star" or "twinkle twinkle little star" or "twinkle little bat" etc. There's not much training data to truly generalize beyond memorating combos. But we might see if it learned that "twinkle" is often followed by "little" or sometimes another "twinkle".

Expected: The loss should decrease. Possibly:

Epoch 50, Loss: 1.x  
Epoch 100, Loss: \~0.5  
Epoch 150, Loss: \~0.2  
Epoch 200, Loss: \~0.1

Because it can probably memorize these two sequences. Actually, interestingly, "twinkle" appears twice in first sequence, which gives it some tricky part to predict second "twinkle" after first "twinkle". But it should learn that repetition.

Generated: \- It might output "twinkle little bat" or "twinkle little star". Which one? Possibly it might lean to one seen pattern. It saw: \- twinkle \-\> twinkle in one case (so maybe it might sometimes repeat twinkle). \- twinkle \-\> little in both (in first sequence after the repeated twinkle, but in second "twinkle little" directly; in first, after second twinkle it was "little").

So likely after one "twinkle", the model might choose "twinkle" again or "little". It might have learned a strong weight for "little" after "twinkle" because both sequences have "twinkle little" (one has it immediately, one after a repeated word). \- It might output: "twinkle little star" or "twinkle little bat". If it doesn't fully memorize ordering, it might output something like "twinkle little little star" or some odd combination if it overfits patterns weirdly. But likely one of the training phrases or a mix. Given how small data is, it might overfit exactly: Maybe it learned to do first sequence when sees "twinkle" and the second sequence is shorter. If we always start with "twinkle", maybe it always goes "twinkle little star" if that had more weight? Actually "twinkle twinkle little star" vs "twinkle little bat": \- In training, it saw one example where after "twinkle" the next word was also "twinkle". And one where after "twinkle" the next word was "little". So it's ambiguous. It might lean to always output "twinkle" or always "little" depending on weights.

We’ll see. Possibly it outputs "twinkle little star" (the more complete phrase). Anyway, that shows it had to juggle multiple possibilities, which RNN can do by adjusting probabilities in context: \- If hidden state after "twinkle" encodes that it's the start of sequence and maybe the network might decide if it's in a scenario of repeating or not. But with so little data, it might average them.

Given we gave it only one "twinkle" and no context, it might be confused whether to do second "twinkle" or go "little". It's essentially trying to guess which sequence we are in. If the model learned context that in first sequence there's a double twinkle (like maybe a clue is that if "twinkle" was at position 0 and we are at position 1 maybe?), but our input is just one "twinkle" at pos0. It might pick one at random due to similar training frequencies. So we might get either or maybe probabilities: If they were equal, sampling might pick either. We'll accept either output as demonstration.

**Quiz / Exercises**

1. **Training Observations:** If the RNN was trained on the two phrases in the corpus, what do you think the hidden state might be learning to distinguish? (Consider "twinkle twinkle little star" vs "twinkle little bat".)  
   **Answer:** The hidden state may be learning to distinguish which sequence (context) it's in. For instance, after the first word "twinkle", the hidden state could encode whether this seems like the repetition case or not. Possibly it learns something like: if at position 1 of the sequence and we just saw "twinkle", maybe it leans towards predicting "twinkle" again (since in one sequence that happened). But if at position 2 or after a "twinkle little", it might lean to "star" vs "bat". Essentially, it might encode a notion of the sequence’s progress: have we already seen a "twinkle" twice or not? Are we in the context of the first rhyme or the second? With such tiny data, the hidden state might simply memorize the exact sequence pattern: e.g., one pattern is 4 words long (with double twinkle), another is 3 words long. The RNN hidden state of size 16 is plenty to memorize these. It might, for example, dedicate some neurons to indicate "the last output was twinkle and it's the first word" versus "we're past the repeated twinkle". So the hidden state distinguishes the scenario after one "twinkle" (could either go to second "twinkle" or to "little"), and after seeing "twinkle little" it knows whether to output "star" or "bat" depending on which sequence it's in. With more data, hidden state generally captures context like "what previous words were" or "what part of sentence are we in".

2. **Generalization:** Suppose we gave the trained model a new start word that it never saw in training, what would it do? e.g., input "hello".  
   **Answer:** If "hello" was not in the training vocab, our model actually can’t handle it directly because it doesn’t have an embedding for "hello". If we assume it’s word-level and out-of-vocabulary (OOV), the model wouldn’t know what to do unless we had an \<UNK\> token. Realistically, we’d have to restrict input to known words. If we had an unknown handling, the RNN might produce some default or most common continuation (maybe it would act like a reset and give something like "twinkle ...", whatever it outputs for unknown context). But with how it’s set now, it literally can’t handle a word not in its dictionary. At char-level, if you give a char-level model an unseen character, similar issue – no embedding. But if it’s a known char but new sequence, it would just follow whatever its learned biases are (like maybe output common letters). So likely, a word-level model would fail on completely unknown word unless we had a fallback. If we restrict to known words, say we gave "little" as a start (which it knows but was never the first word in training), the RNN would still produce something – perhaps after "little" it might output "star" or "bat" because it has seen "little star" and "little bat". But it might be confused because in training "little" always followed "twinkle", not started a sequence. It might still output "bat" or "star" with some probability because it learned "little \-\> bat" and "little \-\> star" contexts. So overall, new start that's truly unknown is not possible in this simple model; if just a different arrangement, it might do something but not guaranteed meaningful.

3. **Temperature Sampling (advanced concept, not explicitly covered but to test intuition):** In text generation, sometimes we use a *temperature* parameter to adjust the randomness of sampling (temperature \> 1 makes it more random, \< 1 makes it more greedy). If our model is perfectly trained on "twinkle twinkle little star" and we set a very low temperature, what will the output look like starting from "twinkle"?  
   **Answer:** With a low temperature (near 0), the sampling process becomes very deterministic – it will almost always pick the highest probability next word. If the model is perfectly trained (so it essentially knows the sequence by heart), the highest probability continuation after "twinkle" might be another "twinkle" (if it learned the double usage strongly) or "little" (depending on what it emphasized, but likely it knows the first sequence had double twinkle at beginning, so it might strongly predict "twinkle" as the next word with probability almost 1). With low temp, it will pick that. Then after the second "twinkle", it will with high confidence pick "little", then "star". Essentially, low temperature will make it output the most learned sequence deterministically. So we’d expect it to output exactly "twinkle twinkle little star". Temperature basically controls randomness; a very low temperature forces the model to follow its top predictions strictly, which in a memorized scenario yields the memorized sequence verbatim.

4. **Multiple Epochs vs One Long Epoch:** In our training loop, we ran through the sequences in each epoch multiple times. If instead we concatenated the two training sequences into one long sequence and trained one epoch on that, would it be different from two sequences with resetting hidden state?  
   **Answer:** If we concatenated them, the RNN would treat "twinkle twinkle little star \<...\> twinkle little bat" as one continuous sequence. The hidden state wouldn’t reset between the first and second phrase. This means it would try to learn a transition from "star" to "twinkle" as well (which wasn’t an actual song lyric, but the model doesn’t know that). It might muddle the context between sequences. In contrast, treating them separately (resetting hidden state for each) ensures the model learns each phrase independently. So yes, difference: one long sequence means the model might learn some spurious connection (like maybe after "star" it should go to the second sequence "twinkle", which isn’t logically part of either original sequence but an artifact of how we fed data). It could also reduce confusion in training signals: by resetting, we made it clear that "twinkle" can start fresh. If we didn’t, the model might think "star" is always followed by some word (like "twinkle" in our combined feed), which isn’t intended. For such small data, it might overfit either way, but conceptually, separate sequences with reset is cleaner. In summary: Combining can blur sequence boundaries and cause the model to learn cross-sequence transitions that are not real, whereas resetting treats each training sequence independently. With enough data in a continuous text (like a book), we wouldn’t reset often, but when data are clearly separate samples (like separate sentences or lines), resetting helps.

5. **Reflect:** Now that we’ve built an RNN and seen it work on a small example, what would you say is the main benefit of an RNN over a Markov model (N-gram) in your own words?  
   **Answer:** *Student’s own phrasing, but key points:* An RNN can *learn* from data rather than just count occurrences, which allows it to generalize to sequences it hasn’t seen exactly. It keeps a running memory via its hidden state, so it can, in principle, use information from much earlier in the sequence when making predictions, something an N-gram model with a limited N can’t do. It doesn’t need to store every possible sequence in a table – instead, it compresses the history into a state, which is more efficient and can capture deeper patterns (like syntax or meaning) rather than just surface statistics. For example, an RNN can learn to balance parentheses or quotes over a long distance, or ensure subject-verb agreement, tasks where fixed-window models fail unless that pattern was explicitly in the training data. Essentially, RNNs have *dynamic memory*, adapting to sequence length, whereas N-grams are fixed memory and blow up in size for larger contexts.

---

[\[1\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=chain%20rule%20doesn%E2%80%99t%20really%20seem,1%20The%20Markov%20assumption) [\[2\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=of%20a%20word%20given%20its,words%2C%20instead%20of%20computing%20the) [\[3\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=word%2C%20we%20are%20thus%20making,future%20unit%20without%20looking%20too) [\[4\]](https://web.stanford.edu/~jurafsky/slp3/3.pdf#:~:text=P%28wn,future%20unit%20without%20looking%20too) web.stanford.edu

[https://web.stanford.edu/\~jurafsky/slp3/3.pdf](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

[\[5\]](https://www.tencentcloud.com/techpedia/112860#:~:text=1,many%20intervening%20words%20becomes%20difficult) [\[6\]](https://www.tencentcloud.com/techpedia/112860#:~:text=RNNs%20process%20data%20sequentially%2C%20which,unlike%20parallelizable%20models%20like%20Transformers) [\[12\]](https://www.tencentcloud.com/techpedia/112860#:~:text=%28RNNs%29%3F%20www,during%20backpropagation%20through%20time) What are the limitations of Recurrent Neural Networks (RNNs)? \- Tencent Cloud

[https://www.tencentcloud.com/techpedia/112860](https://www.tencentcloud.com/techpedia/112860)

[\[7\]](https://huggingface.co/blog/RDTvlokip/when-ai-has-the-memory-of-a-goldfish#:~:text=,Transformers%3A%20obsolete%20for%20most%20tasks)  RNN (Recurrent Neural Networks) — When AI has the memory of a goldfish\! 

[https://huggingface.co/blog/RDTvlokip/when-ai-has-the-memory-of-a-goldfish](https://huggingface.co/blog/RDTvlokip/when-ai-has-the-memory-of-a-goldfish)

[\[8\]](https://pub.towardsai.net/forget-the-math-a-simple-guide-to-the-attention-mechanism-4b65e5b80c83?gi=08d9f770d0fe#:~:text=In%20the%20old%20days%20,that%20worked%20like%20this) [\[9\]](https://pub.towardsai.net/forget-the-math-a-simple-guide-to-the-attention-mechanism-4b65e5b80c83?gi=08d9f770d0fe#:~:text=By%20the%20time%20the%20AI,it%20the%20girl%3F%20The%20ball) Forget the Math: A Beginner’s Guide to How Attention Powers GPT and Transformers | by Manash Pratim | Towards AI

[https://pub.towardsai.net/forget-the-math-a-simple-guide-to-the-attention-mechanism-4b65e5b80c83?gi=08d9f770d0fe](https://pub.towardsai.net/forget-the-math-a-simple-guide-to-the-attention-mechanism-4b65e5b80c83?gi=08d9f770d0fe)

[\[10\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=LSTMs%20are%20explicitly%20designed%20to,something%20they%20struggle%20to%20learn) [\[11\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20Core%20Idea%20Behind%20LSTMs) [\[13\]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#:~:text=The%20LSTM%20does%20have%20the,regulated%20by%20structures%20called%20gates) Understanding LSTM Networks \-- colah's blog

[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)