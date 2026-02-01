# Session 02 — Build a Bigram Text Generator

## Goal
Build a simple **bigram** (N=2) model that learns from text by counting word pairs, then generates new text by sampling likely next words.

## Time
~1 hour

## Key Ideas
- A bigram model learns counts for: $(\text{current word} \rightarrow \text{next word})$.
- Convert counts into probabilities (or sample directly using weighted random choice).

## Learning Outcomes
By the end of this session, the student can:
- Build a word-to-next-word frequency table from a corpus.
- Sample next words according to learned frequencies.
- Describe limitations (short context, repetition, nonsense after a while).

## Agenda (Suggested)
1. Choose a small corpus (public-domain book excerpt, article, etc.).
2. Tokenize text into words (keep it simple at first).
3. Count bigrams: for each word, record which words follow it and how often.
4. Generate text:
   - pick a start word
   - repeatedly sample the next word
5. Inspect output and discuss why it breaks down.

## Deliverables
- A working Python script that:
  - reads a text file
  - builds a bigram table
  - generates 50–150 words of text
- 2–3 generated samples saved in a Markdown file.

## Stretch (Optional)
- Add simple cleanup: lowercase, strip punctuation, or keep punctuation as separate tokens.

## Reflection Questions
- If a word never appears in the training text, what happens?
- What kinds of mistakes appear because the model only remembers 1 word?
