# Session 04 — From Words to Numbers + Next-Word Neural Model

## Goal
Train a simple neural network that predicts the next word from the current word.

## Time
~1 hour

## Key Ideas
- Computers need numbers: words must be encoded.
- **One-hot encoding** represents each word as a vector with a single 1.
- A simple next-word predictor can be: one-hot → linear layer → softmax probabilities.

## Learning Outcomes
By the end of this session, the student can:
- Explain one-hot encoding and why it’s sparse.
- Train a simple classifier for next-word prediction.
- Sample text from the model (even if it’s short and messy).

## Agenda (Suggested)
1. Build a small vocabulary from your corpus.
2. Convert words to indices (ID numbers).
3. One-hot encoding (conceptually) and why embeddings are useful.
4. Model idea:
   - input: current word ID
   - output: probability distribution over next word IDs
5. Train with cross-entropy loss.
6. Generate a short continuation by sampling.

## Deliverables
- A training script/notebook that:
  - builds vocabulary
  - trains a next-word model
  - prints a few predictions for chosen prompt words

## Stretch (Optional)
- Compare predictions from bigrams vs neural model for a few prompt words.

## Reflection Questions
- How is this model similar to bigrams? How is it different?
- Why might it generalize slightly better than pure counts?
