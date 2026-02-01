# Session 01 — What Is a Language Model?

## Goal
Understand what a language model (LM) does: it predicts what token (word/character) is likely to come next, given some context.

## Time
~1 hour

## Key Ideas
- A **token** is a piece of text (we’ll start with *words*; later we’ll use *characters*).
- **Context** means the previous tokens.
- An LM estimates probabilities like: $P(\text{next word} \mid \text{previous words})$.

## Learning Outcomes
By the end of this session, the student can:
- Explain “predict the next token” in plain English.
- Give 2–3 real examples (autocomplete, chatbots, translation).
- Explain what an **N-gram** is (bigram, trigram) and why more context can help.

## Agenda (Suggested)
1. Warm-up: “next word guessing” game using a few example sentences.
2. Define language models: predicting next tokens from context.
3. N-grams: bigram (previous 1 word), trigram (previous 2 words).
4. Why N-grams struggle: sparsity (many sequences never appear in training text).
5. Mini reflection: what would make predictions better?

## Deliverables
- A short written explanation (5–8 sentences): “What is a language model?”
- A glossary list: token, context, probability, N-gram.

## Stretch (Optional)
- Find a short paragraph online and highlight what you think the next word will be at 10 random points.

## Reflection Questions
- Why might a bigram model produce weird long sentences?
- What might a model need in order to remember earlier parts of a paragraph?
