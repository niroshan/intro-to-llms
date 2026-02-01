# Session 08 — Attention: Focusing on Relevant Tokens

## Goal
Understand attention as a way to “look back” at different parts of a sequence and weight them by relevance.

## Time
~1 hour

## Key Ideas
- Attention computes weights over previous states/tokens.
- The output is a weighted sum (a “soft selection”).

## Learning Outcomes
By the end of this session, the student can:
- Explain attention using a “smart highlighter” analogy.
- Compute a tiny attention example using dot products.
- Describe why attention helps with long sequences.

## Agenda (Suggested)
1. Motivation: do we need to compress the whole past into one hidden state?
2. Core idea: compare a query to keys to get weights.
3. Toy example:
   - 3–5 vectors
   - compute similarity scores
   - softmax → weights
   - weighted sum
4. Connect to Transformers at a high level.

## Deliverables
- A worked example (handwritten or notebook) showing attention weights and the final weighted sum.
- A short paragraph: “Why is attention useful?”

## Stretch (Optional)
- Try 2 different queries and see how the weights change.

## Reflection Questions
- What does it mean if attention weight is close to 0?
- How is attention different from a fixed-size hidden state?
