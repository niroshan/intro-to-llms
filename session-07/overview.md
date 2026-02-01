# Session 07 — From RNN to LSTM/GRU (Better Memory)

## Goal
Learn why vanilla RNNs struggle with long-term dependencies and how LSTMs/GRUs improve this using gates.

## Time
~1 hour

## Key Ideas
- Vanilla RNNs can have “fading gradients” (hard to learn long-range memory).
- LSTMs/GRUs add gating to control what to keep/forget.

## Learning Outcomes
By the end of this session, the student can:
- Explain the idea of gates (forget/input/output) in plain language.
- Swap an RNN for an LSTM/GRU in code.
- Compare text samples (RNN vs LSTM) qualitatively.

## Agenda (Suggested)
1. Recap: what the RNN learned, what it failed at.
2. Explain “memory control” with a valve analogy.
3. Replace `nn.RNN` with `nn.LSTM` (or `nn.GRU`).
4. Train briefly and generate samples.
5. Discuss differences and limitations.

## Deliverables
- Updated training run using LSTM/GRU.
- A short comparison note: RNN sample vs LSTM/GRU sample.

## Stretch (Optional)
- Increase sequence length and see which model handles it better.

## Reflection Questions
- What does it mean to “forget” information in a model?
- Why might LSTMs be better for longer texts?
