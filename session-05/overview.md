# Session 05 — Why Sequences Are Hard + Introducing RNNs

## Goal
Understand why “one-word context” is not enough, and learn how an RNN keeps a running memory (hidden state).

## Time
~1 hour

## Key Ideas
- Language is sequential: meaning depends on earlier tokens.
- An RNN updates a hidden state each step:
  - $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$
  - $y_t = W_{hy} h_t$

## Learning Outcomes
By the end of this session, the student can:
- Explain what the hidden state is (a “memory vector”).
- Walk through an RNN step-by-step on a tiny character sequence.
- Describe why RNNs can handle longer context than bigrams.

## Agenda (Suggested)
1. Problem recap: bigrams forget too quickly.
2. RNN concept: same network reused at each time step.
3. Hidden state as memory.
4. Toy forward pass with small matrices (NumPy) over “H-E-L-L-O”.
5. Discuss what the model *could* learn (like after “q-u” comes “e”).

## Deliverables
- A short write-up (or notebook cell) that shows:
  - inputs $x_t$
  - hidden states $h_t$
  - outputs $y_t$
  for a short sequence

## Stretch (Optional)
- Change the hidden size and see how the output changes.

## Reflection Questions
- What information might the hidden state store?
- Why might it be hard for an RNN to remember something from 200 steps ago?
