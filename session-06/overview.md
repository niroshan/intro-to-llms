# Session 06 — Train a Character-Level RNN Language Model

## Goal
Train an RNN (or simple built-in PyTorch RNN) to predict the next character and generate text character-by-character.

## Time
~1 hour

## Key Ideas
- Character-level modeling keeps vocabulary small (letters + punctuation).
- Training examples are sequences; at each step you predict the next character.

## Learning Outcomes
By the end of this session, the student can:
- Prepare character data (mapping char ↔ ID).
- Train an RNN with cross-entropy loss.
- Generate a short text sample from the trained model.

## Agenda (Suggested)
1. Pick a text file (corpus).
2. Build char vocabulary and encode text into IDs.
3. Create training sequences (e.g. length 50–200 chars).
4. Train a small RNN (PyTorch `nn.RNN` is fine).
5. Generate text by sampling repeatedly.
6. Qualitative evaluation: what patterns does it learn?

## Deliverables
- Training code + a saved sample output.
- A short note: “What improved as training loss went down?”

## Stretch (Optional)
- Try different sampling temperatures (or just compare greedy vs random sampling).

## Reflection Questions
- Why might a character model produce real-looking words?
- What kinds of mistakes still appear and why?
