# Session 09 — Mini Transformer (Causal Self-Attention)

## Goal
Build (or assemble) a tiny Transformer-style language model that predicts the next token using causal self-attention.

## Time
~1 hour

## Key Ideas
- Transformers use attention instead of recurrence.
- **Causal mask** prevents looking at future tokens.
- Inputs need **positional information** (positional encoding/embeddings).

## Learning Outcomes
By the end of this session, the student can:
- Identify the main parts: token embeddings, positional encoding, attention layers, output head.
- Explain what “causal” means for language modeling.
- Run a small training loop (even if results are modest).

## Agenda (Suggested)
1. Architecture sketch: embeddings → attention blocks → logits.
2. Explain causal masking.
3. Implement a minimal block (or use `nn.TransformerEncoder` with a mask).
4. Train briefly on a small corpus.
5. Generate a short sample and discuss quality.

## Deliverables
- A runnable script/notebook that trains a tiny Transformer model.
- One generated sample + brief notes on what works/doesn’t.

## Stretch (Optional)
- Try different model sizes (embedding dim, heads, layers) and note impact.

## Reflection Questions
- Why can Transformers train faster than RNNs (in parallel)?
- Why might a tiny Transformer still struggle to generate good text?
