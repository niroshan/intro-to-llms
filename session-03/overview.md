# Session 03 — Neural Networks Basics (Weights, Loss, Learning)

## Goal
Understand neural networks as learnable functions: we adjust **weights** to reduce a **loss** using **gradient descent**.

## Time
~1 hour

## Key Ideas
- A model makes a prediction, we measure error (loss), then update weights to reduce that error.
- Training happens over many examples and multiple **epochs**.

## Learning Outcomes
By the end of this session, the student can:
- Define: weights, bias, loss function, learning rate, epoch.
- Explain gradient descent conceptually (no heavy calculus required).
- Implement (or use) a tiny model that learns from data.

## Agenda (Suggested)
1. Intuition: a model is a function with knobs (weights).
2. Loss: how wrong the model is.
3. Gradient descent: change weights a little to reduce loss.
4. Mini coding exercise:
   - fit a simple line to points, or
   - classify a tiny dataset
5. Plot or print loss as it improves.

## Deliverables
- A small Python program (NumPy or PyTorch) that trains on a toy dataset.
- A short note: “What changed during training and why?”

## Stretch (Optional)
- Try 2 different learning rates and compare whether training is stable.

## Reflection Questions
- What happens if the learning rate is too big?
- Why do we need many examples, not just one?
