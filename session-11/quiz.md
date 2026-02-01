# Session 11: Quiz

## 1. Perplexity Understanding
A model has perplexity 50 on the test set. What does this mean intuitively?

## 2. Overfitting Detection
You plot your training curves and see: Training loss = 0.5, Validation loss = 2.1, and they're diverging. What's happening and what should you do?

## 3. Hyperparameter Intuition
Your model trains but the loss oscillates wildly and never converges. Which hyperparameter is most likely the problem, and how should you change it?

## 4. Temperature Effects
Explain the effect of temperature on text generation. What happens with temperature = 0.5 vs temperature = 2.0?

## 5. Practical Exercise
Given these results, which model would you choose and why?

| Model | Train Loss | Val Loss | Perplexity (Val) | Coherence Rating |
|-------|------------|----------|------------------|------------------|
| A | 0.3 | 2.5 | 12.2 | 3.5/5 |
| B | 1.2 | 1.4 | 4.1 | 4.2/5 |
| C | 0.8 | 1.0 | 2.7 | 3.8/5 |

## 6. Code Exercise
Write a function that computes both train and validation loss, and prints a warning if the model appears to be overfitting.
