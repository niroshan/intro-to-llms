# Session 11: Quiz Answers

## 1. Perplexity Understanding
A model has perplexity 50 on the test set. What does this mean intuitively?

**Answer:** Perplexity of 50 means that, on average, the model is as uncertain about the next token as if it were choosing uniformly from 50 equally likely options. In other words, at each prediction step, the model effectively has about 50 reasonable choices. Lower perplexity is better – perplexity 1 would mean perfect prediction, while perplexity equal to vocabulary size would mean random guessing.

## 2. Overfitting Detection
You plot your training curves and see: Training loss = 0.5, Validation loss = 2.1, and they're diverging. What's happening and what should you do?

**Answer:** This is classic overfitting – the model is memorizing the training data instead of learning generalizable patterns. The large gap between training and validation loss (0.5 vs 2.1) indicates the model performs much better on data it's seen before.

Solutions:
1. Add/increase dropout regularization
2. Add/increase weight decay
3. Use early stopping (stop training earlier)
4. Get more training data
5. Reduce model size (fewer layers/smaller hidden size)
6. Use data augmentation if possible

## 3. Hyperparameter Intuition
Your model trains but the loss oscillates wildly and never converges. Which hyperparameter is most likely the problem, and how should you change it?

**Answer:** The most likely culprit is the **learning rate** being too high. When the learning rate is too large, each update overshoots the optimal point, causing the loss to bounce around instead of smoothly decreasing.

**Solution:** Reduce the learning rate (e.g., from 0.01 to 0.001 or 0.0001). You might also consider:
- Learning rate warmup (start very low, gradually increase)
- Gradient clipping (to prevent extreme updates)
- Learning rate scheduling (start higher, decrease over time)

## 4. Temperature Effects
Explain the effect of temperature on text generation. What happens with temperature = 0.5 vs temperature = 2.0?

**Answer:**
Temperature scales the logits before softmax: `logits / temperature`

**Temperature = 0.5 (low):**
- Dividing by 0.5 = multiplying by 2, making large logits even larger
- Softmax becomes "sharper" – high-probability tokens get even higher probability
- Generation becomes more deterministic/focused
- Output tends to be more repetitive but grammatically safer

**Temperature = 2.0 (high):**
- Dividing by 2.0 makes all logits smaller
- Softmax becomes "flatter" – probabilities become more uniform
- Generation becomes more random/diverse
- Output is more creative but may be less coherent/grammatical

**Tip:** Temperature ~0.7-1.0 often works well for coherent but somewhat varied text.

## 5. Practical Exercise
Given these results, which model would you choose and why?

| Model | Train Loss | Val Loss | Perplexity (Val) | Coherence Rating |
|-------|------------|----------|------------------|------------------|
| A | 0.3 | 2.5 | 12.2 | 3.5/5 |
| B | 1.2 | 1.4 | 4.1 | 4.2/5 |
| C | 0.8 | 1.0 | 2.7 | 3.8/5 |

**Answer:** **Model B** is likely the best choice despite not having the lowest perplexity:

- Model A is severely overfitting (train 0.3 vs val 2.5 is a huge gap)
- Model C has good numbers but lower human coherence rating
- Model B has:
  - Small train/val gap (1.2 vs 1.4) = good generalization
  - Reasonable perplexity (4.1)
  - Highest coherence rating (4.2/5)

The coherence rating matters because perplexity doesn't capture everything humans care about. A model might have low perplexity by predicting common words well but fail on fluency or meaning. Model B seems to have the best balance of generalization and output quality.

## 6. Code Exercise
Write a function that computes both train and validation loss, and prints a warning if the model appears to be overfitting.

**Answer:**
```python
def check_overfitting(model, train_data, val_data, threshold=1.5):
    """
    Compare train and val loss, warn if overfitting
    """
    criterion = nn.CrossEntropyLoss()
    
    train_loss = evaluate(model, train_data, criterion)
    val_loss = evaluate(model, val_data, criterion)
    
    ratio = val_loss / train_loss
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Ratio:      {ratio:.2f}")
    
    if ratio > threshold:
        print(f"⚠️ WARNING: Possible overfitting detected!")
        print(f"   Val loss is {ratio:.1f}x higher than train loss")
        print(f"   Consider: more dropout, weight decay, or early stopping")
    else:
        print(f"✓ Model appears to be generalizing well")
    
    return train_loss, val_loss, ratio
```
