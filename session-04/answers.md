# Session 4: Quiz Answers

## 1. Code Understanding
In the model definition, why do we have `self.activation = nn.Tanh()` between the hidden layer and output layer? What would happen if we removed all activation functions?

**Answer:** The activation function (Tanh) introduces **non-linearity** into the network. Without any non-linearity, stacking multiple linear layers would be equivalent to a single linear transformation (since linear functions composed are still linear). This would severely limit what patterns the network can learn – it could only model linear relationships. With Tanh (or ReLU, etc.), the network can learn complex, non-linear patterns in the data. Removing it would make our model essentially just a fancy lookup table without the ability to learn complex representations.

## 2. Loss Interpretation
If the model's loss is 2.3 at the start and drops to 0.5 after training, what does this tell us about the model's learning?

**Answer:** Cross-entropy loss measures how "surprised" the model is by the correct answer. A loss of 2.3 (which is roughly ln(10) ≈ 2.3) suggests the model is essentially guessing randomly among ~10 words. A loss of 0.5 means the model is much more confident in its predictions – it's assigning high probability to the correct next word. The decrease shows the model successfully learned the patterns in the training data. A loss of 0 would mean perfect predictions with 100% confidence (unlikely in practice).

## 3. Embedding Dimension
Why might we choose different embedding dimensions (like 16 vs 256 vs 512) for different problems?

**Answer:** The embedding dimension determines how much information each word representation can encode:
- **Smaller dimensions (16-64)**: Good for small vocabularies, simple patterns, or limited training data. Uses fewer parameters, faster to train.
- **Larger dimensions (256-512+)**: Can capture more nuanced relationships between words, useful for large vocabularies and complex language patterns. Requires more data to train well and more computational resources.

The choice depends on vocabulary size, complexity of the task, available training data, and computational constraints. Too small might underfit (can't capture enough information), too large might overfit or be wasteful.

## 4. Practical Exercise
Modify the code to use a larger hidden layer (e.g., 32 instead of 16). Does this affect training? Why or why not for this tiny dataset?

**Answer:** For this tiny dataset (only ~10 training examples), increasing the hidden layer from 16 to 32 probably won't make a noticeable difference because:
- The model already has more than enough capacity to memorize these few patterns
- With so few examples, even a smaller network can achieve near-zero loss
- The bottleneck is data, not model capacity

You might see slightly different convergence patterns due to different initialization, but the final loss and predictions should be similar. In contrast, for a much larger dataset with more complex patterns, a larger hidden layer might help capture more intricate relationships.

## 5. Connection to Bigrams
Compare the neural model's predictions to what a pure bigram count model would give. Are they similar? When might they differ?

**Answer:** For this simple training data, they should be very similar:
- Both should predict "cat" after "the" with high probability
- Both should give roughly equal probability to "sat" and "ate" after "cat"
- Both should predict `<END>` after "fish"

They might differ because:
- The neural model might not perfectly converge to exact 50/50 for "sat"/"ate" (small numerical differences)
- The neural model gives non-zero (though tiny) probabilities to unseen combinations, while bigram gives exactly 0
- With more diverse training data, the neural model might generalize better (e.g., treating similar words similarly even if specific combinations weren't seen)

The key advantage of neural models becomes apparent with larger datasets where generalization matters more than memorization.
