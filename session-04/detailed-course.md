# Session 4: Building a Neural Language Model (Code Walkthrough)

## Teaching Script

### Recap

In the previous session, we learned the conceptual foundations of neural network language models: embeddings, hidden layers, softmax outputs, and training via gradient descent. Today, we'll put theory into practice by **implementing a simple neural network language model** using PyTorch. This hands-on session will solidify your understanding of how neural networks learn to predict the next word.

### Setting Up

We'll use **PyTorch**, a popular Python library for neural networks, because it handles the complex gradient calculations (backpropagation) automatically. This lets us focus on understanding the model architecture and training process rather than getting bogged down in calculus.

Our approach:
1. Prepare training data (bigram context -> next word pairs)
2. Define a neural network model
3. Train the model using gradient descent
4. Test predictions

### Data Preparation

Just like before, we'll use a simple corpus. We need to:
- Build a vocabulary (unique words with indices)
- Create training examples as (current_word_index, next_word_index) pairs
- Convert these to tensors (PyTorch's array format)

### Model Architecture

Our neural bigram model will have:
- **Embedding layer**: Converts word indices to dense vectors
- **Hidden layer**: Combines embeddings with learned weights, applies non-linearity (tanh)
- **Output layer**: Produces scores for each vocabulary word
- **Softmax**: Applied implicitly via CrossEntropyLoss during training

### Training Loop

The training process:
1. Zero out gradients from previous iteration
2. Forward pass: compute predictions
3. Compute loss (cross-entropy between predictions and targets)
4. Backward pass: compute gradients
5. Update weights using optimizer
6. Repeat for many epochs

### Key Observations

After training, we'll see:
- Loss decreases over epochs (model is learning)
- Predictions match training data patterns
- The model learned through weight adjustments, not explicit counting

---

## Presentation Slides

### Neural LM Implementation Overview

```mermaid
flowchart TB
    subgraph Data Preparation
        A[Corpus Text] --> B[Tokenize]
        B --> C[Build Vocabulary]
        C --> D[Create word->index mapping]
        D --> E[Generate training pairs]
    end
    
    subgraph Model
        F[Word Index] --> G[Embedding Layer]
        G --> H[Hidden Layer + Tanh]
        H --> I[Output Layer]
        I --> J[Softmax Probabilities]
    end
    
    subgraph Training
        K[Forward Pass] --> L[Compute Loss]
        L --> M[Backward Pass]
        M --> N[Update Weights]
        N --> K
    end
```

### PyTorch Basics

- **Tensors**: PyTorch's version of arrays/matrices (like NumPy but with GPU support and automatic gradients)
- **nn.Module**: Base class for neural network layers
- **nn.Embedding**: Lookup table mapping indices to vectors
- **nn.Linear**: Fully connected layer (matrix multiplication + bias)
- **Optimizer**: Handles weight updates (e.g., SGD, Adam)
- **Loss Function**: Measures prediction error (CrossEntropyLoss for classification)

### Training Workflow

1. **Initialize** model with random weights
2. **For each epoch:**
   - Clear gradients: `optimizer.zero_grad()`
   - Compute predictions: `outputs = model(inputs)`
   - Calculate loss: `loss = loss_fn(outputs, targets)`
   - Compute gradients: `loss.backward()`
   - Update weights: `optimizer.step()`
3. **Monitor** loss to see learning progress

---

## Code Walkthrough

```python
# Simple Neural Network Language Model (Feed-forward, fixed context)
# We'll implement a neural bigram model using PyTorch (for automatic differentiation)

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Prepare training data (bigram context -> next word pairs)
text = "the cat sat on the mat. the cat ate a fish."
# Simple preprocessing
words = text.lower().replace(".", " <END>").split()
vocab = sorted(set(words))
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# Create training examples: (current_word, next_word)
contexts = []
targets = []
for i in range(len(words)-1):
    w = words[i]
    next_w = words[i+1]
    if w == "<end>":  # skip the end token as context to predict next
        continue
    contexts.append(word_to_idx[w])
    targets.append(word_to_idx[next_w])

contexts = torch.tensor(contexts, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)
print("Training examples:", contexts.shape[0])

# 2. Define a simple neural network model
embed_dim = 16   # size of word embeddings
hidden_dim = 16  # size of hidden layer

class NeuralBigramLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)      # embedding layer
        self.hidden = nn.Linear(embed_dim, hidden_dim)        # hidden layer linear transform
        self.activation = nn.Tanh()                           # non-linear activation
        self.output = nn.Linear(hidden_dim, vocab_size)       # output layer (produces scores for each word)
        # Note: we'll apply Softmax as part of loss computation (CrossEntropyLoss does that implicitly)

    def forward(self, x):
        # x is a batch of word indices (tensor of shape [batch_size])
        emb = self.embed(x)                 # shape: [batch_size, embed_dim]
        h = self.hidden(emb)               # shape: [batch_size, hidden_dim]
        h = self.activation(h)             # apply non-linearity
        out_scores = self.output(h)        # shape: [batch_size, vocab_size] (raw scores for each word)
        return out_scores

model = NeuralBigramLM(vocab_size, embed_dim, hidden_dim)
print("Model initialized.")

# 3. Train the model
loss_fn = nn.CrossEntropyLoss()         # this will apply Softmax to output and compute loss against target
optimizer = optim.SGD(model.parameters(), lr=0.1)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(contexts)          # forward pass: get scores for each training context
    loss = loss_fn(outputs, targets)   # compute cross-entropy loss with true next words
    loss.backward()                   # backpropagate to compute gradients
    optimizer.step()                  # update weights
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# 4. Test the model's prediction for a couple of contexts
model.eval()
test_words = ["the", "cat", "fish"]
for w in test_words:
    if w not in word_to_idx:
        continue
    idx = torch.tensor([word_to_idx[w]])
    with torch.no_grad():
        score = model(idx)            # get scores for next word
        probs = torch.softmax(score, dim=1)  # convert to probabilities
    top_prob, top_idx = torch.max(probs, dim=1)
    predicted_word = idx_to_word[top_idx.item()]
    print(f"After '{w}', model predicts '{predicted_word}' with probability {top_prob.item():.2f}")
```

### Step-by-step Explanation

**Data Preparation:**
- We reuse the simple corpus "the cat sat on the mat. the cat ate a fish." as an example
- We tokenize it and build a vocabulary (unique words set)
- We create a mapping `word_to_idx` to convert words to numeric indices and vice versa
- We form training examples as pairs of (current_word_index, next_word_index) for every adjacent pair in the text (except we handle `<END>` carefully)
- So if the text is [the(=0), cat(=1), sat(=2), on(=3), ...], contexts will be [the, cat, sat, on, the, ...] and targets [cat, sat, on, the, mat, ...] (essentially one-step shifted)
- We skip making `<END>` a context, because after an end token, the concept of "next word" might be a start of a new sentence

**Model Definition:**
- We create a class `NeuralBigramLM` that inherits from `nn.Module` (PyTorch's base class for models)
- It has:
  - `nn.Embedding(vocab_size, embed_dim)`: this layer holds a matrix of shape [vocab_size, embed_dim] and looks up the embedding vector for a given word index
  - `nn.Linear(embed_dim, hidden_dim)`: a fully connected layer from embedding to hidden units
  - `nn.Tanh()`: a non-linear activation (we choose tanh here; ReLU or others could be used too). Tanh squashes values between -1 and 1
  - `nn.Linear(hidden_dim, vocab_size)`: output layer mapping hidden features to scores for each word in vocab
- In `forward`, we apply these in sequence: embed -> linear -> tanh -> linear. We return raw scores (logits) for each word in vocab

**Training Setup:**
- We use `nn.CrossEntropyLoss` as the loss function. This loss in PyTorch expects raw scores and integer targets; it internally applies Softmax and computes the negative log likelihood of the correct class
- We use `optim.SGD` (stochastic gradient descent) with a learning rate 0.1 to optimize parameters

**Training Loop:**
- We run a training loop for `n_epochs` (100). Each epoch:
  - We set model to train mode
  - `optimizer.zero_grad()` clears any gradients from previous step
  - `outputs = model(contexts)`: we pass all training contexts in one go (PyTorch can handle batched computation)
  - `loss = loss_fn(outputs, targets)`: compares each output row with the corresponding target word index and computes average cross-entropy loss
  - `loss.backward()`: computes gradients for all model parameters
  - `optimizer.step()`: updates the weights by a small step in direction of negative gradient
  - We print loss every 20 epochs to monitor training. Loss should decrease over epochs

**Testing the model:**
- After training, we switch to `model.eval()` (best practice)
- We test a few input words: "the", "cat", "fish"
- For each, we form a tensor with that word's index, do `model(idx)` to get scores, then softmax to get probabilities
- Find the top predicted word (with `torch.max`)
- Print the result

### Expected Output

During training, the loss will start high and go down:

```
Training examples: 10
Model initialized.
Epoch 20/100, Loss: 2.1000
Epoch 40/100, Loss: 1.5000
Epoch 60/100, Loss: 1.0000
Epoch 80/100, Loss: 0.7000
Epoch 100/100, Loss: 0.5000
```

Testing might output:

```
After 'the', model predicts 'cat' with probability 0.99
After 'cat', model predicts 'sat' with probability 0.70
After 'fish', model predicts '<END>' with probability 0.80
```

**Interpretation:**
- "the" -> "cat" (makes sense because in our corpus "the" was always followed by "cat")
- "cat" -> maybe "sat" (since "cat sat" and "cat ate" were the two occurrences; the model might lean towards "sat" or show roughly 50/50)
- "fish" -> "<END>" likely because "fish" was end of sentence in training

### What Did We Achieve?

We trained a neural network to mimic the bigram probabilities of our tiny corpus. It essentially learned the same thing the count model had, but via adjusting weights:
- The embedding for "the" combined with hidden layer likely got tuned to strongly activate output neuron for "cat"
- The embedding for "cat" got tuned to split activation between "sat" and "ate" output neurons
- This confirms that even a simple neural net can learn our bigram distribution

It's a small step, but an important one: with this, we see how **learning from data** replaces explicit counting. With larger data and more complex networks, this approach scales to powerful models.

---

## Quiz / Exercises

### 1. Code Understanding
In the model definition, why do we have `self.activation = nn.Tanh()` between the hidden layer and output layer? What would happen if we removed all activation functions?

**Answer:** The activation function (Tanh) introduces **non-linearity** into the network. Without any non-linearity, stacking multiple linear layers would be equivalent to a single linear transformation (since linear functions composed are still linear). This would severely limit what patterns the network can learn – it could only model linear relationships. With Tanh (or ReLU, etc.), the network can learn complex, non-linear patterns in the data. Removing it would make our model essentially just a fancy lookup table without the ability to learn complex representations.

### 2. Loss Interpretation
If the model's loss is 2.3 at the start and drops to 0.5 after training, what does this tell us about the model's learning?

**Answer:** Cross-entropy loss measures how "surprised" the model is by the correct answer. A loss of 2.3 (which is roughly ln(10) ≈ 2.3) suggests the model is essentially guessing randomly among ~10 words. A loss of 0.5 means the model is much more confident in its predictions – it's assigning high probability to the correct next word. The decrease shows the model successfully learned the patterns in the training data. A loss of 0 would mean perfect predictions with 100% confidence (unlikely in practice).

### 3. Embedding Dimension
Why might we choose different embedding dimensions (like 16 vs 256 vs 512) for different problems?

**Answer:** The embedding dimension determines how much information each word representation can encode:
- **Smaller dimensions (16-64)**: Good for small vocabularies, simple patterns, or limited training data. Uses fewer parameters, faster to train.
- **Larger dimensions (256-512+)**: Can capture more nuanced relationships between words, useful for large vocabularies and complex language patterns. Requires more data to train well and more computational resources.

The choice depends on vocabulary size, complexity of the task, available training data, and computational constraints. Too small might underfit (can't capture enough information), too large might overfit or be wasteful.

### 4. Practical Exercise
Modify the code to use a larger hidden layer (e.g., 32 instead of 16). Does this affect training? Why or why not for this tiny dataset?

**Answer:** For this tiny dataset (only ~10 training examples), increasing the hidden layer from 16 to 32 probably won't make a noticeable difference because:
- The model already has more than enough capacity to memorize these few patterns
- With so few examples, even a smaller network can achieve near-zero loss
- The bottleneck is data, not model capacity

You might see slightly different convergence patterns due to different initialization, but the final loss and predictions should be similar. In contrast, for a much larger dataset with more complex patterns, a larger hidden layer might help capture more intricate relationships.

### 5. Connection to Bigrams
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
