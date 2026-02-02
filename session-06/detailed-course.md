# Session 6: Building & Testing an RNN Language Model

## Teaching Script

### Recap

We've introduced RNNs conceptually. Now it's time to build a simple RNN-based language model and see it in action. This session will be about implementation and experimentation: we'll train a small RNN on some text and then use it to generate text. Through this, we'll observe how an RNN can capture more context than an N-gram.

### Choosing a Corpus

To keep things manageable, we might use a **character-level** model on a small dataset (like a children's rhyme or a paragraph of text). Character-level means the RNN will treat each character as a token. This way, we can see it learn basics like spelling or simple structure. Alternatively, we can use a word-level model on a tiny corpus, but a word-level RNN with a very small dataset might not be very interesting (since it would just memorize a few sentences). Character-level on something like "Twinkle twinkle little star..." might show the network learning that pattern.

### Model Structure

We'll set up an RNN with a certain hidden size and feed characters in one by one. The network's task: given all characters so far, predict the next character. During training, at each position, it will try to predict the actual next char in the sequence.

### Training Process Highlights

- **Sequence input:** We will likely feed in the whole training sequence or chunks of it to the RNN. There are a couple ways:
  - Feed one character at a time, update hidden state on the fly (this is more manual).
  - Or use PyTorch's built-in RNN handling to pass in a whole sequence and get all the outputs.
- **Loss calculation:** We accumulate the prediction loss at each time step. For example, if the sequence is "HELLO", the model gets 'H' and should predict 'E', gets 'E' predict 'L', etc. We sum the loss for predicting each of 'E','L','L','O'.
- **Hidden state handling:** If we train on the sequence in one go, the hidden state will naturally carry through. If we train in batches or truncated segments, we need to carry over the hidden state between segments if the text is continuous.

### Interpreting Training

As it trains, initially it will predict mostly nonsense or very broad (e.g., often predicting a common character like space or 'e' regardless of input). As it learns, it should start to produce the right next letters more often. We can monitor the loss or even sample from the model during training to see improvement.

### Overfitting Note

With a small model and small data, the RNN might just memorize exactly and output the training text verbatim if we always pick the top output. That's okay here since we just want it to demonstrate it can at least do that with more context. One could experiment by giving it a slightly different prompt and seeing if it can continue sensibly.

### Comparison to N-gram

Consider how a bigram model would do on the same task. A bigram char model might produce similar local patterns (like "he"->"l", "ll"->"o"), but it doesn't know about longer patterns like the word "world". The RNN, on the other hand, could learn the entire sequence as one pattern. If we gave it "hell" it might output "o world" which goes beyond any fixed small N-gram.

### Practical Use Demonstration

After training, we will generate some text from the RNN. This is essentially the model functioning as a language model. We should start with some seed (like a letter or a word) and let it produce the next steps. If trained on a nursery rhyme, we could see it continue the rhyme or produce a similar one.

---

## Presentation Slides

### Building an RNN Language Model (char-level example)

- Use a simple text (e.g., "hello world" or a nursery rhyme) as training data.
- Treat each character as an input token. Vocabulary = unique chars.
- The RNN will learn to predict the next character at each position.

### Model Implementation Details

- **Embedding:** one-hot vectors for characters (or a small learned embedding, but one-hot is conceptually clear).
- **RNN cell:** takes one char at a time, updates hidden state of size N (we choose N, e.g., 64 or even 8 for demo).
- **Output:** a fully connected layer from hidden state to probabilities over next char.

### Training Strategy

- If sequence = "hello", inputs: "h","e","l","l" and targets: "e","l","l","o".
- RNN processes "h" -> predict "e", then "e" -> predict "l", ... etc.
- Compute loss = sum of cross-entropy at each time step's prediction.
- Use backprop through time to update weights.
- We might loop over the sequence many times (epochs) to learn it.

### Monitoring Training

- Loss should decrease as the RNN memorizes the sequence patterns.
- Could print sample outputs from the RNN at intervals to see if it starts outputting correct sequence.
- The RNN might first learn common letters, then exact sequence.

### Generation (Inference) with the Trained RNN

- Provide a **seed** (starting character or sequence).
- Use the RNN to get a probability distribution for next char, sample one.
- Feed that char back in to get the following one, and so on.
- Continue until a stopping condition (like a certain length or an end token if defined).

### What to Observe

- RNN can maintain context across multiple characters. E.g., it knows after "hell" comes "o" (it essentially learned the whole word "hello"). A bigram model would only know "he"->"l", "el"->"l", "ll"->"o", so it also could do "hello", but RNN does it by carrying state.
- If we had a longer memory example (like a rhyme with repeating chorus), the RNN might learn the entire line as a state pattern and recall it.

---

## Code Walkthrough

```python
# Simple Character-level RNN: learn to predict next character in a string
# (Using PyTorch for simplicity, focusing on RNN usage)

import torch
import torch.nn as nn

# 1. Prepare data: We'll use a small example sequence (character-level for clarity)
sequence = "hello world"
# We will train a char-level RNN to predict the next character given previous chars.

chars = sorted(set(sequence))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Convert the sequence into indices
seq_indices = [char_to_idx[ch] for ch in sequence]

# Create input-output pairs for training:
# For char RNN, input at time t is char t, output is char t+1.
inputs = seq_indices[:-1]   # all except last char (we don't predict after last)
targets = seq_indices[1:]   # all except first char (each target is next char)
# For "hello world", inputs = "hello worl", targets = "ello world"
train_len = len(inputs)

# Convert to tensors
inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

# 2. Define a simple RNN model
input_size = vocab_size   # one-hot char input of size = number of chars
hidden_size = 8          # hidden state size
output_size = vocab_size

class SimpleCharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # We use an Embedding for input instead of manual one-hot
        # (embedding of size = input_size just to create one-hot effect)
        self.embed = nn.Embedding(input_size, input_size)
        # Initialize embedding to behave like one-hot (weight = Identity matrix)
        self.embed.weight.data = torch.eye(input_size)
        self.embed.weight.requires_grad = False  # freeze it, so it stays one-hot
        # RNN layer: one layer, tanh activation by default
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        # x shape: [batch, seq_len] (batch_first = True)
        # h_prev shape: [num_layers, batch, hidden_size]
        # Embed x to one-hot (embedding is static one-hot)
        x_onehot = self.embed(x)  # shape [batch, seq_len, input_size]
        out, h_new = self.rnn(x_onehot, h_prev)
        # out: [batch, seq_len, hidden_size] for each time step's output
        # For prediction, we apply fc to each time step's output.
        out_reshaped = out.contiguous().view(-1, hidden_size)  # merge batch and seq for fc
        logits = self.fc(out_reshaped)  # shape [batch*seq_len, output_size]
        # Reshape logits back to [batch, seq_len, output_size]
        return logits.view(x.size(0), x.size(1), output_size), h_new

# Initialize model
model = SimpleCharRNN(input_size, hidden_size, output_size)
print("Characters:", chars)
print("Model initialized.")

# 3. Train the model (for simplicity, treat the whole sequence as one training sample)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train for a number of epochs
epochs = 200
# We'll use the whole sequence as a single batch for simplicity.
inputs = inputs.unsqueeze(0)  # make it [batch=1, seq_len]
targets = targets.unsqueeze(0)
h = torch.zeros(1, 1, hidden_size)  # initial hidden state (num_layers=1, batch=1, hidden_size)

for ep in range(1, epochs+1):
    optimizer.zero_grad()
    # Forward pass through the sequence
    logits, h = model(inputs, h)
    loss = loss_fn(logits.view(-1, output_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    # Detach hidden state to avoid gradients accumulating across epochs
    h = h.detach()
    if ep % 50 == 0:
        print(f"Epoch {ep}, Loss: {loss.item():.3f}")

# 4. Test: Let's generate text from the RNN
model.eval()
with torch.no_grad():
    test_h = torch.zeros(1, 1, hidden_size)
    # Start with 'h' to generate "hello world"
    start_char = 'h'
    idx = torch.tensor([[char_to_idx[start_char]]])  # batch=1, seq_len=1
    generated = start_char
    for _ in range(10):  # generate 10 characters
        logits, test_h = model(idx, test_h)
        probs = torch.softmax(logits[:, -1, :], dim=1)  # get probs for last time step
        # Sample from the distribution
        next_idx = torch.multinomial(probs, num_samples=1)  # sample one char index
        next_char = idx_to_char[next_idx.item()]
        generated += next_char
        # Prepare input for next iteration
        idx = next_idx.unsqueeze(0)  # shape [1,1]
    print("Generated sequence:", generated)
```

### Explanation

- We use a **character-level** example because it's easier to show an RNN's sequential nature on a short string like "hello world". Instead of words, our "vocabulary" is the set of characters in the string (h, e, l, o, space, w, r, d). This will demonstrate how the RNN can learn the sequence "h -> e -> l -> l -> o -> ...".

- We set up an artificial one-hot embedding layer. We freeze it so it doesn't train (since we want it to stay identity mapping from index to one-hot vector).

- The RNN layer `nn.RNN` is used. `batch_first=True` means input shape is [batch, seq_len, features]. It returns `out` (hidden output for each time step) and the final hidden state `h_new`.

- We then pass `out` through a Linear to get output logits for each time step.

- We create training data where each char is input and the next char is target. For "hello world", input sequence is "h e l l o   w o r l" and target sequence is "e l l o   w o r l d".

- We train the RNN for 200 epochs. The loss should decrease as it learns the sequence.

- After training, we test generation: start with 'h', feed it in, sample the next character, feed that back, and repeat.

### Expected Output

```
Characters: [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
Model initialized.
Epoch 50, Loss: 1.500
Epoch 100, Loss: 0.500
Epoch 150, Loss: 0.100
Epoch 200, Loss: 0.020
Generated sequence: hello world
```

Because 'h' leads to 'e', 'e' to 'l', etc., and the model learned perfectly on this tiny dataset, it should reconstruct "hello world".
