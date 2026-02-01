# Worksheet: Build an N-gram Text Generator (Console App)

## Learning goal

You will build a command-line program that:

1. reads a text file (Alice in Wonderland)
2. tokenises it into “words”
3. trains an **n-gram model** (n = 2, 3, 4…)
4. generates a sentence from a **seed** word/phrase

No external packages. (Only built-in Python allowed.)

---

## Rules you must follow

### Token rules (what counts as a “word”)

* Everything must be **lowercase**
* A word token is:

  * letters `a`–`z`
  * apostrophe `'` allowed inside a word (`don't`, `alice's`)
* Sentence endings `. ! ?` become the special token: `"<END>"`

Example:
`"Alice's cat!"` → `["alice's", "cat", "<END>"]`

---

# File setup

Create a folder, e.g. `ngram_project/`

Inside, create:

* `ngram_console.py` (your code)
* (optional) `README.md` where you record what you did each milestone

Training file is here:
`/mnt/data/alice-in-wonderland-pg11.txt`

---

# Starter skeleton (copy into `ngram_console.py`)

Your job is to fill in the empty functions.

```python
import random
import re

END_TOKEN = "<END>"
DEFAULT_FILE_PATH = "/mnt/data/alice-in-wonderland-pg11.txt"


# =========================
# TASK 1: File reading
# =========================
def read_file(path):
    """
    Parameters:
        path (str): file path to read
    Returns:
        text (str): entire contents of the file as a single string
    """
    # TODO: implement
    pass


# =========================
# TASK 2: Tokenisation
# =========================
def tokenize(text, end_token=END_TOKEN):
    """
    Parameters:
        text (str): raw text from the book
        end_token (str): token to use for sentence endings, default "<END>"
    Returns:
        tokens (list of str): tokenised text

    Token rules:
      - lowercase
      - replace . ! ? with end_token
      - keep letters a-z and apostrophe '
      - everything else becomes spaces
      - split on whitespace
      - remove empty tokens
    """
    # TODO: implement
    pass


# =========================
# TASK 3: Build n-gram model
# =========================
def build_ngram_model(tokens, n):
    """
    Parameters:
        tokens (list of str): token list from tokenize()
        n (int): size of n-gram, e.g. 2=bigram, 3=trigram
    Returns:
        model (dict):
            key: state (tuple of n-1 tokens)
            value: dict of { next_token: count }

    Example (n=3):
        state = ("alice", "was")
        model[state] might be {"beginning": 2, "not": 1, "<END>": 1}
    """
    # TODO: implement
    pass


# =========================
# TASK 4: Weighted choice
# =========================
def weighted_choice(next_counts):
    """
    Parameters:
        next_counts (dict): {token: count}
    Returns:
        chosen_token (str): randomly chosen based on weights

    Example:
      {"cat": 10, "dog": 2} should pick "cat" about 5x more than "dog".
    """
    # TODO: implement
    pass


# =========================
# TASK 5: Choose start state
# =========================
def choose_start_state(model, seed_tokens, n):
    """
    Parameters:
        model (dict): trained n-gram model
        seed_tokens (list of str): tokenised seed words
        n (int): n-gram size
    Returns:
        state (tuple): a valid starting state (length n-1)

    Rules:
      1) If seed_tokens has >= n-1 tokens, try using the LAST n-1 as the state.
      2) Else if seed_tokens has at least 1 token, find states that start with seed_tokens[0].
      3) Else choose a random state from model.
    """
    # TODO: implement
    pass


# =========================
# TASK 6: Generate sentence
# =========================
def generate_sentence(model, n, seed_text, max_words=25, end_token=END_TOKEN):
    """
    Parameters:
        model (dict): trained model
        n (int): n-gram size used to train
        seed_text (str): user input seed, e.g. "alice" or "alice was"
        max_words (int): max words to generate
        end_token (str): "<END>" token
    Returns:
        sentence (str): generated sentence

    Stop generating when:
      - next token is end_token, OR
      - max_words reached, OR
      - the state has no next words
    """
    # TODO: implement
    pass


# =========================
# Console UI helpers
# =========================
def print_menu():
    print("\n=== N-Gram Text Generator ===")
    print("1) Train model")
    print("2) Generate sentence")
    print("3) Show model info")
    print("4) Quit")


def main():
    file_path = DEFAULT_FILE_PATH

    tokens = []
    model = {}
    n = 3
    max_words = 25

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            n_text = input("Choose n (2, 3, 4...): ").strip()
            if n_text.isdigit():
                n = int(n_text)
            else:
                print("Invalid n. Keeping:", n)

            print("Reading:", file_path)
            text = read_file(file_path)
            print("Tokenising...")
            tokens = tokenize(text)
            print("Tokens:", len(tokens))

            print("Building model...")
            model = build_ngram_model(tokens, n)
            print("States:", len(model))
            print("Training complete.")

        elif choice == "2":
            if not model:
                print("Train the model first (option 1).")
                continue

            seed = input("Enter seed word(s): ").strip()
            mw = input("Max words (Enter for default 25): ").strip()
            if mw.isdigit():
                max_words = int(mw)

            sentence = generate_sentence(model, n, seed, max_words)
            print("\nGenerated:")
            print(sentence)

        elif choice == "3":
            print("\n--- Model info ---")
            print("n:", n)
            print("Tokens loaded:", len(tokens))
            print("States:", len(model))
            if model:
                example_state = random.choice(list(model.keys()))
                print("Example state:", example_state)
                next_counts = model[example_state]
                # top 10 next tokens by count
                sorted_next = sorted(next_counts.items(), key=lambda x: x[1], reverse=True)
                print("Top next tokens:")
                for token, count in sorted_next[:10]:
                    print(" ", token, "->", count)

        elif choice == "4":
            print("Goodbye.")
            break

        else:
            print("Invalid option. Choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
```

---

# Milestone checklist + tasks

## Milestone A — Program runs + menu works

✅ **Task A1:** Run the script. It should show the menu.
Expected: it should not crash even though functions are empty (it will crash only if you choose training).

✅ **Task A2:** Choose option `4` to quit.

---

## Milestone B — Implement `read_file(path)`

### Task B1: Implement

**Function contract**

* Input: `path` (string)
* Output: file contents (string)

✅ **How to test**
Add temporarily after reading:

```python
print(text[:200])
```

Expected: prints first part of the book.

**Common bug:** forgetting `encoding="utf-8"` and getting decode errors.

---

## Milestone C — Implement `tokenize(text)`

### Task C1: Lowercase

Make sure you do: `text = text.lower()`

### Task C2: Replace sentence ends

Replace `. ! ?` with `" <END> "`

**Hint (allowed built-in):**
Use `re.sub(r"[.!?]+", f" {end_token} ", text)`

### Task C3: Remove unwanted characters

Replace everything except:

* letters a-z
* apostrophes
* whitespace
* `<` and `>` (so `<END>` survives)

**Hint:**
`re.sub(r"[^a-z'<>\s]", " ", text)`

### Task C4: Split into tokens

Collapse multiple spaces and split.

✅ **How to test**
After tokenising, print:

```python
print(tokens[:50])
```

Expected:

* all lowercase
* occasional `<END>`
* no weird punctuation tokens like `,` or `—`

**Quick manual test**
In a Python shell (or by printing inside code), try:

Input: `"Alice's cat!"`
Expected tokens: `["alice's", "cat", "<END>"]`

---

## Milestone D — Implement `build_ngram_model(tokens, n)`

### Task D1: Decide “state size”

`state_size = n - 1`

### Task D2: Walk along the tokens

Use a `for` loop with index `i` from `0` to `len(tokens) - n`

### Task D3: Build dictionary of dictionaries

You must produce:

* `model` is a dictionary
* `model[state]` is another dictionary of next tokens and counts

✅ **How to test**
After training, choose option `3` (Show model info).
Expected:

* “States” is a large number (thousands+)
* Example state prints something like `('alice', 'was')` for n=3
* Top next tokens are shown

**Common bug:** using a list as a dictionary key. Keys must be tuples.

---

## Milestone E — Implement `weighted_choice(next_counts)`

### Task E1: Total counts

Sum all counts in the dictionary.

### Task E2: Pick a random integer from `1` to `total`

Use `random.randint(1, total)`

### Task E3: Walk through counts until you reach the random number

Return that token.

✅ **How to test**
Temporarily call:

```python
test = {"cat": 10, "dog": 2}
# run weighted_choice 1000 times and count results
```

Expected: “cat” appears far more than “dog”.

---

## Milestone F — Implement `choose_start_state(model, seed_tokens, n)`

### Task F1: If seed is long enough

If `seed_tokens` length is at least `n-1`, take the last `n-1` tokens and make a tuple.

### Task F2: Otherwise, match first seed token

Find all states whose first word matches `seed_tokens[0]`

### Task F3: Else random

Pick a random state from `model.keys()`

✅ **How to test**
Use seed: `"alice was"` for trigram model.
Expected: output usually starts with “Alice was …”

---

## Milestone G — Implement `generate_sentence(model, n, seed_text, max_words)`

### Task G1: Tokenise the seed text

Call `tokenize(seed_text)`

### Task G2: Choose a start state

Call `choose_start_state(...)`

### Task G3: Build output list

Start output words as the words in the state.

### Task G4: Generate next tokens loop

Repeat until:

* you hit `<END>`
* reach max words
* state not found

Update the state each time by keeping last `n-1` words.

✅ **How to test**
Train `n=2`, generate with seed `alice`
Train `n=3`, generate with seed `alice was`

Expected: trigram is more coherent than bigram.

---

# “Extension” worksheet (optional challenges)

1. **Add option 5:** “Change training file path” (ask user to input path).
2. **Add option:** “Set random seed” so output can be repeatable.
3. **Fallback:** if trigram state missing, fall back to smaller n.

---

# Marking rubric (simple)

✅ **Pass**

* console runs
* tokenisation works + lowercase
* trains a model for any `n`
* generates sentences

⭐ **Merit**

* good input checking (n must be >= 2, model must be trained)
* max words works

🏆 **Distinction**

* repeatable random seed option
* fallback from n=3 to n=2 when stuck
* clear README explaining design
