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
