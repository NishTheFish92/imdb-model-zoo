import re
from collections import Counter

from datasets import load_dataset


def tokenize(text):
    # lowercase, then pull out runs of word chars -> strips punctuation/HTML tags for free
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, min_freq=2):
    '''
    Takes whole text as input, tokenizes it and outputs a
    dict, that has the token as the key and its corresponding encoded number as the value.
    '''
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counts.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    '''
    Returns a list of encoded text based on the built vocab dict.
    '''
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]


def main():
    imdb = load_dataset("stanfordnlp/imdb")

    print(imdb)
    print()

    sample = imdb["train"][0]
    print("label:", sample["label"], "(0 = neg, 1 = pos)")
    print("text:", sample["text"][:300], "...")
    print()

    vocab = build_vocab(imdb["train"]["text"])
    print("vocab size:", len(vocab))
    print("encoded sample (first 20 tokens):", encode(sample["text"], vocab)[:20])


if __name__ == "__main__":
    main()
