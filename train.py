"""Toy RNN sentiment classifier for IMDB — hand-built recurrence, no nn.RNN.

Extracted from rnn_sentiment.ipynb. Run:  python train.py
Trains 45 epochs at lr=0.2, then 10 more at lr=0.05, then saves rnn_imdb.pt
"""

import re
from collections import Counter

import torch
from datasets import load_dataset
from tqdm.auto import tqdm


def tokenize(text):
    # lowercase, then pull out runs of word chars -> strips punctuation/HTML tags
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, min_freq=2):
    '''
    This function takes whole text as input, tokenizes it and outputs a
    dict, that has the token as the key and the tokenid as the value.

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
    Encodes the text based on the built vocab
    '''
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]


def pad_batch(encoded_reviews, pad_value=0):
    '''
    Right-pads a batch of variable-length encoded reviews to the batch's own max length.

    Returns:
        padded: (batch_size, max_len) torch.long tensor of token ids
        lengths: (batch_size,) torch.long tensor of each review's real (unpadded) length
    The lengths are later used for making the mask tensor that keeps track of padding
    '''
    lengths = torch.tensor([len(r) for r in encoded_reviews], dtype=torch.long)
    max_len = lengths.max().item()

    padded = torch.full((len(encoded_reviews), max_len), pad_value, dtype=torch.long)
    for i, review in enumerate(encoded_reviews):
        padded[i, :len(review)] = torch.tensor(review, dtype=torch.long)

    return padded, lengths


embed_dim = 100
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward(padded, lengths, training=False, p_drop=0.3):
    '''
    Takes the padded text, embeds the text, creates the mask and simulates one forward pass.
    Loops through all timesteps.

    In order to mitigate the gradient vanishing as much as possible, timestep pooling and averaging was performed.
    Neurons also have a 30% chance of being dropped to ensure the model doesn't learn incorrect patterns
    which prevents overfitting.
    '''
    B, T = padded.shape
    embedded = embedding(padded)
    embedded = torch.nn.functional.dropout(embedded, p=p_drop, training=training)
    mask = torch.arange(T, device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)

    h = torch.zeros(B, hidden_dim, device=padded.device)
    h_sum = torch.zeros(B, hidden_dim, device=padded.device)
    for t in range(T):
        m = mask[:, t].unsqueeze(1)
        h_new = torch.tanh(embedded[:, t, :] @ W_xh + h @ W_hh + b_h)
        h = torch.where(m, h_new, h)
        h_sum = h_sum + h * m          # only real steps contribute to the sum

    h_mean = h_sum / lengths.unsqueeze(1)
    return h_mean @ W_hy + b_y


def evaluate(encoded_reviews, labels, bs=128):
    '''
    Gets the evaluation metrics for validation.
    '''
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(encoded_reviews), bs):
            padded_b, lengths_b = pad_batch(encoded_reviews[start:start + bs])
            padded_b, lengths_b = padded_b.to(device), lengths_b.to(device)
            y_b = labels[start:start + bs].to(device)

            logits = forward(padded_b, lengths_b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_b)
            total_loss += loss.item() * len(y_b)
            total_correct += ((logits > 0).float() == y_b).sum().item()
    return total_loss / len(encoded_reviews), total_correct / len(encoded_reviews)


if __name__ == "__main__":
    imdb = load_dataset("stanfordnlp/imdb")
    print(imdb)

    vocab = build_vocab(imdb["train"]["text"])
    print("vocab size:", len(vocab))

    print("device:", device)

    max_tokens = 200
    n_train = 22000
    n_val = 2000 
    bs = 128 #batch size

    # Randomly arranges data so positive and negative reviews are almost equal in number
    torch.manual_seed(0)
    perm = torch.randperm(25000)
    subset = imdb["train"].select(perm[:n_train].tolist())
    val_subset = imdb["train"].select(perm[n_train:n_train + n_val].tolist())

    encoded = [encode(t, vocab)[-max_tokens:] for t in subset["text"]] #Captures only the last 200 tokens
    labels = torch.tensor(subset["label"], dtype=torch.float32).unsqueeze(1)
    val_encoded = [encode(t, vocab)[-max_tokens:] for t in val_subset["text"]]
    val_labels = torch.tensor(val_subset["label"], dtype=torch.float32).unsqueeze(1)

    print("train positive fraction:", labels.mean().item(), "| val positive fraction:", val_labels.mean().item())
    print("texts shared between train and val:", len(set(subset["text"]) & set(val_subset["text"])))

    # All trainable parameters
    embedding = torch.nn.Embedding(len(vocab), embed_dim, padding_idx=vocab["<PAD>"]).to(device)
    W_xh = (torch.randn(embed_dim, hidden_dim, device=device) * 0.01).requires_grad_()
    W_hh = (torch.randn(hidden_dim, hidden_dim, device=device) / hidden_dim ** 0.5).requires_grad_()
    b_h = torch.zeros(hidden_dim, device=device, requires_grad=True)
    W_hy = (torch.randn(hidden_dim, 1, device=device) * 0.01).requires_grad_()
    b_y = torch.zeros(1, device=device, requires_grad=True)

    params = [embedding.weight, W_xh, W_hh, b_h, W_hy, b_y]
    print("all leaf & on device:", all(p.is_leaf and p.device.type == device.type for p in params))

    # 45 epochs at lr=0.2, then 10 more at lr=0.05. same loop body, re-run with a lower lr.
    # This prevents learning from being too spiky when its almost generalized
    for epochs, lr in [(45, 0.2), (10, 0.05)]:
        print(f"\n=== {epochs} epochs at lr={lr} ===")
        for epoch in range(epochs):
            order = torch.randperm(n_train)
            total_loss, total_correct = 0.0, 0
            for start in tqdm(range(0, n_train, bs), desc=f"epoch {epoch}"):
                idx = order[start:start + bs].tolist()
                padded_b, lengths_b = pad_batch([encoded[i] for i in idx])
                padded_b, lengths_b = padded_b.to(device), lengths_b.to(device)
                y_b = labels[idx].to(device)

                logits = forward(padded_b, lengths_b, training=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_b)

                for p in params:
                    p.grad = None
                loss.backward()
                with torch.no_grad():
                    for p in params:
                        p -= lr * p.grad

                total_loss += loss.item() * len(idx)
                total_correct += ((logits > 0).float() == y_b).sum().item()

            val_loss, val_acc = evaluate(val_encoded, val_labels)
            print(f"epoch {epoch}: train loss {total_loss / n_train:.4f} acc {total_correct / n_train:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

    checkpoint = {
        # weights, in the same order as `params`
        "params": [p.detach().cpu() for p in params],
        "param_names": ["embedding.weight", "W_xh", "W_hh", "b_h", "W_hy", "b_y"],

        # without these the weights are meaningless
        "vocab": vocab,              # row i of the embedding table means whatever THIS vocab says
        "max_tokens": max_tokens,    # inputs must be truncated the same way
        "embed_dim": embed_dim,      # to rebuild the Embedding module
        "hidden_dim": hidden_dim,    # forward() reads this as a global

        # provenance
        "val_acc": val_acc,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, "rnn_imdb.pt")
    print(f"saved rnn_imdb.pt | {sum(p.numel() for p in params):,} params | val acc {val_acc:.3f}")
