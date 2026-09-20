"""Final held-out evaluation of the saved model on the IMDB test split.

Run:  python test.py      (needs rnn_imdb.pt, produced by train.py)
"""

import torch
from datasets import load_dataset

import train
from train import encode, evaluate, device

ckpt = torch.load("rnn_imdb.pt")
vocab = ckpt["vocab"]
max_tokens = ckpt["max_tokens"]

# forward() resolves these as globals in train.py's module namespace
train.embed_dim = ckpt["embed_dim"]
train.hidden_dim = ckpt["hidden_dim"]
train.embedding = torch.nn.Embedding(len(vocab), ckpt["embed_dim"], padding_idx=vocab["<PAD>"]).to(device)
train.embedding.weight.data, train.W_xh, train.W_hh, train.b_h, train.W_hy, train.b_y = [
    t.to(device) for t in ckpt["params"]
]

imdb = load_dataset("stanfordnlp/imdb")

test_encoded = [encode(t, vocab)[-max_tokens:] for t in imdb["test"]["text"]]
test_labels = torch.tensor(imdb["test"]["label"], dtype=torch.float32).unsqueeze(1)

test_loss, test_acc = evaluate(test_encoded, test_labels)
print(f"TEST: loss {test_loss:.4f} | acc {test_acc:.3f}  (n = {len(test_encoded)})")
