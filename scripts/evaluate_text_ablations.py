#!/usr/bin/env python3
"""
Evaluation-only script:
- Loads a trained text-conditioned U-Net checkpoint
- Runs rollout comparison under text ablations:
    1) correct text
    2) shuffled text (dataset-level permutation)
    3) empty text
"""

# =========================
# Imports
# =========================
import json
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Global vocab restored from checkpoint metadata
GLOBAL_VOCAB = {}


# =========================
# Dataset
# =========================
class PDETensorTextDataset(Dataset):
    def __init__(self, jsonl_path, dtype=torch.float32):
        self.root = Path(jsonl_path).parent
        with open(jsonl_path, "r") as f:
            self.samples = [json.loads(l) for l in f]
        self.dtype = dtype

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tensor = torch.from_numpy(np.load(self.root / s["tensor"])).to(self.dtype)
        return tensor, s["text"]


# =========================
# Tokenizer (must match training)
# =========================
def simple_tokenize(texts, max_len=16, vocab=GLOBAL_VOCAB):
    """
    texts: list/tuple[str], length B
    returns: LongTensor [B, max_len]
    """
    token_ids = torch.zeros(len(texts), max_len, dtype=torch.long)

    for i, txt in enumerate(texts):
        words = txt.lower().split()[:max_len]
        for j, w in enumerate(words):
            if w not in vocab:
                # NOTE: for evaluation we should NOT expand vocab,
                # but we keep this safe: unseen words -> 0 (padding)
                token_ids[i, j] = 0
            else:
                token_ids[i, j] = vocab[w]

    return token_ids


# =========================
# Model definitions (must match training)
# =========================
class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids).mean(dim=1)


class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1)
        # Stabilized FiLM (recommended)
        return (1 + 0.1 * gamma) * x + 0.1 * beta


class TextConditionedUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, vocab_size, base_ch=64, cond_dim=128):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size=vocab_size, emb_dim=cond_dim)

        self.enc1 = DoubleConv1D(in_channels, base_ch)
        self.enc2 = DoubleConv1D(base_ch, base_ch * 2)
        self.enc3 = DoubleConv1D(base_ch * 2, base_ch * 4)

        self.film1 = FiLM(cond_dim, base_ch)
        self.film2 = FiLM(cond_dim, base_ch * 2)
        self.film3 = FiLM(cond_dim, base_ch * 4)

        self.pool = nn.MaxPool1d(2)
        self.bottleneck = DoubleConv1D(base_ch * 4, base_ch * 8)

        self.up3 = nn.Conv1d(base_ch * 8, base_ch * 4, 1)
        self.up2 = nn.Conv1d(base_ch * 4, base_ch * 2, 1)
        self.up1 = nn.Conv1d(base_ch * 2, base_ch, 1)

        self.dec3 = DoubleConv1D(base_ch * 8, base_ch * 4)
        self.dec2 = DoubleConv1D(base_ch * 4, base_ch * 2)
        self.dec1 = DoubleConv1D(base_ch * 2, base_ch)

        self.out = nn.Conv1d(base_ch, out_channels, 1)

    def forward(self, x, token_ids):
        cond = self.text_encoder(token_ids)

        e1 = self.film1(self.enc1(x), cond)
        e2 = self.film2(self.enc2(self.pool(e1)), cond)
        e3 = self.film3(self.enc3(self.pool(e2)), cond)

        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, e3.shape[-1], mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(d3), e3], dim=1))

        d2 = F.interpolate(d3, e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))

        d1 = F.interpolate(d2, e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))

        return self.out(d1)


# =========================
# Rollout
# =========================
@torch.no_grad()
def rollout_prediction(model, init, token_ids, steps, device):
    model.eval()
    ctx = init.clone().to(device)
    preds = []

    for _ in range(steps):
        x = ctx.unsqueeze(0)                # [1, k, X]
        y = model(x, token_ids).squeeze()   # [X]
        preds.append(y.cpu())
        ctx = torch.cat([ctx[1:], y.unsqueeze(0)], dim=0)

    return torch.stack(preds)              # [T, X]


# =========================
# Text Ablation Setup (dataset-level)
# =========================
def build_text_views(dataset, seed=0):
    """
    Returns 3 lists of texts aligned with dataset indices:
        texts_correct[i]  = original text for sample i
        texts_shuffled[i] = text from a different sample (permutation)
        texts_empty[i]    = ""
    """
    texts_correct = [dataset[i][1] for i in range(len(dataset))]

    rng = random.Random(seed)
    perm = list(range(len(dataset)))
    rng.shuffle(perm)
    texts_shuffled = [texts_correct[j] for j in perm]

    texts_empty = ["" for _ in range(len(dataset))]

    return texts_correct, texts_shuffled, texts_empty


@torch.no_grad()
def evaluate_rollout_with_text_list(model, dataset, text_list, steps, device):
    """
    dataset: PDETensorTextDataset
    text_list: list[str] aligned with dataset indices
    """
    mse = torch.zeros(steps)
    n = 0

    for i in range(len(dataset)):
        sol, _ = dataset[i]
        text = text_list[i]

        token_ids = simple_tokenize((text,)).to(device)  # [1, L]

        sol = sol.float()
        init = sol[:5]
        gt = sol[5:5 + steps]

        pred = rollout_prediction(model, init, token_ids, steps, device)
        mse += ((pred - gt) ** 2).mean(dim=1)
        n += 1

    return mse / n


# =========================
# Main
# =========================
def main():
    # ----- Device -----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ----- Paths -----
    test_jsonl = "/Users/divyam/Course/Project Arbeit/pde_solver/src/dataset/annotations_test.jsonl"
    ckpt_path = "/Users/divyam/Course/Project Arbeit/pde_solver/scripts/checkpoints/text_conditioned_unet.pt"

    # ----- Load dataset -----
    dataset = PDETensorTextDataset(test_jsonl)

    # ----- Load checkpoint -----
    ckpt = torch.load(ckpt_path, map_location=device)

    # ----- Restore vocab correctly -----
    GLOBAL_VOCAB.clear()
    GLOBAL_VOCAB.update(ckpt["metadata"]["vocab"])

    vocab_size = ckpt["model_state_dict"]["text_encoder.embedding.weight"].shape[0]

    # ----- Build model with correct vocab size -----
    model = TextConditionedUNet1D(
        in_channels=5,
        out_channels=1,
        vocab_size=vocab_size
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ----- Build text ablation views (dataset-level shuffle) -----
    texts_correct, texts_shuffled, texts_empty = build_text_views(dataset, seed=0)

    steps = 20

    print("\n=== Text Ablation Rollout MSE (Evaluation Only) ===")

    mse_correct = evaluate_rollout_with_text_list(model, dataset, texts_correct, steps, device)
    mse_shuffled = evaluate_rollout_with_text_list(model, dataset, texts_shuffled, steps, device)
    mse_empty = evaluate_rollout_with_text_list(model, dataset, texts_empty, steps, device)

    print("\nStep | Correct | Shuffled | Empty")
    print("----------------------------------")
    for t in range(steps):
        print(
            f"{t+1:02d}   | "
            f"{mse_correct[t].item():10.2f} | "
            f"{mse_shuffled[t].item():10.2f} | "
            f"{mse_empty[t].item():10.2f}"
        )


if __name__ == "__main__":
    main()
