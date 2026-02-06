#!/usr/bin/env python3
"""
Text-conditioned U-Net for next-step PDE prediction.
Identical to vision-only baseline except for text conditioning.
Includes model saving and centralized hyperparameters.
"""

# =========================
# Imports
# =========================
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# ---- Hyperparameters ----
# =========================
input_steps = 5
batch_size = 16
epochs = 20
rollout_steps = 20
lr = 3e-4


# =========================
# Global vocabulary (IMPORTANT)
# =========================
GLOBAL_VOCAB = {}


# =========================
# Base Dataset
# =========================
class PDETensorTextDataset(Dataset):
    def __init__(self, jsonl_path, dtype=torch.float32):
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.dtype = dtype

        with open(self.jsonl_path, "r") as f:
            self.samples = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tensor = np.load(self.root / sample["tensor"])
        tensor = torch.from_numpy(tensor).to(self.dtype)

        text = sample["text"]
        return tensor, text


# =========================
# Next-step Dataset (TEXT AWARE)
# =========================
class NextStepPDEDataset(Dataset):
    def __init__(self, base_dataset, input_steps=5):
        self.base = base_dataset
        self.input_steps = input_steps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        solution, text = self.base[idx]

        x = solution[:self.input_steps]          # [k, X]
        y = solution[self.input_steps]           # [X]

        return (
            x.float(),                           # [k, X]
            y.float().unsqueeze(0),              # [1, X]
            text,
        )


# =========================
# Tokenizer
# =========================
def simple_tokenize(texts, max_len=16, vocab=GLOBAL_VOCAB):
    """
    texts: tuple/list of strings (length B)
    returns: LongTensor [B, max_len]
    """
    token_ids = torch.zeros(len(texts), max_len, dtype=torch.long)

    for i, txt in enumerate(texts):
        words = txt.lower().split()[:max_len]
        for j, w in enumerate(words):
            if w not in vocab:
                vocab[w] = len(vocab) + 1  # 0 reserved for padding
            token_ids[i, j] = vocab[w]

    return token_ids


# =========================
# Text Encoder
# =========================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids):
        emb = self.embedding(token_ids)   # [B, L, D]
        return emb.mean(dim=1)             # [B, D]


# =========================
# FiLM Layer
# =========================
class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_channels)
        self.beta = nn.Linear(cond_dim, num_channels)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1)
        return gamma * x + beta


# =========================
# U-Net Blocks
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


# =========================
# Text-conditioned U-Net
# =========================
class TextConditionedUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_ch=64, cond_dim=128):
        super().__init__()

        self.text_encoder = TextEncoder(emb_dim=cond_dim)

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
# Training / Evaluation
# =========================
def train_next_step(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y, texts in dataloader:
            token_ids = simple_tokenize(texts).to(device)
            x, y = x.to(device), y.to(device)

            preds = model(x, token_ids)
            loss = F.mse_loss(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | MSE = {total_loss / len(dataloader):.6f}")


@torch.no_grad()
def evaluate_next_step(model, dataloader, device):
    model.eval()
    mse, n = 0.0, 0
    for x, y, texts in dataloader:
        token_ids = simple_tokenize(texts).to(device)
        x, y = x.to(device), y.to(device)
        preds = model(x, token_ids)
        mse += F.mse_loss(preds, y, reduction="sum").item()
        n += y.numel()
    return mse / n


# =========================
# Rollout
# =========================
@torch.no_grad()
def rollout_prediction(model, init_sequence, token_ids, rollout_steps, device):
    model.eval()
    context = init_sequence.clone().to(device)
    preds = []

    for _ in range(rollout_steps):
        x = context.unsqueeze(0)
        next_step = model(x, token_ids).squeeze()
        preds.append(next_step.cpu())
        context = torch.cat([context[1:], next_step.unsqueeze(0)], dim=0)

    return torch.stack(preds)


@torch.no_grad()
def evaluate_rollout(model, dataloader, rollout_steps, device):
    model.eval()
    mse_per_step = torch.zeros(rollout_steps)
    count = 0

    for solution, text in dataloader.dataset.base:
        solution = solution.float()
        token_ids = torch.zeros(1, 16, dtype=torch.long).to(device)

        k = dataloader.dataset.input_steps
        init = solution[:k]
        gt = solution[k:k + rollout_steps]

        preds = rollout_prediction(model, init, token_ids, rollout_steps, device)
        mse_per_step += ((preds - gt) ** 2).mean(dim=1)
        count += 1

    return mse_per_step / count


# =========================
# Model Saving
# =========================
def save_model(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


# =========================
# Main
# =========================
def main():
    train_base = PDETensorTextDataset(
        "/Users/divyam/Course/Project Arbeit/pde_solver/vl_dataset/annotations.jsonl"
    )
    test_base = PDETensorTextDataset(
        "/Users/divyam/Course/Project Arbeit/pde_solver/vl_dataset/annotations_test.jsonl"
    )

    train_dataset = NextStepPDEDataset(train_base, input_steps=input_steps)
    test_dataset = NextStepPDEDataset(test_base, input_steps=input_steps)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = TextConditionedUNet1D(
        in_channels=input_steps,
        out_channels=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_next_step(model, train_loader, optimizer, device, epochs)

    print(
        "Text-conditioned next-step MSE:",
        evaluate_next_step(model, test_loader, device),
    )

    rollout_mse = evaluate_rollout(
        model, test_loader, rollout_steps, device
    )
    for t, err in enumerate(rollout_mse, 1):
        print(f"Step {t:02d}: MSE = {err.item():.6f}")

    save_model(
        model,
        "checkpoints/text_conditioned_unet.pt",
        metadata={
            "input_steps": input_steps,
            "batch_size": batch_size,
            "epochs": epochs,
            "rollout_steps": rollout_steps,
            "lr": lr,
            "vocab": GLOBAL_VOCAB,
            "model": "TextConditionedUNet1D",
        },
    )


if __name__ == "__main__":
    main()
