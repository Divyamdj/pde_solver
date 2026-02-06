#!/usr/bin/env python3
"""
Train a 1D U-Net to predict the next time step of a 1D PDE solution
and evaluate multi-step rollouts.
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
# Base Dataset (Tensor + Text)
# =========================
class PDETensorTextDataset(Dataset):
    def __init__(self, jsonl_path, dtype=torch.float32):
        """
        Args:
            jsonl_path: path to JSONL annotations file
            dtype: torch dtype for loaded tensors
        """
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.dtype = dtype

        with open(self.jsonl_path, "r") as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tensor_path = self.root / sample["tensor"]
        tensor = np.load(tensor_path)           # [T, X]

        tensor = torch.from_numpy(tensor).to(self.dtype)
        text = sample["text"]

        return tensor, text


# =========================
# Dataset Wrapper (Next-step prediction)
# =========================
class NextStepPDEDataset(Dataset):
    def __init__(self, base_dataset, input_steps=5):
        """
        Args:
            base_dataset: PDETensorTextDataset
            input_steps: number of past time steps (k)
        """
        self.base = base_dataset
        self.input_steps = input_steps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        solution, _ = self.base[idx]   # solution: [T, X]

        x = solution[:self.input_steps]        # [k, X]
        y = solution[self.input_steps]         # [X]

        x = x.float()                          # [k, X]
        y = y.float().unsqueeze(0)             # [1, X]

        return x, y


# =========================
# Model (1D U-Net)
# =========================
class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_ch=64):
        super().__init__()

        self.enc1 = DoubleConv1D(in_channels, base_ch)
        self.enc2 = DoubleConv1D(base_ch, base_ch * 2)
        self.enc3 = DoubleConv1D(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool1d(2)

        self.bottleneck = DoubleConv1D(base_ch * 4, base_ch * 8)

        self.up3 = nn.Conv1d(base_ch * 8, base_ch * 4, kernel_size=1)
        self.up2 = nn.Conv1d(base_ch * 4, base_ch * 2, kernel_size=1)
        self.up1 = nn.Conv1d(base_ch * 2, base_ch, kernel_size=1)

        self.dec3 = DoubleConv1D(base_ch * 8, base_ch * 4)
        self.dec2 = DoubleConv1D(base_ch * 4, base_ch * 2)
        self.dec1 = DoubleConv1D(base_ch * 2, base_ch)

        self.out = nn.Conv1d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, k, X]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[-1], mode="linear", align_corners=False)
        d3 = self.up3(d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.up1(d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# =========================
# Training / Evaluation
# =========================
def train_next_step(model, dataloader, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = F.mse_loss(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1:02d} | MSE = {total_loss / len(dataloader):.6f}")


@torch.no_grad()
def evaluate_next_step(model, dataloader, device):
    model.eval()
    mse, n = 0.0, 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        mse += F.mse_loss(preds, y, reduction="sum").item()
        n += y.numel()

    return mse / n


# =========================
# Rollout utilities
# =========================
@torch.no_grad()
def rollout_prediction(model, init_sequence, rollout_steps, device):
    model.eval()

    context = init_sequence.clone().to(device)   # [k, X]
    preds = []

    for _ in range(rollout_steps):
        x = context.unsqueeze(0)                  # [1, k, X]
        next_step = model(x)                      # [1, 1, X]
        next_step = next_step.squeeze(0).squeeze(0)

        preds.append(next_step.cpu())

        context = torch.cat(
            [context[1:], next_step.unsqueeze(0)],
            dim=0
        )

    return torch.stack(preds, dim=0)              # [T, X]


@torch.no_grad()
def evaluate_rollout(model, dataloader, rollout_steps, device):
    model.eval()
    mse_per_step = torch.zeros(rollout_steps)
    count = 0

    for solution, _ in dataloader.dataset.base:
        solution = solution.float()

        k = dataloader.dataset.input_steps
        init = solution[:k]
        gt = solution[k : k + rollout_steps]

        preds = rollout_prediction(model, init, rollout_steps, device)

        mse_per_step += ((preds - gt) ** 2).mean(dim=1)
        count += 1

    return mse_per_step / count


# =========================
# Main
# =========================
def main():
    # ---- Dataset paths ----
    base_dataset = PDETensorTextDataset(
        "/Users/divyam/Course/Project Arbeit/pde_solver/src/dataset/annotations.jsonl"
    )

    test_base_dataset = PDETensorTextDataset(
        "/Users/divyam/Course/Project Arbeit/pde_solver/src/dataset/annotations_test.jsonl"
    )

    # ---- Hyperparameters ----
    input_steps = 5
    batch_size = 16
    epochs = 20
    rollout_steps = 20
    lr = 3e-4

    train_dataset = NextStepPDEDataset(base_dataset, input_steps=input_steps)
    test_dataset = NextStepPDEDataset(test_base_dataset, input_steps=input_steps)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet1D(in_channels=input_steps, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_next_step(model, train_loader, optimizer, device, epochs=epochs)

    test_mse = evaluate_next_step(model, test_loader, device)
    print("Vision-only next-step MSE:", test_mse)

    rollout_mse = evaluate_rollout(
        model=model,
        dataloader=test_loader,
        rollout_steps=rollout_steps,
        device=device
    )

    for t, err in enumerate(rollout_mse, start=1):
        print(f"Step {t:02d}: MSE = {err.item():.6f}")


if __name__ == "__main__":
    main()
