## Results

Trained and evaluated two models for **1D PDE next-step prediction**:

- **Vision-only U-Net baseline** (`baseline.py`)
- **Text-conditioned U-Net** (`train_text_conditioned_unet_pde.py`) using **FiLM conditioning**

Both models predict the next PDE timestep given the previous **k = 5** timesteps as input.

---

## Vision-only baseline (U-Net)

**Test next-step MSE:** `1.7387`

**Rollout MSE (20 steps):**

| Step | MSE |
|------|-----|
| 1    | 1.74 |
| 5    | 103.25 |
| 10   | 844.69 |
| 15   | 1251.89 |
| 20   | 1285.60 |

---

## Text-conditioned U-Net (FiLM)

**Test next-step MSE:** `1.6236`

**Rollout MSE (20 steps):**

| Step | MSE |
|------|-----|
| 1    | 1.62 |
| 5    | 61.37 |
| 10   | 195.01 |
| 15   | 328.37 |
| 20   | 473.15 |

---

## Text ablation study

We evaluated rollout performance under three text conditions:

- **Correct:** true text paired with each sample  
- **Shuffled:** text randomly permuted across samples  
- **Empty:** text replaced with an empty string  

**Results (rollout MSE):**

| Condition | Step 1 | Step 10 | Step 20 |
|----------|--------|---------|---------|
| Correct  | 1.62   | 195.01  | 473.15  |
| Shuffled | 1.62   | 195.01  | 473.15  |
| Empty    | 2.37   | 799.99  | 3581.56 |

---

## Interpretation

- **Text-conditioning improves rollout stability significantly**, and also improves next-step prediction accuracy.
- **Correct vs Shuffled are identical**, which suggests the model is not using *sample-specific* text information.
- **Empty text is much worse**, indicating the model relies on the presence of text embeddings as a stabilizing signal (even if the exact pairing is not important).

---

## Summary

Text-conditioning improves both:

- **Next-step prediction MSE** (1.7387 → 1.6236)
- **Long-horizon rollout stability** (Step 20 MSE: 1285.60 → 473.15)

However, the ablation results suggest the model currently treats text more like a global prior than a sample-specific conditioning signal.
