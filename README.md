## Results

Trained and evaluated two models for **1D PDE next-step prediction**:

- **Vision-only U-Net baseline** (`baseline.py`)
- **Text-conditioned U-Net** (`train_text_conditioned_unet_pde.py`) using **FiLM conditioning**

Both models predict the next PDE timestep given the previous **k = 5** timesteps as input.

---

## Vision-only baseline (U-Net)

**Test next-step MSE:** `1.6853`

**Rollout MSE (20 steps):**

| Step | MSE |
|------|-----|
| 1    | 1.69 |
| 5    | 74.87 |
| 10   | 340.37 |
| 15   | 722.39 |
| 20   | 1322.13 |

---

## Text-conditioned U-Net (FiLM)

**Test next-step MSE:** `1.9386`

**Rollout MSE (20 steps):**

| Step | MSE |
|------|-----|
| 1    | 1.94 |
| 5    | 125.95 |
| 10   | 642.56 |
| 15   | 1675.80 |
| 20   | 3517.97 |

---

## Text ablation study

We evaluated rollout performance under three text conditions:

- **Correct:** true text paired with each sample  
- **Shuffled:** text randomly permuted across samples  
- **Empty:** text replaced with an empty string  

**Results (rollout MSE):**

| Condition | Step 1 | Step 10 | Step 20 |
|----------|--------|---------|---------|
| Correct  | 1.85   | 480.93  | 2438.72 |
| Shuffled | 1.85   | 484.05  | 2441.80 |
| Empty    | 1.79   | 198.27  | 463.94  |
