# Model Comparison: UNetClassic vs UNetClassicConditioned

## Experiment Setup

* **Training datasets:** `turbulent_radiative_layer_2D` + `active_matter`
* **Test dataset:** `helmholtz_staircase`
* **Resolution:** 128 × 384
* **Input timesteps (Ti):** 4
* **Output timesteps (To):** 1
* **Optimizer:** AdamW
* **LR Scheduler:** LinearWarmupCosineAnnealingLR
* **Evaluation:** Multi-step rollout validation

Two models were compared:

1. **UNetClassic** (baseline, no conditioning)
2. **UNetClassicConditioned** (PDE text embedding conditioning)

The conditioned model augments the input with a projected PDE embedding derived from dataset metadata (field names, resolution, boundary conditions).

---

# Overall Rollout Performance

| Model                  | Rollout Test Loss |
| ---------------------- | ----------------- |
| **UNetClassic**        | **0.15425**       |
| UNetClassicConditioned | 0.16617           |

**Observation:**
The baseline UNet slightly outperforms the conditioned model (~7–8% lower rollout loss).

---

# Field-wise Metrics (Example: `D_xx`)

| Metric    | UNetClassic | UNetClassicConditioned |
| --------- | ----------- | ---------------------- |
| L∞        | 0.30205     | 0.30968                |
| MSE       | 0.00699     | 0.00795                |
| RMSE      | 0.06996     | 0.07174                |
| NRMSE     | 221.23      | 226.87                 |
| Pearson R | 0           | 0                      |

**Observation:**

* All error metrics are slightly higher for the conditioned model.
* No improvement in correlation structure for this field.
* Differences are small but consistent.

---

# Model Capacity

| Model                  | Parameter Norm |
| ---------------------- | -------------- |
| UNetClassic            | 101.06         |
| UNetClassicConditioned | 113.86         |

The conditioned model has higher parameter norm due to the embedding projection layers, increasing capacity without improving performance in this experiment.

---

# Interpretation

### 1. Conditioning is Stable but Not Beneficial (Yet)

* PDE text conditioning does not degrade training stability.
* However, it does not improve short-horizon rollout accuracy.
* After 1 epoch, performance differences are marginal and slightly favor the baseline.

---

### 2. Possible Reasons

* Channel padding already provides implicit dataset identity.
* The model may distinguish datasets via input structure alone.
* Only 1 epoch of training may be insufficient for conditioning benefits to emerge.
* Conditioning advantages may appear in stronger OOD scenarios.

---







the-well-download --base-path "/Users/divyam/Course/Project Arbeit" --dataset active_matter

cd the_well/benchmark
python3 train.py experiment=unet_classic server=local data=turbulent_radiative_layer_2D
python3 train_multi_pde.py experiment=unet_classic_conditioned server=local data=turbulent_radiative_layer_2D

python3 train.py experiment=unet_classic server=local data=multi_dataset

epoch: 1
auto_resume: true/false