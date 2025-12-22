# parameter_recovery.py
# -------------------------------------------------
# VALIDATION: Parameter Recovery for Drift-Varying Model

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyddm import Model, Sample, Fittable
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPoint, Drift
from pyddm.functions import fit_adjust_model
from pyddm.models.loss import LossRobustLikelihood
import warnings

warnings.filterwarnings("ignore", message=".*Setting undecided probability.*")
warnings.filterwarnings("ignore", message=".*This variable.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*This function.*is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Output Folder ---
FIGURES_DIR = "generated_figures"

def ensure_figures_dir():
    """Create the figures directory if it doesn't exist."""
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        print(f"Created directory: {FIGURES_DIR}")

# 1. Define the Ground Truth Model (The "Generator")
class DriftDistractor(Drift):
    name = "Drift depends on distractor language"
    required_parameters = ["v_dutch", "v_english"]
    required_conditions = ["distractor_language"]

    def get_drift(self, x, t, conditions, **kwargs):
        lang = str(conditions.get("distractor_language", "")).lower()
        return self.v_dutch if lang == "dutch" else self.v_english


# Define "True" values (close to what you found in your real data)
TRUE_PARAMS = {
    "v_dutch": 1.95,
    "v_english": 2.45,
    "B": 0.55,
    "nondectime": 0.17
}

# Create the generating model with fixed TRUE parameters
generator_model = Model(
    name="Generator",
    drift=DriftDistractor(v_dutch=TRUE_PARAMS["v_dutch"], v_english=TRUE_PARAMS["v_english"]),
    bound=BoundConstant(B=TRUE_PARAMS["B"]),
    IC=ICPoint(x0=0.0),
    overlay=OverlayNonDecision(nondectime=TRUE_PARAMS["nondectime"]),
    noise=NoiseConstant(noise=1),
    dx=0.005, dt=0.01, T_dur=3
)

# 2. Simulate Synthetic Data
print(f"Simulating synthetic trials using solve() and resample()...")

sol_dutch = generator_model.solve(conditions={"distractor_language": "dutch"})
dutch_sample = sol_dutch.resample(2500)

sol_english = generator_model.solve(conditions={"distractor_language": "english"})
english_sample = sol_english.resample(2500)

synthetic_sample = dutch_sample + english_sample

print(f"Generated {len(synthetic_sample)} trials successfully.")

# 3. Fit the Model to the Synthetic Data (The "Recovery")
recovery_model = Model(
    name="Recovery Fit",
    drift=DriftDistractor(v_dutch=Fittable(minval=0, maxval=5),
                          v_english=Fittable(minval=0, maxval=5)),
    bound=BoundConstant(B=Fittable(minval=0.3, maxval=1.5)),
    IC=ICPoint(x0=0.0),
    overlay=OverlayNonDecision(nondectime=Fittable(minval=0.05, maxval=0.5)),
    noise=NoiseConstant(noise=1),
    dx=0.005, dt=0.01, T_dur=3
)

print("Fitting model to synthetic data (Attempting Recovery)...")
fit_adjust_model(sample=synthetic_sample, model=recovery_model, 
                 lossfunction=LossRobustLikelihood, verbose=False)

# 4. Compare True vs Recovered
recovered_params = {
    "v_dutch": float(recovery_model.parameters()['drift']['v_dutch']),
    "v_english": float(recovery_model.parameters()['drift']['v_english']),
    "B": float(recovery_model.parameters()['bound']['B']),
    "nondectime": float(recovery_model.parameters()['overlay']['nondectime'])
}

print("\n--- Parameter Recovery Results ---")
print(f"{'Parameter':<15} {'True Value':<15} {'Recovered':<15} {'Difference':<15}")
print("-" * 60)
for key in TRUE_PARAMS:
    true_val = TRUE_PARAMS[key]
    rec_val = recovered_params[key]
    diff = abs(true_val - rec_val)
    print(f"{key:<15} {true_val:<15.4f} {rec_val:<15.4f} {diff:<15.4f}")

# 5. Visualization
ensure_figures_dir()

labels = list(TRUE_PARAMS.keys())
true_vals = [TRUE_PARAMS[k] for k in labels]
rec_vals = [recovered_params[k] for k in labels]

plt.figure(figsize=(6, 6))
plt.scatter(true_vals, rec_vals, c='blue', s=100, zorder=2)

# Draw identity line (perfect recovery)
min_v = min(min(true_vals), min(rec_vals)) * 0.9
max_v = max(max(true_vals), max(rec_vals)) * 1.1
plt.plot([min_v, max_v], [min_v, max_v], 'r--', label="Perfect Recovery", zorder=1)

for i, txt in enumerate(labels):
    plt.annotate(txt, (true_vals[i], rec_vals[i]), xytext=(5, 5), textcoords='offset points')

plt.xlabel("True Parameter Value")
plt.ylabel("Recovered Parameter Value")
plt.title("Parameter Recovery Check")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "figure6_parameter_recovery.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR}/figure6_parameter_recovery.png")
plt.show()