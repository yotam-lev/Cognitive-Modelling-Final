# parameter_recovery.py
# -------------------------------------------------
# VALIDATION: Parameter Recovery
# Receives parameters from the main analysis pipeline.

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import PyDDM classes and functions
from pyddm import Model, Sample, Fittable
from pyddm.models import (
    DriftConstant, NoiseConstant, BoundConstant, 
    OverlayNonDecision, ICPoint, ICRange, 
    OverlayNonDecisionUniform, BoundCollapsingExponential
)
from pyddm.functions import fit_adjust_model
from pyddm.models.loss import LossRobustLikelihood

# Import Custom Classes from modeling file to ensure consistency
from modeling_and_analysis import DriftDistractor, BoundDistractor, OverlayDistractor

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Setting undecided probability.*")
warnings.filterwarnings("ignore", category=UserWarning)

FIGURES_DIR = "generated_figures"

def run_parameter_recovery(best_model_obj):
    """
    Runs a parameter recovery simulation using the parameters from the 
    best fitted model object passed from the analysis phase.
    """
    if best_model_obj is None:
        print("No model object provided for recovery. Skipping.")
        return

    print("\n" + "="*60)
    print(" STARTING PARAMETER RECOVERY VALIDATION ")
    print("="*60)

    # 1. Extract Parameters from the Best Model
    # We use the 'fitted' values as the 'Ground Truth' for simulation
    params = best_model_obj.parameters()
    
    # Flatten the parameter dictionary for easier access
    true_params = {}
    for component, p_dict in params.items():
        for p_name, p_val in p_dict.items():
            true_params[p_name] = float(p_val)
            
    print(f"Ground Truth Parameters (from Best Fit):")
    for k, v in true_params.items():
        print(f"  {k}: {v:.4f}")

    # 2. Define the Generative Model
    # We must dynamically reconstruct the model using the same classes as the input model
    # Ideally, we would clone the model, but PyDDM models are tied to data.
    # We will instantiate a new model using the specific classes of the passed model object.
    
    print(f"Using Model Class Structure: {best_model_obj.name}")
    
    # We assume standard structure based on name for this assignment scope
    # (A fully generic cloner is complex; this covers the assignment models)
    
    if "Collapsing" in best_model_obj.name:
        gen_model = Model(name="Generator",
                          drift=DriftDistractor(v_dutch=true_params['v_dutch'], v_english=true_params['v_english']),
                          bound=BoundCollapsingExponential(B=true_params['B'], tau=true_params['tau']),
                          IC=ICPoint(x0=0.0),
                          overlay=OverlayNonDecision(nondectime=true_params['nondectime']),
                          noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
        
        # Recovery Model (Fittable)
        rec_model = Model(name="Recovery",
                          drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5), v_english=Fittable(minval=-5, maxval=5)),
                          bound=BoundCollapsingExponential(B=Fittable(minval=0.5, maxval=3), tau=Fittable(minval=0.1, maxval=5)),
                          IC=ICPoint(x0=0.0),
                          overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                          noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)

    elif "Drift-varying" in best_model_obj.name or "Drift" in best_model_obj.name:
        # Default Drift Varying
        gen_model = Model(name="Generator",
                          drift=DriftDistractor(v_dutch=true_params['v_dutch'], v_english=true_params['v_english']),
                          bound=BoundConstant(B=true_params['B']),
                          IC=ICPoint(x0=0.0), # Assuming point for basic drift varying
                          overlay=OverlayNonDecision(nondectime=true_params['nondectime']),
                          noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
                          
        rec_model = Model(name="Recovery",
                          drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5), v_english=Fittable(minval=-5, maxval=5)),
                          bound=BoundConstant(B=Fittable(minval=0.5, maxval=3)),
                          IC=ICPoint(x0=0.0),
                          overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                          noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
    else:
        print("Warning: Model type not fully supported for auto-recovery logic. Using Drift-Varying default.")
        return

    # 3. Simulate Data
    print("Simulating 2000 trials per condition...")
    sol_d = gen_model.solve(conditions={"distractor_language": "dutch"})
    samp_d = sol_d.resample(2000)
    sol_e = gen_model.solve(conditions={"distractor_language": "english"})
    samp_e = sol_e.resample(2000)
    synthetic_sample = samp_d + samp_e

    # 4. Recover Parameters
    print("Fitting recovery model to synthetic data...")
    fit_adjust_model(sample=synthetic_sample, model=rec_model, 
                     lossfunction=LossRobustLikelihood, verbose=False)

    # 5. Report Results
    print("\nRECOVERY RESULTS:")
    print(f"{'Param':<15} {'True':<10} {'Recovered':<10} {'Diff':<10}")
    print("-" * 45)
    
    rec_params = rec_model.parameters()
    rec_flat = {}
    for component, p_dict in rec_params.items():
        for p_name, p_val in p_dict.items():
            rec_flat[p_name] = float(p_val)

    # Plot lists
    p_names = []
    vals_true = []
    vals_rec = []

    for p_name, t_val in true_params.items():
        if p_name in rec_flat:
            r_val = rec_flat[p_name]
            diff = abs(t_val - r_val)
            print(f"{p_name:<15} {t_val:<10.3f} {r_val:<10.3f} {diff:<10.3f}")
            p_names.append(p_name)
            vals_true.append(t_val)
            vals_rec.append(r_val)

    # 6. Plot
    if not os.path.exists(FIGURES_DIR): os.makedirs(FIGURES_DIR)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(vals_true, vals_rec, c='blue', s=100, zorder=3)
    
    # Identity line
    min_v = min(min(vals_true), min(vals_rec)) * 0.9
    max_v = max(max(vals_true), max(vals_rec)) * 1.1
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5, zorder=2, label="Perfect Recovery")
    
    for i, txt in enumerate(p_names):
        plt.annotate(txt, (vals_true[i], vals_rec[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel("True Value (from Empirical Fit)")
    plt.ylabel("Recovered Value (from Simulation)")
    plt.title(f"Parameter Recovery: {best_model_obj.name}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure6_parameter_recovery.png"))
    print(f"Saved: {FIGURES_DIR}/figure6_parameter_recovery.png")
    plt.show()