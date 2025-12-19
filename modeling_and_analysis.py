# modeling_and_analysis.py
# -------------------------------------------------
# Module for PyDDM modeling (Phase 2).
# UPDATED: 
# 1. Added warning suppression for "undecided probability" messages.
# 2. Implements Option A: Subject-wise fitting loop.
# 3. Tighter parameter bounds for variability (sz, st).
# 4. New BIC calculation for nested/hierarchical data.
# 5. Added Null & Bound models to the comparison loop.
# 6. Added "Win Count" analysis.
# 7. BIC comparison now uses a line graph with zoomed y-axis.

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import warnings

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", message=".*Setting undecided probability.*")

# PyDDM Imports
try:
    from pyddm import Model, Sample, Fittable
    from pyddm.models import (
        DriftConstant, NoiseConstant, BoundConstant, 
        OverlayNonDecision, ICPoint, Drift, Bound,
        ICRange, OverlayNonDecisionUniform
    )
    from pyddm.functions import fit_adjust_model
    from pyddm.models.loss import LossRobustLikelihood 
    PYDDM_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: PyDDM is not installed. Run 'pip install pyddm'")
    PYDDM_AVAILABLE = False

# --- 1. Custom Drift/Bound Classes ---
class DriftDistractor(Drift):
    name = "Drift depends on distractor language"
    required_parameters = ["v_dutch", "v_english"]
    required_conditions = ["distractor_language"]

    def get_drift(self, x, t, conditions, **kwargs):
        lang = str(conditions.get("distractor_language", "")).lower()
        return self.v_dutch if lang == "dutch" else self.v_english


class BoundDistractor(Bound):
    name = "Bound depends on distractor language"
    required_parameters = ["a_dutch", "a_english"]
    required_conditions = ["distractor_language"]

    def get_bound(self, t, conditions, **kwargs):
        lang = str(conditions.get("distractor_language", "")).lower()
        return self.a_dutch if lang == "dutch" else self.a_english


# --- 2. Helper Functions ---

def compute_bic_with_grouping(nll, n_params, n_trials, n_subjects):
    """
    Compute BIC for hierarchical/nested data.
    Uses geometric mean of trials and subjects as effective sample size.
    """
    effective_n = np.sqrt(n_trials * n_subjects)
    return n_params * np.log(effective_n) + 2 * nll


def fit_model_safe(sample, model, verbose=False):
    """Wrapper to fit model and catch errors."""
    try:
        fitted = fit_adjust_model(sample=sample, model=model, 
                                   lossfunction=LossRobustLikelihood, verbose=verbose)
        return fitted
    except Exception as e:
        if verbose:
            print(f"  Fit failed: {e}")
        return None


# --- 3. Main Analysis Function ---

def run_ddm_analysis(df_model):
    if not PYDDM_AVAILABLE:
        print("Cannot run DDM analysis: PyDDM not installed.")
        return
    
    print(f"\nData ready for modeling. N={len(df_model)} trials.")
    
    # 1. Fit each subject separately
    unique_subjects = df_model['participant'].unique()
    n_subjects = len(unique_subjects)
    print(f"Starting Subject-wise Fitting for {n_subjects} participants...")
    
    # Store results
    results_by_subject = []
    win_counts = {}
    
    # Initialize global NLL trackers
    global_stats = {
        "Null Model":    {"nll": 0, "k": 0},
        "Drift-varying": {"nll": 0, "k": 0},
        "Bound-varying": {"nll": 0, "k": 0},
        "Drift + sz":    {"nll": 0, "k": 0},
        "Drift + st":    {"nll": 0, "k": 0},
        "Full DDM":      {"nll": 0, "k": 0}
    }
    
    # -- LOOP OVER SUBJECTS --
    for i, subject in enumerate(unique_subjects):
        print(f"Processing Subject {subject} ({i+1}/{n_subjects})...")
        
        subject_data = df_model[df_model['participant'] == subject].copy()
        subject_sample = Sample.from_pandas_dataframe(
            subject_data, 
            rt_column_name="rt", 
            choice_column_name="correct"
        )
        
        subject_bics = {}
        
        # 1. Null Model
        model_null = Model(name="Null Model",
                           drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
                           bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
                           IC=ICPoint(x0=0.0), 
                           overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                           noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
        fit_null = fit_model_safe(subject_sample, model_null)
        if fit_null:
            loss = LossRobustLikelihood(sample=subject_sample, model=fit_null, 
                                         required_conditions=[], dt=0.01, T_dur=3).loss(fit_null)
            k = len(fit_null.get_model_parameters())
            global_stats["Null Model"]["nll"] += loss
            global_stats["Null Model"]["k"] += k
            subject_bics["Null Model"] = k * np.log(len(subject_data)) + 2 * loss
        
        # 2. Drift-varying (Anchor)
        model_anchor = Model(name="Drift-varying",
                             drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5), 
                                                    v_english=Fittable(minval=-5, maxval=5)),
                             bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
                             IC=ICPoint(x0=0.0), 
                             overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                             noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
        fit_anchor = fit_model_safe(subject_sample, model_anchor)
        
        # 3. Bound-varying
        model_bound = Model(name="Bound-varying",
                            drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
                            bound=BoundDistractor(a_dutch=Fittable(minval=0.5, maxval=3.0), 
                                                   a_english=Fittable(minval=0.5, maxval=3.0)),
                            IC=ICPoint(x0=0.0), 
                            overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                            noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
        fit_bound = fit_model_safe(subject_sample, model_bound)
        if fit_bound:
            loss = LossRobustLikelihood(sample=subject_sample, model=fit_bound, 
                                         required_conditions=["distractor_language"], dt=0.01, T_dur=3).loss(fit_bound)
            k = len(fit_bound.get_model_parameters())
            global_stats["Bound-varying"]["nll"] += loss
            global_stats["Bound-varying"]["k"] += k
            subject_bics["Bound-varying"] = k * np.log(len(subject_data)) + 2 * loss
        
        # Process anchor model
        if fit_anchor:
            p = fit_anchor.parameters()
            best_v_dutch = float(p['drift']['v_dutch'])
            best_v_eng = float(p['drift']['v_english'])
            best_B = float(p['bound']['B'])
            best_t0 = float(p['overlay']['nondectime'])
            
            loss_anchor = LossRobustLikelihood(sample=subject_sample, model=fit_anchor, 
                                               required_conditions=["distractor_language"], dt=0.01, T_dur=3).loss(fit_anchor)
            k_anchor = len(fit_anchor.get_model_parameters())
            global_stats["Drift-varying"]["nll"] += loss_anchor
            global_stats["Drift-varying"]["k"] += k_anchor
            subject_bics["Drift-varying"] = k_anchor * np.log(len(subject_data)) + 2 * loss_anchor
            
            results_by_subject.append({
                "subject": subject,
                "model": "Drift-varying",
                "v_dutch": best_v_dutch,
                "v_english": best_v_eng,
                "B": best_B,
                "t0": best_t0
            })
            
            # 4. Drift + sz
            model_sz = Model(name="Drift + sz",
                             drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                                                   v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
                             bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=best_B)),
                             IC=ICRange(sz=Fittable(minval=0.0, maxval=0.2)),
                             overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5, default=best_t0)),
                             noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
            fit_sz = fit_model_safe(subject_sample, model_sz)
            if fit_sz:
                loss = LossRobustLikelihood(sample=subject_sample, model=fit_sz, 
                                             required_conditions=["distractor_language"], dt=0.01, T_dur=3).loss(fit_sz)
                k = len(fit_sz.get_model_parameters())
                global_stats["Drift + sz"]["nll"] += loss
                global_stats["Drift + sz"]["k"] += k
                subject_bics["Drift + sz"] = k * np.log(len(subject_data)) + 2 * loss
            
            # 5. Drift + st
            model_st = Model(name="Drift + st",
                             drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                                                   v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
                             bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=best_B)),
                             IC=ICPoint(x0=0.0),
                             overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.5, default=best_t0),
                                                               halfwidth=Fittable(minval=0.001, maxval=0.08)),
                             noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
            fit_st = fit_model_safe(subject_sample, model_st)
            if fit_st:
                loss = LossRobustLikelihood(sample=subject_sample, model=fit_st, 
                                             required_conditions=["distractor_language"], dt=0.01, T_dur=3).loss(fit_st)
                k = len(fit_st.get_model_parameters())
                global_stats["Drift + st"]["nll"] += loss
                global_stats["Drift + st"]["k"] += k
                subject_bics["Drift + st"] = k * np.log(len(subject_data)) + 2 * loss
            
            # 6. Full DDM
            model_full = Model(name="Full DDM",
                               drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                                                     v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
                               bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=best_B)),
                               IC=ICRange(sz=Fittable(minval=0.0, maxval=0.2)),
                               overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.5, default=best_t0),
                                                                 halfwidth=Fittable(minval=0.001, maxval=0.08)),
                               noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
            fit_full = fit_model_safe(subject_sample, model_full)
            if fit_full:
                loss = LossRobustLikelihood(sample=subject_sample, model=fit_full, 
                                             required_conditions=["distractor_language"], dt=0.01, T_dur=3).loss(fit_full)
                k = len(fit_full.get_model_parameters())
                global_stats["Full DDM"]["nll"] += loss
                global_stats["Full DDM"]["k"] += k
                subject_bics["Full DDM"] = k * np.log(len(subject_data)) + 2 * loss
        
        # Determine winner for this subject
        if subject_bics:
            winner = min(subject_bics, key=subject_bics.get)
            win_counts[winner] = win_counts.get(winner, 0) + 1
    
    # --- 4. Aggregate & Compare ---
    print("\n" + "="*60)
    print("GLOBAL MODEL COMPARISON (Aggregated over Subjects)")
    print("="*60)
    
    final_table = []
    total_trials = len(df_model)
    
    for m_name, stats in global_stats.items():
        if stats["nll"] == 0: 
            continue
        
        total_k = stats["k"] 
        bic_val = compute_bic_with_grouping(stats["nll"], total_k, total_trials, n_subjects)
        
        final_table.append({
            "Model": m_name,
            "Total NLL": stats["nll"],
            "Total Params (k)": total_k,
            "BIC (Grouped)": bic_val,
            "Subjects Won": win_counts.get(m_name, 0)
        })
    
    df_res = pd.DataFrame(final_table).sort_values("BIC (Grouped)")
    print(df_res.to_string(index=False))
    
    best_model_name = df_res.iloc[0]["Model"]
    print(f"\n*** Best Model: {best_model_name} ***")
    
    # --- 5. BIC Line Graph (Zoomed Y-Axis) ---
    if len(df_res) > 1:
        plt.figure(figsize=(10, 6))
        
        # Sort by BIC for consistent ordering
        df_plot = df_res.sort_values("BIC (Grouped)").reset_index(drop=True)
        
        # Extract values
        models = df_plot["Model"].tolist()
        bic_values = df_plot["BIC (Grouped)"].tolist()
        
        # Create x positions
        x_pos = np.arange(len(models))
        
        # Plot line with markers
        plt.plot(x_pos, bic_values, 'o-', linewidth=2, markersize=10, color='steelblue')
        
        # Highlight best model
        best_idx = 0  # Already sorted, so first is best
        plt.plot(x_pos[best_idx], bic_values[best_idx], 'o', markersize=15, 
                 color='green', label=f'Best: {models[best_idx]}', zorder=5)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(x_pos, bic_values)):
            plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9)
        
        # Set y-axis limits based on data range (zoomed in)
        bic_min = min(bic_values)
        bic_max = max(bic_values)
        bic_range = bic_max - bic_min
        padding = bic_range * 0.15 if bic_range > 0 else 10
        plt.ylim(bic_min - padding, bic_max + padding)
        
        # Labels and formatting
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.xlabel("Model")
        plt.ylabel("BIC (lower is better)")
        plt.title("Model Comparison: BIC Scores")
        plt.legend(loc='upper right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # --- 6. Delta BIC from Best Model ---
        plt.figure(figsize=(10, 6))
        
        delta_bic = [b - bic_min for b in bic_values]
        
        colors = ['green' if d == 0 else 'steelblue' for d in delta_bic]
        plt.bar(x_pos, delta_bic, color=colors)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(x_pos, delta_bic)):
            plt.annotate(f'Δ={y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0, 5), ha='center', fontsize=9)
        
        plt.xticks(x_pos, models, rotation=45, ha='right')
        plt.xlabel("Model")
        plt.ylabel("ΔBIC (relative to best model)")
        plt.title("Model Comparison: ΔBIC from Best Model")
        plt.tight_layout()
        plt.show()
    
    # --- 7. Visualizations (Subject Params) ---
    if results_by_subject:
        df_indiv = pd.DataFrame(results_by_subject)
        
        plt.figure(figsize=(8, 6))
        for i, row in df_indiv.iterrows():
            plt.plot([0, 1], [row['v_dutch'], row['v_english']], 'o-', color='grey', alpha=0.5)
        
        plt.plot([0, 1], [df_indiv['v_dutch'].mean(), df_indiv['v_english'].mean()], 
                 'o-', color='red', linewidth=3, label='Group Mean')
        
        plt.xticks([0, 1], ['Dutch', 'English'])
        plt.ylabel('Drift Rate (v)')
        plt.title('Individual Drift Rates (Subject-wise Fit)')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # T-Test
        t_stat, p_val = ttest_rel(df_indiv['v_english'], df_indiv['v_dutch'])
        print(f"\nPaired t-test (Drift): t({len(df_indiv)-1}) = {t_stat:.2f}, p = {p_val:.4f}")
    
    print("\n--- DDM Analysis Complete ---")