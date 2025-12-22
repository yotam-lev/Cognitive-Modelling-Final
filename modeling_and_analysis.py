# modeling_and_analysis.py
# -------------------------------------------------
# Module for PyDDM modeling (Phase 2).
# UPDATED:
# 1. Generates PPC plots for ALL models to visualize fit quality.
# 2. Exports custom classes for recovery.
# 3. Returns the best fitted model object.

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import warnings

# Set random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Output Folder ---
FIGURES_DIR = "generated_figures"

def ensure_figures_dir():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        print(f"Created directory: {FIGURES_DIR}")

warnings.filterwarnings("ignore", message=".*Setting undecided probability.*")

# PyDDM Imports
try:
    from pyddm import Model, Sample, Fittable
    from pyddm.models import (
        DriftConstant, NoiseConstant, BoundConstant, 
        OverlayNonDecision, ICPoint, Drift, Bound,
        ICRange, OverlayNonDecisionUniform, BoundCollapsingExponential
    )
    from pyddm.functions import fit_adjust_model
    from pyddm.models.loss import LossRobustLikelihood 
    PYDDM_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: PyDDM is not installed. Run 'pip install pyddm'")
    PYDDM_AVAILABLE = False

# --- 1. Custom Classes (Exportable) ---

class DriftDistractor(Drift):
    name = "Drift depends on distractor language"
    required_parameters = ["v_dutch", "v_english"]
    required_conditions = ["distractor_language"]

    def get_drift(self, x, t, conditions, **kwargs):
        lang = str(conditions.get("distractor_language", "")).lower().strip()
        return self.v_dutch if lang == "dutch" else self.v_english

class BoundDistractor(Bound):
    name = "Bound depends on distractor language"
    required_parameters = ["a_dutch", "a_english"]
    required_conditions = ["distractor_language"]

    def get_bound(self, t, conditions, **kwargs):
        lang = str(conditions.get("distractor_language", "")).lower().strip()
        return self.a_dutch if lang == "dutch" else self.a_english

class OverlayDistractor(OverlayNonDecision):
    name = "Non-decision time depends on distractor"
    required_parameters = ["t0_dutch", "t0_english"]
    required_conditions = ["distractor_language"]

    def get_nondecision_time(self, conditions):
        lang = str(conditions.get("distractor_language", "")).lower().strip()
        return self.t0_dutch if lang == "dutch" else self.t0_english

# --- 2. Helper Functions ---

def compute_bic_with_grouping(nll, n_params, n_trials, n_subjects):
    """BIC for hierarchical/nested data."""
    effective_n = np.sqrt(n_trials * n_subjects)
    return n_params * np.log(effective_n) + 2 * nll

def fit_model_safe(sample, model, verbose=False):
    """Wrapper to fit model and catch errors."""
    try:
        # Using a slightly larger dx for speed in assignment context
        fitted = fit_adjust_model(sample=sample, model=model, 
                                   lossfunction=LossRobustLikelihood, verbose=verbose)
        return fitted
    except Exception as e:
        if verbose: print(f"  Fit failed for {model.name}: {e}")
        return None

def plot_ppc_quantile_probability(model, df_data, model_label=None):
    """
    Generates a Standard Quantile-Probability (QP) Plot.
    Shows RT quantiles (x-axis) vs cumulative probability (y-axis).
    For correct responses: quantiles at 0.1, 0.3, 0.5, 0.7, 0.9 of correct RT distribution
    For error responses: quantiles at 0.1, 0.3, 0.5, 0.7, 0.9 of error RT distribution
    
    The y-axis represents: P(correct) * quantile_position for correct responses
                          P(error) * quantile_position for error responses
    """
    if model_label is None:
        model_label = model.name
        
    print(f"Generating Corrected QP Plot for {model_label}...")
    
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    conditions = ["dutch", "english"]
    colors = {"dutch": "blue", "english": "orange"}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for cond in conditions:
        subset = df_data[df_data["distractor_language"] == cond]
        
        # Calculate overall accuracy for this condition
        n_total = len(subset)
        n_correct = (subset["correct"] == 1).sum()
        n_error = (subset["correct"] == 0).sum()
        p_correct_data = n_correct / n_total
        p_error_data = n_error / n_total
        
        # --- 1. REAL DATA (Dots) ---
        # Correct responses
        if n_correct >= 10:
            rt_correct = subset[subset["correct"] == 1]["rt"].values
            q_data_correct = np.quantile(rt_correct, quantiles)
            # Y-values: scale quantile positions by overall P(correct)
            y_data_correct = np.array(quantiles) * p_correct_data
            
            ax.scatter(q_data_correct, y_data_correct, 
                       color=colors[cond], marker='o', s=80, alpha=0.7, 
                       edgecolors='black', linewidths=1.5,
                       label=f"Data {cond.capitalize()} Correct")
        
        # Error responses
        if n_error >= 10:
            rt_error = subset[subset["correct"] == 0]["rt"].values
            q_data_error = np.quantile(rt_error, quantiles)
            # Y-values: scale quantile positions by overall P(error)
            y_data_error = np.array(quantiles) * p_error_data
            
            ax.scatter(q_data_error, y_data_error, 
                       color=colors[cond], marker='o', s=80, alpha=0.7, 
                       edgecolors='black', linewidths=1.5, facecolors='none',
                       label=f"Data {cond.capitalize()} Error")

        # --- 2. MODEL PREDICTIONS (X markers with lines) ---
        sol = model.solve(conditions={"distractor_language": cond})
        
        p_correct_model = sol.prob("correct")
        p_error_model = sol.prob("error")
        
        # Generate synthetic samples for extracting quantiles
        # Use the solution's PDF directly to compute quantiles
        t_domain = sol.model.t_domain()
        dt = sol.model.dt
        
        # For correct responses
        if p_correct_model > 0.01:
            # Get PDF for correct responses
            pdf_correct = sol.pdf("correct")
            # Normalize to create proper probability distribution
            pdf_correct_norm = pdf_correct / (np.sum(pdf_correct) * dt)
            
            # Compute CDF
            cdf_correct = np.cumsum(pdf_correct_norm) * dt
            
            # Find quantiles by interpolation
            q_model_correct = np.interp(quantiles, cdf_correct, t_domain)
            y_model_correct = np.array(quantiles) * p_correct_model
            
            ax.plot(q_model_correct, y_model_correct, 
                    color=colors[cond], marker='x', markersize=10, 
                    linestyle='-', linewidth=2, alpha=0.8,
                    label=f"Model {cond.capitalize()} Correct")
        
        # For error responses
        if p_error_model > 0.01:
            # Get PDF for error responses
            pdf_error = sol.pdf("error")
            # Normalize to create proper probability distribution
            pdf_error_norm = pdf_error / (np.sum(pdf_error) * dt)
            
            # Compute CDF
            cdf_error = np.cumsum(pdf_error_norm) * dt
            
            # Find quantiles by interpolation
            q_model_error = np.interp(quantiles, cdf_error, t_domain)
            y_model_error = np.array(quantiles) * p_error_model
            
            ax.plot(q_model_error, y_model_error, 
                    color=colors[cond], marker='x', markersize=10, 
                    linestyle='--', linewidth=2, alpha=0.8,
                    label=f"Model {cond.capitalize()} Error")

    plt.xlabel("Reaction Time (s)", fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    plt.title(f"Quantile-Probability Plot: {model_label}\n(Circles=Data, X=Model; Filled=Correct, Open=Error)", 
              fontsize=13)
    
    # Legend
    plt.legend(loc='best', fontsize=9, framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(0.2, 1.2)  # Focus on typical RT range
    plt.ylim(0, 1)      # Probability range
    plt.tight_layout()
    
    clean_name = model_label.replace(" ", "_").replace("+", "plus")
    filename = f"figure7_ppc_{clean_name}.png"
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    print(f"Saved: {FIGURES_DIR}/{filename}")
    plt.close()
    
            
def print_simulated_vs_real_comparison(fitted_models, df_data):
    """Prints a comparison table of Real Data vs. Predictions."""
    print("\n" + "="*95)
    print(f" MODEL SIMULATION vs REAL DATA (Subject 1) ")
    print("="*95)
    
    conditions = ["dutch", "english"]
    
    # Real Data Stats
    real_stats = {}
    for cond in conditions:
        sub = df_data[df_data["distractor_language"] == cond]
        corr_rts = sub[sub["correct"] == 1]["rt"]
        real_stats[cond] = {
            "Acc": sub["correct"].mean(),
            "MRT": corr_rts.mean() if not corr_rts.empty else 0
        }
        
    print(f"{'Model':<20} | {'D_Acc':<6} {'D_MRT':<6} | {'E_Acc':<6} {'E_MRT':<6} | {'Error (SSE)':<10}")
    print("-" * 95)
    
    print(f"{'REAL DATA':<20} | {real_stats['dutch']['Acc']:.3f}  {real_stats['dutch']['MRT']:.3f}  | "
          f"{real_stats['english']['Acc']:.3f}  {real_stats['english']['MRT']:.3f}  | {'-':<10}")
    print("-" * 95)
    
    for m_name, model in fitted_models.items():
        if model is None: continue
        row_str = f"{m_name:<20} | "
        total_error = 0
        
        for cond in conditions:
            sol = model.solve(conditions={"distractor_language": cond})
            pred_acc = sol.prob("correct")
            if pred_acc > 0.001:
                pdf_norm = sol.pdf("correct") / pred_acc
                pred_mrt = np.sum(sol.t_domain * pdf_norm) * model.dt 
            else:
                pred_mrt = 0.0
            
            err = (pred_acc - real_stats[cond]['Acc'])**2 + (pred_mrt - real_stats[cond]['MRT'])**2
            total_error += err
            row_str += f"{pred_acc:.3f}  {pred_mrt:.3f}  | "
            
        print(f"{row_str} {total_error:.4f}")
    print("="*95 + "\n")

# --- 3. Main Analysis Function ---

def run_ddm_analysis(df_model):
    if not PYDDM_AVAILABLE:
        print("Cannot run DDM analysis: PyDDM not installed.")
        return None
    
    ensure_figures_dir()
    print(f"\nData ready for modeling. N={len(df_model)} trials.")
    
    unique_subjects = df_model['participant'].unique()
    n_subjects = len(unique_subjects)
    print(f"Starting Subject-wise Fitting for {n_subjects} participants...")
    
    results_by_subject = []
    win_counts = {}
    
    # Dictionary to store ALL fitted models for Subject 1
    subj1_models = {}
    
    # Global stats tracker
    global_stats = {k: {"nll": 0, "k": 0} for k in [
        "Null Model", "Drift-varying", "Bound-varying", "t0-varying", 
        "Collapsing Bound", "Drift + sz", "Drift + st", "Full DDM"
    ]}
    
    # -- LOOP OVER SUBJECTS --
    for i, subject in enumerate(unique_subjects):
        print(f"Processing Subject {subject} ({i+1}/{n_subjects})...")
        
        subject_data = df_model[df_model['participant'] == subject].copy()
        subject_data["distractor_language"] = subject_data["distractor_language"].astype(str).str.lower().str.strip()
        
        subject_sample = Sample.from_pandas_dataframe(
            subject_data, rt_column_name="rt", choice_column_name="correct"
        )
        
        subject_bics = {}
        
        # --- Define Models ---
        # 1. Null
        m_null = Model(name="Null Model",
                       drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
                       bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
                       IC=ICPoint(x0=0.0), overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                       noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
        
        # 2. Drift
        m_drift = Model(name="Drift-varying",
                        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5), v_english=Fittable(minval=-5, maxval=5)),
                        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
                        IC=ICPoint(x0=0.0), overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                        noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
        
        # 3. Bound
        m_bound = Model(name="Bound-varying",
                        drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
                        bound=BoundDistractor(a_dutch=Fittable(minval=0.5, maxval=3.0), a_english=Fittable(minval=0.5, maxval=3.0)),
                        IC=ICPoint(x0=0.0), overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5)),
                        noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)

        # 4. t0
        m_t0 = Model(name="t0-varying",
                     drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
                     bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
                     IC=ICPoint(x0=0.0), 
                     overlay=OverlayDistractor(t0_dutch=Fittable(minval=0.1, maxval=0.5), t0_english=Fittable(minval=0.1, maxval=0.5)),
                     noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)

        # --- Fit Basic Models ---
        fit_null = fit_model_safe(subject_sample, m_null)
        fit_drift = fit_model_safe(subject_sample, m_drift)
        fit_bound = fit_model_safe(subject_sample, m_bound)
        fit_t0 = fit_model_safe(subject_sample, m_t0)

        # Helper to log results
        def log_fit(fit_obj, name):
            if fit_obj:
                # All models except Null require conditions due to our setup
                req_conds = [] if name == "Null Model" else ["distractor_language"]
                
                loss = LossRobustLikelihood(sample=subject_sample, model=fit_obj, 
                                            required_conditions=req_conds,
                                            dt=0.01, T_dur=3).loss(fit_obj)
                k = len(fit_obj.get_model_parameters())
                global_stats[name]["nll"] += loss
                global_stats[name]["k"] += k
                subject_bics[name] = k * np.log(len(subject_data)) + 2 * loss
                if i == 0: subj1_models[name] = fit_obj
        
        log_fit(fit_null, "Null Model")
        log_fit(fit_drift, "Drift-varying")
        log_fit(fit_bound, "Bound-varying")
        log_fit(fit_t0, "t0-varying")

        # --- Advanced Models (using Drift Anchor) ---
        if fit_drift:
            p = fit_drift.parameters()
            def_v_d = float(p['drift']['v_dutch'])
            def_v_e = float(p['drift']['v_english'])
            def_B = float(p['bound']['B'])
            def_t0 = float(p['overlay']['nondectime'])
            
            results_by_subject.append({
                "subject": subject, "v_dutch": def_v_d, "v_english": def_v_e, "B": def_B, "t0": def_t0
            })

            # 5. Collapsing
            m_col = Model(name="Collapsing Bound",
                          drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=def_v_d), 
                                                v_english=Fittable(minval=-5, maxval=5, default=def_v_e)),
                          bound=BoundCollapsingExponential(B=Fittable(minval=0.5, maxval=3.0, default=def_B), 
                                                           tau=Fittable(minval=0.1, maxval=5.0)),
                          IC=ICPoint(x0=0.0), overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5, default=def_t0)),
                          noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)
            
            # 6. Drift + sz
            m_sz = Model(name="Drift + sz",
                         drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=def_v_d), 
                                               v_english=Fittable(minval=-5, maxval=5, default=def_v_e)),
                         bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=def_B)),
                         IC=ICRange(sz=Fittable(minval=0.0, maxval=0.2)), 
                         overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5, default=def_t0)),
                         noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)

            # 7. Drift + st
            m_st = Model(name="Drift + st",
                         drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=def_v_d), 
                                               v_english=Fittable(minval=-5, maxval=5, default=def_v_e)),
                         bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=def_B)),
                         IC=ICPoint(x0=0.0), 
                         overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.5, default=def_t0),
                                                           halfwidth=Fittable(minval=0.001, maxval=0.08)),
                         noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)

            # 8. Full
            m_full = Model(name="Full DDM",
                           drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=def_v_d), 
                                                 v_english=Fittable(minval=-5, maxval=5, default=def_v_e)),
                           bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=def_B)),
                           IC=ICRange(sz=Fittable(minval=0.0, maxval=0.2)),
                           overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.5, default=def_t0),
                                                             halfwidth=Fittable(minval=0.001, maxval=0.08)),
                           noise=NoiseConstant(noise=1), dx=0.005, dt=0.01, T_dur=3)

            fit_col = fit_model_safe(subject_sample, m_col)
            fit_sz = fit_model_safe(subject_sample, m_sz)
            fit_st = fit_model_safe(subject_sample, m_st)
            fit_full = fit_model_safe(subject_sample, m_full)

            log_fit(fit_col, "Collapsing Bound")
            log_fit(fit_sz, "Drift + sz")
            log_fit(fit_st, "Drift + st")
            log_fit(fit_full, "Full DDM")

        # Winner for this subject
        if subject_bics:
            winner = min(subject_bics, key=subject_bics.get)
            win_counts[winner] = win_counts.get(winner, 0) + 1

    # --- 4. Comparison Results ---
    print("\n" + "="*60)
    print("GLOBAL MODEL COMPARISON")
    print("="*60)
    
    final_table = []
    total_trials = len(df_model)
    
    for m_name, stats in global_stats.items():
        if stats["nll"] == 0: continue
        bic_val = compute_bic_with_grouping(stats["nll"], stats["k"], total_trials, n_subjects)
        final_table.append({
            "Model": m_name, "Total NLL": stats["nll"], "k": stats["k"], 
            "BIC (Grouped)": bic_val, "Subjects Won": win_counts.get(m_name, 0)
        })
    
    df_res = pd.DataFrame(final_table).sort_values("BIC (Grouped)")
    print(df_res.to_string(index=False))
    
    best_model_name = df_res.iloc[0]["Model"]
    print(f"\n*** Best Model: {best_model_name} ***")

    # --- 5. Plots ---
    if len(df_res) > 1:
        plt.figure(figsize=(10, 6))
        df_plot = df_res.sort_values("BIC (Grouped)").reset_index(drop=True)
        plt.plot(np.arange(len(df_plot)), df_plot["BIC (Grouped)"], 'o-', color='steelblue')
        plt.xticks(np.arange(len(df_plot)), df_plot["Model"], rotation=45, ha='right')
        plt.ylabel("BIC")
        plt.title("Model Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure3_bic_comparison.png"), dpi=150)
        plt.show()

    if results_by_subject:
        df_indiv = pd.DataFrame(results_by_subject)
        plt.figure(figsize=(8, 6))
        for i, row in df_indiv.iterrows():
            plt.plot([0, 1], [row['v_dutch'], row['v_english']], 'o-', color='grey', alpha=0.5)
        plt.plot([0, 1], [df_indiv['v_dutch'].mean(), df_indiv['v_english'].mean()], 'o-', color='red', linewidth=3)
        plt.xticks([0, 1], ['Dutch', 'English'])
        plt.ylabel('Drift Rate (v)')
        plt.title('Individual Drift Rates')
        plt.savefig(os.path.join(FIGURES_DIR, "figure5_individual_drift_rates.png"), dpi=150)
        plt.show()
        
        t_stat, p_val = ttest_rel(df_indiv['v_english'], df_indiv['v_dutch'])
        print(f"\nPaired t-test (Drift): t({len(df_indiv)-1}) = {t_stat:.2f}, p = {p_val:.4f}")

    # --- 6. Detailed Analysis of Representative Subject (Subj 1) ---
    subj1_data = df_model[df_model['participant'] == unique_subjects[0]].copy()
    subj1_data["distractor_language"] = subj1_data["distractor_language"].astype(str).str.lower().str.strip()
    
    print_simulated_vs_real_comparison(subj1_models, subj1_data)
    
    # --- 7. Generate PPC Plots for ALL Fitted Models ---
    print("\nGenerating PPC Plots for all models...")
    for m_name, m_obj in subj1_models.items():
        if m_obj:
            plot_ppc_quantile_probability(m_obj, subj1_data, model_label=m_name)
    
    # Return best model object for subject 1 (closest proxy to global best for parameter recovery)
    best_model_obj = subj1_models.get(best_model_name, None)
    return best_model_obj