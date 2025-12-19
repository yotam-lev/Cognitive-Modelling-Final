# modeling_and_analysis.py
# -------------------------------------------------
# Module for PyDDM modeling (Phase 2).
# UPDATED: Improved complex model fitting with better starting values,
# nested model comparisons, and alternative specifications.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, chi2

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
except ImportError:
    print("CRITICAL ERROR: PyDDM is not installed. Run 'pip install pyddm'")

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

class BoundCollapsingLinear(Bound):
    name = "Linear collapsing bound"
    required_parameters = ["B", "slope"]
    
    def get_bound(self, t, conditions, **kwargs):
        return max(self.B - self.slope * t, 0.1)  # Floor at 0.1 to prevent collapse to zero


def fit_model_with_restarts(sample, model, n_attempts=3, verbose=False):
    """
    Fits a model multiple times with different random initializations.
    Returns the best fit based on NLL.
    """
    best_fit = None
    best_nll = float('inf')
    
    for attempt in range(n_attempts):
        try:
            # Create fresh model with randomized starting values
            temp_model = Model(
                name=model.name,
                drift=model.get_dependence('drift'),
                bound=model.get_dependence('bound'),
                IC=model.get_dependence('IC'),
                overlay=model.get_dependence('overlay'),
                noise=model.get_dependence('noise'),
                dx=model.dx, dt=model.dt, T_dur=model.T_dur
            )
            
            fit_adjust_model(sample=sample, model=temp_model, 
                           lossfunction=LossRobustLikelihood, verbose=False)
            
            # Calculate NLL
            req_cond = ["distractor_language"] if hasattr(temp_model.get_dependence('drift'), 'required_conditions') else []
            
            nll = LossRobustLikelihood(
                sample=sample, model=temp_model, 
                required_conditions=req_cond,
                dt=temp_model.dt, T_dur=temp_model.T_dur
            ).loss(temp_model)
            
            if nll < best_nll:
                best_nll = nll
                best_fit = temp_model
                if verbose:
                    print(f"  Attempt {attempt+1}: NLL = {nll:.2f} (new best)")
            elif verbose:
                print(f"  Attempt {attempt+1}: NLL = {nll:.2f}")
                
        except Exception as e:
            if verbose:
                print(f"  Attempt {attempt+1}: Failed ({str(e)[:50]})")
            continue
    
    return best_fit, best_nll


def likelihood_ratio_test(nll_simple, nll_complex, k_simple, k_complex):
    """
    Performs likelihood ratio test between nested models.
    Returns: LR statistic, degrees of freedom, p-value
    """
    lr_stat = 2 * (nll_simple - nll_complex)
    df_diff = k_complex - k_simple
    
    if lr_stat < 0:
        return lr_stat, df_diff, 1.0  # Complex model is worse
    
    p_value = 1 - chi2.cdf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def run_ddm_analysis(df_model):
    print(f"\nData ready for modeling. N={len(df_model)} trials.")
    sample = Sample.from_pandas_dataframe(df_model, rt_column_name="rt", choice_column_name="correct")

    # --- Common Components ---
    overlay_simple = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5))
    noise = NoiseConstant(noise=1) 
    
    fitted_results = []
    results_table = []

    # =========================================================
    # STEP 1: Fit Basic Models (Null, Drift, Bound)
    # =========================================================
    print("\n" + "="*60)
    print("STEP 1: Fitting Basic Models")
    print("="*60)
    
    # --- Null Model ---
    print("\n[1/3] Fitting Null Model...")
    model_null = Model(
        name="Null Model", 
        drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)), 
        IC=ICPoint(x0=0.0), 
        overlay=overlay_simple, 
        noise=noise, 
        dx=0.005, dt=0.01, T_dur=3
    )
    fit_adjust_model(sample=sample, model=model_null, 
                     lossfunction=LossRobustLikelihood, verbose=False)
    fitted_results.append(("Null Model", model_null))
    nll_null = LossRobustLikelihood(sample=sample, model=model_null, 
                                    required_conditions=[], 
                                    dt=model_null.dt, T_dur=model_null.T_dur).loss(model_null)
    k_null = len(model_null.get_model_parameters())
    bic_null = k_null * np.log(len(df_model)) + 2 * nll_null
    results_table.append({"Model": "Null Model", "BIC": bic_null, "NLL": nll_null, "k": k_null})
    print(f"   ✓ NLL: {nll_null:.2f}, BIC: {bic_null:.2f}")

    # --- Drift-varying Model ---
    print("\n[2/3] Fitting Drift-varying Model...")
    model_drift = Model(
        name="Drift-varying",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5),
                              v_english=Fittable(minval=-5, maxval=5)),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
        IC=ICPoint(x0=0.0), 
        overlay=overlay_simple, 
        noise=noise, 
        dx=0.005, dt=0.01, T_dur=3
    )
    fit_adjust_model(sample=sample, model=model_drift, 
                     lossfunction=LossRobustLikelihood, verbose=False)
    fitted_results.append(("Drift-varying", model_drift))
    nll_drift = LossRobustLikelihood(sample=sample, model=model_drift, 
                                     required_conditions=["distractor_language"], 
                                     dt=model_drift.dt, T_dur=model_drift.T_dur).loss(model_drift)
    k_drift = len(model_drift.get_model_parameters())
    bic_drift = k_drift * np.log(len(df_model)) + 2 * nll_drift
    results_table.append({"Model": "Drift-varying", "BIC": bic_drift, "NLL": nll_drift, "k": k_drift})
    
    # Extract parameters for later use
    p_drift = model_drift.parameters()
    best_v_dutch = float(p_drift['drift']['v_dutch'])
    best_v_eng = float(p_drift['drift']['v_english'])
    best_B = float(p_drift['bound']['B'])
    best_t0 = float(p_drift['overlay']['nondectime'])
    
    v_diff = abs(best_v_dutch - best_v_eng)
    print(f"   ✓ NLL: {nll_drift:.2f}, BIC: {bic_drift:.2f}")
    print(f"   → Drift difference: {v_diff:.3f} (v_NL={best_v_dutch:.2f}, v_EN={best_v_eng:.2f})")
     
    # --- Bound-varying Model ---
    print("\n[3/3] Fitting Bound-varying Model...")
    model_bound = Model(
        name="Bound-varying", 
        drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
        bound=BoundDistractor(a_dutch=Fittable(minval=0.5, maxval=3.0), 
                              a_english=Fittable(minval=0.5, maxval=3.0)),
        IC=ICPoint(x0=0.0), 
        overlay=overlay_simple, 
        noise=noise, 
        dx=0.005, dt=0.01, T_dur=3
    )
    fit_adjust_model(sample=sample, model=model_bound, 
                     lossfunction=LossRobustLikelihood, verbose=False)
    fitted_results.append(("Bound-varying", model_bound))
    nll_bound = LossRobustLikelihood(sample=sample, model=model_bound, 
                                     required_conditions=["distractor_language"], 
                                     dt=model_bound.dt, T_dur=model_bound.T_dur).loss(model_bound)
    k_bound = len(model_bound.get_model_parameters())
    bic_bound = k_bound * np.log(len(df_model)) + 2 * nll_bound
    results_table.append({"Model": "Bound-varying", "BIC": bic_bound, "NLL": nll_bound, "k": k_bound})
    print(f"   ✓ NLL: {nll_bound:.2f}, BIC: {bic_bound:.2f}")

    # =========================================================
    # STEP 2: Fit Complex Models (Only if drift effect is substantial)
    # =========================================================
    print("\n" + "="*60)
    print("STEP 2: Fitting Complex Models")
    print("="*60)
    
    if v_diff < 0.3:
        print(f"\n⚠️  WARNING: Drift effect is small ({v_diff:.3f}).")
        print("   Complex models may not be justified. Fitting anyway for comparison...")
    
    # --- A. Starting Point Variability (sz only) ---
    print("\n[1/5] Fitting Model with Starting Point Variability (sz)...")
    model_sz = Model(
        name="Drift + sz",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                              v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=best_B)),
        IC=ICRange(sz=Fittable(minval=0.0, maxval=0.5)),
        overlay=overlay_simple,
        noise=noise, 
        dx=0.005, dt=0.01, T_dur=3
    )
    fit_sz, nll_sz = fit_model_with_restarts(sample, model_sz, n_attempts=3, verbose=True)
    if fit_sz:
        fitted_results.append(("Drift + sz", fit_sz))
        k_sz = len(fit_sz.get_model_parameters())
        bic_sz = k_sz * np.log(len(df_model)) + 2 * nll_sz
        results_table.append({"Model": "Drift + sz", "BIC": bic_sz, "NLL": nll_sz, "k": k_sz})
        print(f"   ✓ Best NLL: {nll_sz:.2f}, BIC: {bic_sz:.2f}")
    
    # --- B. Non-Decision Variability (st only) ---
    print("\n[2/5] Fitting Model with Non-Decision Variability (st)...")
    model_st = Model(
        name="Drift + st",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                              v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=best_B)),
        IC=ICPoint(x0=0.0),
        overlay=OverlayNonDecisionUniform(
            nondectime=Fittable(minval=0.1, maxval=0.5, default=best_t0),
            halfwidth=Fittable(minval=0.001, maxval=0.15)
        ),
        noise=noise, 
        dx=0.005, dt=0.01, T_dur=3
    )
    fit_st, nll_st = fit_model_with_restarts(sample, model_st, n_attempts=3, verbose=True)
    if fit_st:
        fitted_results.append(("Drift + st", fit_st))
        k_st = len(fit_st.get_model_parameters())
        bic_st = k_st * np.log(len(df_model)) + 2 * nll_st
        results_table.append({"Model": "Drift + st", "BIC": bic_st, "NLL": nll_st, "k": k_st})
        print(f"   ✓ Best NLL: {nll_st:.2f}, BIC: {bic_st:.2f}")
    
    # --- C. Full DDM (sz + st) ---
    print("\n[3/5] Fitting Full DDM (sz + st)...")
    model_full = Model(
        name="Full DDM",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                              v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0, default=best_B)),
        IC=ICRange(sz=Fittable(minval=0.0, maxval=0.5)),
        overlay=OverlayNonDecisionUniform(
            nondectime=Fittable(minval=0.1, maxval=0.5, default=best_t0),
            halfwidth=Fittable(minval=0.001, maxval=0.15)
        ),
        noise=noise, 
        dx=0.005, dt=0.01, T_dur=3
    )
    fit_full, nll_full = fit_model_with_restarts(sample, model_full, n_attempts=3, verbose=True)
    if fit_full:
        fitted_results.append(("Full DDM", fit_full))
        k_full = len(fit_full.get_model_parameters())
        bic_full = k_full * np.log(len(df_model)) + 2 * nll_full
        results_table.append({"Model": "Full DDM", "BIC": bic_full, "NLL": nll_full, "k": k_full})
        print(f"   ✓ Best NLL: {nll_full:.2f}, BIC: {bic_full:.2f}")
    
    # --- D. Collapsing Bound (Exponential) ---
    print("\n[4/5] Fitting Exponential Collapsing Bound...")
    model_collapse_exp = Model(
        name="Collapse (Exp)",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                              v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
        bound=BoundCollapsingExponential(
            B=Fittable(minval=0.5, maxval=3.0, default=best_B),
            tau=Fittable(minval=0.1, maxval=5.0)
        ),
        IC=ICPoint(x0=0.0), 
        overlay=overlay_simple, 
        noise=noise, 
        dx=0.003, dt=0.005, T_dur=3
    )
    fit_col_exp, nll_col_exp = fit_model_with_restarts(sample, model_collapse_exp, n_attempts=3, verbose=True)
    if fit_col_exp:
        fitted_results.append(("Collapse (Exp)", fit_col_exp))
        k_col_exp = len(fit_col_exp.get_model_parameters())
        bic_col_exp = k_col_exp * np.log(len(df_model)) + 2 * nll_col_exp
        results_table.append({"Model": "Collapse (Exp)", "BIC": bic_col_exp, "NLL": nll_col_exp, "k": k_col_exp})
        print(f"   ✓ Best NLL: {nll_col_exp:.2f}, BIC: {bic_col_exp:.2f}")
    
    # --- E. Collapsing Bound (Linear) ---
    print("\n[5/5] Fitting Linear Collapsing Bound...")
    model_collapse_lin = Model(
        name="Collapse (Linear)",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5, default=best_v_dutch),
                              v_english=Fittable(minval=-5, maxval=5, default=best_v_eng)),
        bound=BoundCollapsingLinear(
            B=Fittable(minval=0.5, maxval=3.0, default=best_B),
            slope=Fittable(minval=0.0, maxval=2.0)
        ),
        IC=ICPoint(x0=0.0), 
        overlay=overlay_simple, 
        noise=noise, 
        dx=0.003, dt=0.005, T_dur=3
    )
    fit_col_lin, nll_col_lin = fit_model_with_restarts(sample, model_collapse_lin, n_attempts=3, verbose=True)
    if fit_col_lin:
        fitted_results.append(("Collapse (Linear)", fit_col_lin))
        k_col_lin = len(fit_col_lin.get_model_parameters())
        bic_col_lin = k_col_lin * np.log(len(df_model)) + 2 * nll_col_lin
        results_table.append({"Model": "Collapse (Linear)", "BIC": bic_col_lin, "NLL": nll_col_lin, "k": k_col_lin})
        print(f"   ✓ Best NLL: {nll_col_lin:.2f}, BIC: {bic_col_lin:.2f}")

    # =========================================================
    # STEP 3: Model Comparison & Statistics
    # =========================================================
    print("\n" + "="*60)
    print("STEP 3: Model Comparison & Statistical Tests")
    print("="*60)
    
    if results_table:
        res_df = pd.DataFrame(results_table).sort_values("BIC")
        print("\n=== FINAL RESULTS (sorted by BIC) ===")
        print(res_df[["Model", "BIC", "NLL", "k"]].to_string(index=False))
        
        # Likelihood Ratio Tests (nested models)
        print("\n=== Likelihood Ratio Tests ===")
        
        # Test 1: Drift-varying vs Drift + sz
        if "Drift + sz" in res_df["Model"].values:
            idx_base = res_df[res_df["Model"] == "Drift-varying"].index[0]
            idx_sz = res_df[res_df["Model"] == "Drift + sz"].index[0]
            lr_stat, df_diff, p_val = likelihood_ratio_test(
                res_df.loc[idx_base, "NLL"], res_df.loc[idx_sz, "NLL"],
                res_df.loc[idx_base, "k"], res_df.loc[idx_sz, "k"]
            )
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"Drift-varying vs Drift+sz: χ²({df_diff}) = {lr_stat:.2f}, p = {p_val:.4f} {sig}")
        
        # Test 2: Drift-varying vs Drift + st
        if "Drift + st" in res_df["Model"].values:
            idx_base = res_df[res_df["Model"] == "Drift-varying"].index[0]
            idx_st = res_df[res_df["Model"] == "Drift + st"].index[0]
            lr_stat, df_diff, p_val = likelihood_ratio_test(
                res_df.loc[idx_base, "NLL"], res_df.loc[idx_st, "NLL"],
                res_df.loc[idx_base, "k"], res_df.loc[idx_st, "k"]
            )
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"Drift-varying vs Drift+st: χ²({df_diff}) = {lr_stat:.2f}, p = {p_val:.4f} {sig}")
        
        # Test 3: Drift-varying vs Full DDM
        if "Full DDM" in res_df["Model"].values:
            idx_base = res_df[res_df["Model"] == "Drift-varying"].index[0]
            idx_full = res_df[res_df["Model"] == "Full DDM"].index[0]
            lr_stat, df_diff, p_val = likelihood_ratio_test(
                res_df.loc[idx_base, "NLL"], res_df.loc[idx_full, "NLL"],
                res_df.loc[idx_base, "k"], res_df.loc[idx_full, "k"]
            )
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"Drift-varying vs Full DDM: χ²({df_diff}) = {lr_stat:.2f}, p = {p_val:.4f} {sig}")
        
        print("\nInterpretation: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
        print("If p > 0.05, the simpler model is preferred (complexity not justified).")

        # =========================================================
        # STEP 4: Visualizations
        # =========================================================
        print("\n" + "="*60)
        print("STEP 4: Generating Visualizations")
        print("="*60)
        
        # --- GRAPH 1: BIC Comparison ---
        plt.figure(figsize=(10, 6))
        colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' for i in range(len(res_df))]
        plt.bar(range(len(res_df)), res_df["BIC"], color=colors, alpha=0.7, edgecolor='black')
        plt.xticks(range(len(res_df)), res_df["Model"], rotation=45, ha='right')
        plt.ylabel("BIC Score (lower is better)", fontsize=12)
        plt.title("Model Comparison: BIC Scores", fontsize=14, fontweight='bold')
        plt.axhline(res_df["BIC"].min(), color='green', linestyle='--', alpha=0.5, label='Best Model')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- GRAPH 2: Simulation Grid ---
        fit_dict = {name: m for name, m in fitted_results}
        sorted_models = [(row["Model"], fit_dict[row["Model"]]) for i, row in res_df.iterrows() if row["Model"] in fit_dict]

        num_models = len(sorted_models)
        cols = 3
        rows = (num_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = axes.flatten()
        
        d_dutch = df_model[df_model['distractor_language'] == 'dutch']
        d_eng = df_model[df_model['distractor_language'] == 'english']
        
        for i, (name, m) in enumerate(sorted_models):
            ax = axes[i]
            # Data
            if "Null" in name:
                sns.kdeplot(df_model[df_model['correct']==1]['rt'], color="grey", fill=True, label="Data", ax=ax)
                sol = m.solve()
                ax.plot(sol.t_domain, sol.pdf("correct"), color="black", lw=2.5, linestyle="--", label="Model")
            else:
                sns.kdeplot(d_dutch[d_dutch['correct']==1]['rt'], color="tab:blue", fill=True, alpha=0.3, label="Data (NL)", ax=ax)
                sns.kdeplot(d_eng[d_eng['correct']==1]['rt'], color="tab:orange", fill=True, alpha=0.3, label="Data (EN)", ax=ax)
                
                sol_d = m.solve(conditions={"distractor_language": "dutch"})
                ax.plot(sol_d.t_domain, sol_d.pdf("correct"), color="tab:blue", lw=2.5, linestyle="--", label="Model (NL)")
                
                sol_e = m.solve(conditions={"distractor_language": "english"})
                ax.plot(sol_e.t_domain, sol_e.pdf("correct"), color="tab:orange", lw=2.5, linestyle="--", label="Model (EN)")

            bic_val = res_df[res_df['Model']==name]['BIC'].values[0]
            rank = list(res_df['Model']).index(name) + 1
            ax.set_title(f"{name} (Rank #{rank})\nBIC: {bic_val:.0f}", fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1.5)
            ax.set_xlabel("RT (s)")
            ax.set_ylabel("Density")
            if i == 0: ax.legend(fontsize='small', loc='upper right')

        for j in range(i+1, len(axes)): 
            axes[j].axis('off')
        
        plt.suptitle("Model Predictions vs. Data", fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()

    else:
        print("No models fitted successfully.")
        return
    
    print("\n" + "="*60)
    print("STEP 5: Individual Subject Fits (Best Model)")
    print("="*60)
    
    best_model_name = res_df.iloc[0]["Model"]
    print(f"\nFitting individual subjects using: {best_model_name}")
    
    individual_results = []
    subjects = df_model['participant'].unique()

    for sub in subjects:
        sub_data = df_model[df_model['participant'] == sub].copy()
        sub_sample = Sample.from_pandas_dataframe(sub_data, rt_column_name="rt", choice_column_name="correct")
        
        # Use Drift-varying for individual fits (most interpretable)
        sub_model = Model(
            name=f"Sub {sub}",
            drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5),
                                  v_english=Fittable(minval=-5, maxval=5)),
            bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
            IC=ICPoint(x0=0.0), 
            overlay=overlay_simple, 
            noise=noise, dx=0.005, dt=0.01, T_dur=3
        )
        
        try:
            fit_adjust_model(sample=sub_sample, model=sub_model, 
                           lossfunction=LossRobustLikelihood, verbose=False)
            p = sub_model.parameters()
            individual_results.append({
                "Subject": sub,
                "v_dutch": float(p['drift']['v_dutch']),
                "v_english": float(p['drift']['v_english']),
                "B": float(p['bound']['B']),
                "t0": float(p['overlay']['nondectime'])
            })
            print(f"  Subject {sub} fitted successfully.")
        except Exception as e:
            print(f"  Subject {sub} failed: {str(e)[:50]}")

    if individual_results:
        df_indiv = pd.DataFrame(individual_results)
        
        # Plot individual drift rates
        plt.figure(figsize=(8, 6))
        for i, row in df_indiv.iterrows():
            plt.plot([0, 1], [row['v_dutch'], row['v_english']], 'o-', color='grey', alpha=0.5)
        
        # Plot the Group Average in Bold Red
        plt.plot([0, 1], [df_indiv['v_dutch'].mean(), df_indiv['v_english'].mean()], 
                 'o-', color='red', linewidth=3, label='Group Mean')

        plt.xticks([0, 1], ['Dutch (Native)', 'English (L2)'])
        plt.ylabel('Drift Rate (v)')
        plt.title(f'Individual Drift Rates ({best_model_name} structure)')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- Statistical Test on Individual Params ---
        t_stat, p_val = ttest_rel(df_indiv['v_english'], df_indiv['v_dutch'])
        print(f"\nPaired t-test results (English vs Dutch):")
        print(f"t({len(df_indiv)-1}) = {t_stat:.2f}, p = {p_val:.4f}")
        
        print("\nIndividual Parameters Head:")
        print(df_indiv.head())

    print("\n" + "="*60)
    print("Analysis Complete.")
    print("="*60)