# python
# modeling_and_analysis.py
# -------------------------------------------------
# Phase 2: PyDDM Modeling (Corrected for fit_result error)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyddm import Model, Sample, Fittable
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPoint, Drift, Bound
from pyddm.functions import fit_adjust_model
from pyddm.models.loss import LossRobustLikelihood 

# 1. Load & prepare data
df = pd.read_csv("dataset-4.tsv", sep="\t", header=None,
                 names=["participant", "stimulus", "distractor_language", "response", "rt"])
df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
df = df.dropna(subset=["rt", "response", "stimulus"]).reset_index(drop=True)
df["distractor_language"] = df["distractor_language"].astype(str).str.lower()
df["correct"] = (df["stimulus"] == df["response"]).astype(int)

# keep reasonable RTs but keep errors for DDM
df_model = df[(df["rt"] >= 0.2) & (df["rt"] <= 2.5)].copy()

print(f"Data ready for modeling. N={len(df_model)} trials.")
print(f"Accuracy: {df_model['correct'].mean():.2%}")

# Create PyDDM Sample
# Updated to avoid deprecation warning: use choice_column_name
sample = Sample.from_pandas_dataframe(df_model, rt_column_name="rt", choice_column_name="correct")

# 2. Custom components --------------------------------------------

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


# Common components
overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.5))
noise = NoiseConstant(noise=1)  # fixed

# 3. Define models -----------------------------------------------
model_null = Model(
    name="Null Model",
    drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
    bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
    IC=ICPoint(x0=0.0),
    overlay=overlay,
    noise=noise,
    dx=0.005, dt=0.01, T_dur=3
)

model_drift = Model(
    name="Drift-varying Model",
    drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5),
                           v_english=Fittable(minval=-5, maxval=5)),
    bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
    IC=ICPoint(x0=0.0),
    overlay=overlay,
    noise=noise,
    dx=0.005, dt=0.01, T_dur=3
)

model_bound = Model(
    name="Bound-varying Model",
    drift=DriftConstant(drift=Fittable(minval=-5, maxval=5)),
    bound=BoundDistractor(a_dutch=Fittable(minval=0.5, maxval=3.0),
                          a_english=Fittable(minval=0.5, maxval=3.0)),
    IC=ICPoint(x0=0.0),
    overlay=overlay,
    noise=noise,
    dx=0.005, dt=0.01, T_dur=3
)

models = [model_null, model_drift, model_bound]

# 4. Fit models & Calculate BIC ----------------------------------
fitted_results = []
results_table = []

print("\nStarting Model Fitting...")

for m in models:
    print(f"Fitting {m.name} ...")
    try:
        # Fit the model
        fitted = fit_adjust_model(sample=sample, model=m, lossfunction=LossRobustLikelihood, verbose=False)
        
        # --- FIXED MANUAL LOSS CALCULATION ---
        # Determine if this model uses the condition
        if "Null" in m.name:
            # Null model doesn't use conditions, so we pass an empty list
            conds = [] 
        else:
            # Drift/Bound models need to split by language
            conds = ["distractor_language"]
            
        loss_obj = LossRobustLikelihood(sample=sample, model=fitted, 
                                        required_conditions=conds,  # <--- Use variable here
                                        dt=m.dt, T_dur=m.T_dur)
        
        nll = loss_obj.loss(fitted)
        
        # Calculate BIC
        k = len(fitted.get_model_parameters())
        n = len(df_model)
        bic = k * np.log(n) + 2 * nll
        
        # Store results
        fitted_results.append((m.name, fitted))
        results_table.append({"Model": m.name, "BIC": bic, "NLL": nll, "Params": fitted.get_model_parameters()})
        
        print(f"-> Success! BIC: {bic:.2f}")
        
    except Exception as e:
        print(f"Failed to fit {m.name}: {e}")
        import traceback
        traceback.print_exc()

# 5. Show Final Table & Plot Winner
if results_table:
    res_df = pd.DataFrame(results_table).sort_values("BIC")
    print("\n=== FINAL RESULTS (Lowest BIC Wins) ===")
    print(res_df[["Model", "BIC", "NLL"]])
    
    winner_name = res_df.iloc[0]["Model"]
    winner_model = [m for name, m in fitted_results if name == winner_name][0]
    
    print(f"\nVisualizing Winner: {winner_name}")
    
    # Posterior Predictive Check Plot
    import pyddm.plot
    plt.figure(figsize=(10, 6))
    pyddm.plot.plot_fit_diagnostics(model=winner_model, sample=sample)
    plt.suptitle(f"Winning Model Fit: {winner_name}\n(Solid=Data, Line=Model)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. EXTENSION: Fit Individual Subjects
# ==========================================
print("\n--- Starting Individual Subject Fits (Scope & Ambition) ---")

individual_results = []
winning_model_class = model_drift.__class__ # We use the structure of the winner (Drift Model)

# Loop through each unique participant
subjects = df_model['participant'].unique()

for sub in subjects:
    # 1. Filter data for this subject
    sub_data = df_model[df_model['participant'] == sub].copy()
    sub_sample = Sample.from_pandas_dataframe(sub_data, rt_column_name="rt", choice_column_name="correct")
    
    # 2. Re-create the model fresh for this subject
    # We use the same structure as the winning 'Drift Model'
    sub_model = Model(
        name=f"Sub {sub}",
        drift=DriftDistractor(v_dutch=Fittable(minval=-5, maxval=5),
                              v_english=Fittable(minval=-5, maxval=5)),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=3.0)),
        IC=ICPoint(x0=0.0),
        overlay=overlay,
        noise=noise,
        dx=0.005, dt=0.01, T_dur=3
    )
    
    try:
        # 3. Fit
        fit_adjust_model(sample=sub_sample, model=sub_model, lossfunction=LossRobustLikelihood, verbose=False)
        
        # 4. Extract Parameters
        params = {
            "Subject": sub,
            "v_dutch": float(sub_model.parameters()['drift']['v_dutch']),
            "v_english": float(sub_model.parameters()['drift']['v_english']),
            "threshold": float(sub_model.parameters()['bound']['B']),
            "non_dec": float(sub_model.parameters()['overlay']['nondectime'])
        }
        individual_results.append(params)
        print(f"Subject {sub} fitted.")
        
    except Exception as e:
        print(f"Failed to fit Subject {sub}: {e}")

# Create DataFrame of Individual Params
df_indiv = pd.DataFrame(individual_results)
print("\nIndividual Parameters Head:")
print(df_indiv.head())

# ==========================================
# 7. Visualization: Slope Plot (Dutch vs English Drift)
# ==========================================


plt.figure(figsize=(6, 6))

# Plot lines connecting each subject's condition
# x=0 is Dutch, x=1 is English
for i, row in df_indiv.iterrows():
    plt.plot([0, 1], [row['v_dutch'], row['v_english']], 'o-', color='grey', alpha=0.5)

# Plot the Group Average in Bold Red
plt.plot([0, 1], [df_indiv['v_dutch'].mean(), df_indiv['v_english'].mean()], 
         'o-', color='red', linewidth=3, label='Group Mean')

plt.xticks([0, 1], ['Dutch (Native)', 'English (L2)'])
plt.ylabel('Drift Rate (v)')
plt.title('Individual Drift Rates by Distractor Language')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Perform a quick T-Test to report statistics
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(df_indiv['v_english'], df_indiv['v_dutch'])
print(f"\nPaired t-test results: t({len(df_indiv)-1}) = {t_stat:.2f}, p = {p_val:.4f}")