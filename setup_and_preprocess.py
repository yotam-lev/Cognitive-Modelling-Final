# setup_and_preprocess.py
# -------------------------------------------------
# Module for loading data and running descriptive statistics (Phase 1).

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random

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

def load_data(filepath):
    """
    Loads the raw TSV data, fixes types, and creates the 'correct' column.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep="\t", header=None,
                         names=["participant", "stimulus", "distractor_language", "response", "rt"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    # Ensure types / drop malformed rows
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["participant", "stimulus", "response", "rt"]).reset_index(drop=True)
    
    # Normalize participant type and distractor labels
    try:
        df["participant"] = df["participant"].astype(int)
    except Exception:
        df["participant"] = df["participant"].astype(str)
        
    df["distractor_language"] = df["distractor_language"].astype(str).str.lower()
    
    # Accuracy
    df["correct"] = (df["stimulus"] == df["response"]).astype(int)
    
    return df

def get_phase1_data(df):
    """
    Prepares data for Descriptive Stats:
    - Keep CORRECT trials only.
    - Keep RTs within [0.2s, 2.5s].
    """
    return df[(df["correct"] == 1) & (df["rt"] >= 0.2) & (df["rt"] <= 2.5)].copy()

def get_phase2_data(df):
    """
    Prepares data for DDM Modeling:
    - Keep RTs within [0.2s, 2.5s].
    - KEEP ERRORS (essential for DDM).
    """
    return df[(df["rt"] >= 0.2) & (df["rt"] <= 2.5)].copy()

def generate_descriptive_plots(df):
    """
    Generates the Table of Means and the RT Distribution Plot (Figure 1 & 2).
    """
    ensure_figures_dir()
    print("\n--- Generating Descriptive Plots ---")
    
    # --- Compute means (per-participant -> across participants) ---
    pp_means = df.groupby(["participant", "distractor_language"])["rt"].mean().unstack()
    conds = ["dutch", "english"]
    
    # Ensure columns exist
    for c in conds:
        if c not in pp_means.columns:
            pp_means[c] = np.nan
    
    # Compute across-participant mean and SEM
    means = pp_means[conds].mean(axis=0, skipna=True)
    sems = pp_means[conds].apply(lambda col: stats.sem(col.dropna()) if col.dropna().size > 0 else np.nan)
    
    table_df = pd.DataFrame({
        "mean_s": means.round(4),
        "sem_s": sems.round(4)
    }).loc[conds]
    
    print("\nMean RTs (across participants' means):")
    print(table_df)
    
    # --- Figure 1: Table of mean times ---
    fig, ax = plt.subplots(figsize=(4.5, 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=table_df.reset_index().values,
                   colLabels=["distractor", "mean_s", "sem_s"],
                   cellLoc="center",
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    plt.title("Mean RTs by distractor (s)", pad=6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_mean_rt_table.png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/figure1_mean_rt_table.png")
    plt.show()
    
    # --- Figure 2: Distribution plot (Dutch vs English) ---
    df_plot = df[df["distractor_language"].isin(conds)].copy()
    if not df_plot.empty:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df_plot, x="rt", hue="distractor_language", common_norm=False, fill=False)
        plt.xlabel("Reaction time (s)")
        plt.ylabel("Density (1/s)")
        plt.title("RT distribution — Dutch vs English (trial-level)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure2_rt_distribution.png"), dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR}/figure2_rt_distribution.png")
        plt.show()
    else:
        print("Not enough data for distribution plot.")