# python
# setup_and_preprocess.py
# -------------------------------------------------
# Load data, compute accuracy, basic cleaning, table of means, and distribution plot.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1) Load: file has no header; columns are:
# participant, stimulus, distractor_language, response, rt
df = pd.read_csv("dataset-4.tsv", sep="\t", header=None,
                 names=["participant", "stimulus", "distractor_language", "response", "rt"])

# 2) Ensure types / drop malformed rows
df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
df = df.dropna(subset=["participant", "stimulus", "response", "rt"]).reset_index(drop=True)

# normalize participant type and distractor labels
try:
    df["participant"] = df["participant"].astype(int)
except Exception:
    df["participant"] = df["participant"].astype(str)
df["distractor_language"] = df["distractor_language"].astype(str).str.lower()

# 3) Accuracy
df["correct"] = (df["stimulus"] == df["response"]).astype(int)

# 4) Basic RT cleaning: keep correct trials within [0.2s, 2.5s]
df = df[(df["correct"] == 1) & (df["rt"] >= 0.2) & (df["rt"] <= 2.5)].copy()

# --- Compute means (per-participant -> across participants) ------------------
# per-participant mean RT for each distractor language
pp_means = df.groupby(["participant", "distractor_language"])["rt"].mean().unstack()

# focus only on english and dutch
conds = ["dutch", "english"]
# ensure columns exist
for c in conds:
    if c not in pp_means.columns:
        pp_means[c] = np.nan

# compute across-participant mean and SEM using participant means (preferred)
means = pp_means[conds].mean(axis=0, skipna=True)
sems = pp_means[conds].apply(lambda col: stats.sem(col.dropna()) if col.dropna().size > 0 else np.nan)

table_df = pd.DataFrame({
    "mean_s": means.round(4),
    "sem_s": sems.round(4)
}).loc[conds]

print("\nMean RTs (across participants' means):")
print(table_df)

# --- Figure 1: Table of mean times -----------------------------------------
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
plt.show()

# --- Figure 2: Distribution plot (Dutch vs English) -------------------------
df_plot = df[df["distractor_language"].isin(conds)].copy()
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df_plot, x="rt", hue="distractor_language", common_norm=False, fill=False)
plt.xlabel("Reaction time (s)")
plt.ylabel("Density (1/s)")
plt.title("RT distribution — Dutch vs English (trial-level)")
plt.tight_layout()
plt.show()