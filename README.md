# Cognitive Modeling Final Assignment 2025-26

**Leiden University**  
**Author:** Yotam Lev

---

## Project Overview

This project investigates the effect of **distractor language** (Dutch vs. English) on reaction times in a **lexical decision task** using **Drift Diffusion Models (DDM)**. The analysis compares multiple DDM variants to determine which cognitive mechanism best explains the observed behavioral differences.

---

## Workspace Structure

```
├── dataset-4.tsv                    # Raw experimental data (TSV format)
├── setup_and_preprocess.py          # Data loading, cleaning, descriptive stats
├── modeling_and_analysis.py         # PyDDM model fitting and comparison
├── parameter_recovery.py            # Parameter recovery validation
├── plot_hypothesis.py               # Hypothesis visualization
├── report.tex                       # Full LaTeX report
├── README.md                        # This file
└── generated_figures/               # Output directory for all figures
```

---

## Data Description

The dataset (`dataset-4.tsv`) contains trial-level data from a lexical decision experiment with the following columns:

| Column             | Description                          |
|--------------------|--------------------------------------|
| `participant`      | Participant ID (1-12)                |
| `stimulus`         | Stimulus type (word/nonword)         |
| `distractor_language` | Distractor language (dutch/english) |
| `response`         | Response given (word/nonword)        |
| `rt`               | Reaction time in seconds             |

---

## Pipeline

### Phase 1: Preprocessing & Descriptive Statistics

**File:** `setup_and_preprocess.py`

- Loads and cleans raw TSV data
- Computes accuracy (`correct` column)
- Filters trials: correct responses with RT ∈ [0.2s, 2.5s]
- Generates summary statistics (mean RT, SEM by condition)
- Produces RT distribution plots (Dutch vs. English)
- Output: **Figure 1-2** (mean RT table, distributions)

### Phase 2: DDM Model Fitting & Comparison

**File:** `modeling_and_analysis.py`

- Fits 5 DDM variants: Null, Drift-varying, Drift+st, Bound-varying, Full DDM
- Compares models using BIC
- Generates posterior predictive checks (PPC) for each model
- Performs paired t-tests on drift parameters across participants
- Output: **Figure 3-7** (BIC comparison, PPC plots, parameter estimates)

### Phase 3: Model Validation

**File:** `parameter_recovery.py`

- Validates parameter recovery using simulated data
- Ensures drift parameters are identifiable
- Generates recovery plots
- Output: **Figures 8-9** (recovery validation)

### Phase 4: Posterior Predictive Checks

**File:** `main.py` (Cells 4-5)

- Accuracy comparisons
- Summary statistics validation


---

## Running the Analysis
```bash
python main.py
```

---

## Key Findings

- **Best Model:** Drift-varying (language-dependent drift rates)
- **Effect:** Dutch distractors increase drift rate (v_dutch > v_english)
- **Mechanism:** Foreign language impairs *evidence quality*, not response strategy
- **Significance:** Paired t-test: t(11) = -3.61, p = 0.0041

---


All figures saved to `generated_figures/`

---

## Requirements

- Python 3.11+
- PyDDM
- pandas, numpy, scipy, matplotlib, seaborn

