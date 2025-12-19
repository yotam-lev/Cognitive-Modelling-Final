# Cognitive Modeling Final Assignment 2025-26

**Leiden University**  
**Author:** Yotam Lev

---

## Project Overview

This project investigates the effect of **distractor language** (Dutch vs. English) on reaction times in a **lexical decision task** using **Drift Diffusion Models (DDM)**. The analysis compares multiple DDM variants to determine which cognitive mechanism best explains the observed behavioral differences.

---

## Workspace Structure

├── dataset-4.tsv # Raw experimental data (TSV format) 
├── setup_and_preprocess.py # Data loading, cleaning, and descriptive statistics 
├── modeling and analysis.py # PyDDM model fitting and comparison 
├── fit_ddm_models.py # Alternative DDM implementation 
├── README.md # This file

---

## Data Description

The dataset (`dataset-4.tsv`) contains trial-level data from a lexical decision experiment with the following columns:

| Column      | Description                          |
|-------------|--------------------------------------|
| `subjects`  | Participant ID (1-12)                |
| `S`         | Stimulus type (word/nonword)         |
| `distractor`| Distractor language (dutch/english)  |
| `R`         | Response given (word/nonword)        |
| `rt`        | Reaction time in seconds             |

---

## Pipeline

### 1. Preprocessing (`setup_and_preprocess.py`)

- Loads and cleans the raw data
- Computes accuracy (`correct` column)
- Filters trials: correct responses with RT ∈ [0.2s, 2.5s]
- Generates summary statistics (mean RT, SEM by condition)
- Produces RT distribution plots (Dutch vs. English)

**Run:**
```bash
python setup_and_preprocess.py