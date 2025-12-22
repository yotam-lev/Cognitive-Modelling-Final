# main.py
# -------------------------------------------------
# Entry point: controls the workflow and calls other modules.

import sys
import setup_and_preprocess as pre
import modeling_and_analysis as ddm
import random 
import numpy as np
import warnings



# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DATA_FILE = "dataset-4.tsv"

warnings.filterwarnings("ignore", message=".*Setting undecided probability.*")
warnings.filterwarnings("ignore", message=".*is deprecated and will be removed in a future version.*")
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("==========================================")
    print("   Reaction Time & DDM Analysis Pipeline  ")
    print("==========================================")
    
    # 1. Load Data
    try:
        raw_df = pre.load_data(DATA_FILE)
        print(f"Data loaded successfully. Total rows: {len(raw_df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. User Input
    print("\nSelect Analysis Mode:")
    print("[1] Descriptive Stats (Phase 1: Correct Only, Plots)")
    print("[2] DDM Modeling (Phase 2: All Trials, Model Fitting)")
    print("[3] Run Full Pipeline (Both)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()

    # 3. Execution Logic
    if choice == "1":
        print("\n--> Running Phase 1: Descriptive Statistics...")
        # Get correct-only data for phase 1
        df_phase1 = pre.get_phase1_data(raw_df)
        pre.generate_descriptive_plots(df_phase1)
        
    elif choice == "2":
        print("\n--> Running Phase 2: DDM Modeling...")
        # Get data with errors for phase 2
        df_phase2 = pre.get_phase2_data(raw_df)
        ddm.run_ddm_analysis(df_phase2)
        
    elif choice == "3":
        print("\n--> Running Full Pipeline...")
        
        # Phase 1
        print("\n[PART 1: Descriptive Stats]")
        df_phase1 = pre.get_phase1_data(raw_df)
        pre.generate_descriptive_plots(df_phase1)
        
        print("\n" + "-"*40 + "\n")
        
        # Phase 2
        print("[PART 2: DDM Modeling]")
        df_phase2 = pre.get_phase2_data(raw_df)
        ddm.run_ddm_analysis(df_phase2)
        
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()