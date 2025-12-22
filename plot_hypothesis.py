import matplotlib.pyplot as plt
import numpy as np

def plot_hypotheses():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Parameters
    boundary = 1.0
    start_point = 0.5
    
    # 1. Draw Boundaries
    ax.axhline(boundary, color='black', linewidth=2, label='Decision Boundary (a)')
    ax.axhline(0, color='black', linewidth=2)
    ax.axhline(start_point, color='grey', linestyle=':', label='Start Point (z)')
    
    # 2. Draw Drift Rates (The "Interference" Hypothesis)
    # English (Steep slope)
    x = np.linspace(0, 1.0, 100)
    y_english = start_point + 0.8 * x
    ax.plot(x, y_english, 'g-', linewidth=2, label='English Distractor (High v)')
    
    # Dutch (Shallow slope) - The Interference Effect
    x_dutch = np.linspace(0, 1.5, 100)
    y_dutch = start_point + 0.5 * x_dutch 
    ax.plot(x_dutch, y_dutch, 'r--', linewidth=2, label='Dutch Distractor (Low v)')
    
    # 3. Annotations
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Accumulated Evidence', fontsize=12)
    ax.set_title('Hypothesis 1: Distractor Interference Reduces Drift Rate', fontsize=14)
    ax.legend(loc='lower right')
    
    # Remove top/right spines for a "scientific" look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

plot_hypotheses()