# 
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to extract values from log file
def extract_values(log_path):
    alphas = []
    signals = []
    sigmoids = []
    iterations = []
    iter_count = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extract Signal, Sigmoid, and Alpha values
            alpha_match = re.search(r'Signal: ([-\d\.]+), Sigmoid: ([\d\.]+), Alpha: ([\d\.]+)', line)
            if alpha_match:
                iter_count += 1
                signals.append(float(alpha_match.group(1)))
                sigmoids.append(float(alpha_match.group(2)))
                alphas.append(float(alpha_match.group(3)))
                iterations.append(iter_count)
    
    return iterations, alphas, signals, sigmoids

# Path to your log file
log_path = "/Users/bhumikagupta/genLoss/glam_vision_datasets/example/greedy_lookahead_optimizer/alpha_adaptive results/temp.log"

# Extract values
iterations, alphas, signals, sigmoids = extract_values(log_path)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot Alpha values
ax1.plot(iterations, alphas, 'b-', label='Alpha')
ax1.set_ylabel('Alpha Value')
ax1.set_title('Adaptive Alpha Value Over Training')
ax1.set_ylim([min(0.1, min(alphas)-0.05), max(0.9, max(alphas)+0.05)])  # Set y-limits with padding
ax1.grid(True)
ax1.legend()

# Plot Signal values with positive/negative highlighting
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Add a line at y=0
ax2.plot(iterations, signals, 'g-', label='Signal')
ax2.fill_between(iterations, signals, 0, where=np.array(signals) > 0, 
                 color='green', alpha=0.3, label='Positive Signal (Improving)')
ax2.fill_between(iterations, signals, 0, where=np.array(signals) <= 0, 
                 color='red', alpha=0.3, label='Negative Signal (Stagnating)')
ax2.set_ylabel('Signal Value')
ax2.set_title('Signal Value Over Training (Improvement Metric)')
ax2.grid(True)
ax2.legend()

# Plot Sigmoid values
ax3.plot(iterations, sigmoids, 'r-', label='Sigmoid')
ax3.set_ylabel('Sigmoid Value')
ax3.set_xlabel('Iteration')
ax3.set_title('Sigmoid Value Over Training (Controls Alpha Scaling)')
ax3.set_ylim([0, 1])  # Sigmoid is always between 0 and 1
ax3.grid(True)
ax3.legend()

# Add explanation text box
plt.figtext(0.5, 0.01, 
         "Signal: Normalized metric showing optimization progress.\n"
         "Sigmoid: Transforms signal to (0,1) range for alpha calculation.\n"
         "Alpha: Dynamic lookahead parameter; higher values favor more stable/conservative updates.",
         ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for text
plt.savefig('adaptive_alpha_analysis.png', dpi=300)
plt.show()