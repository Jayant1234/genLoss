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
            alpha_match = re.search(r'Signal: ([\d\.]+), Sigmoid: ([\d\.]+), Alpha: ([\d\.]+)', line)
            if alpha_match:
                iter_count += 1
                signals.append(float(alpha_match.group(1)))
                sigmoids.append(float(alpha_match.group(2)))
                alphas.append(float(alpha_match.group(3)))
                iterations.append(iter_count)
    
    return iterations, alphas, signals, sigmoids

# Path to your log file
log_path = '/Users/bhumikagupta/genLoss/glam_vision_datasets/example/greedy_lookahead_optimizer/alpha_adaptive results/adaptive_alpha_v3_K10_lr0.1_output.log'

# Extract values
iterations, alphas, signals, sigmoids = extract_values(log_path)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot Alpha values
ax1.plot(iterations, alphas, 'b-', label='Alpha')
ax1.set_ylabel('Alpha Value')
ax1.set_title('Adaptive Alpha Value Over Training')
ax1.grid(True)
ax1.legend()

# Plot Signal values
ax2.plot(iterations, signals, 'g-', label='Signal')
ax2.set_ylabel('Signal Value')
ax2.set_title('Signal Value Over Training')
ax2.grid(True)
ax2.legend()

# Plot Sigmoid values
ax3.plot(iterations, sigmoids, 'r-', label='Sigmoid')
ax3.set_ylabel('Sigmoid Value')
ax3.set_xlabel('Iteration')
ax3.set_title('Sigmoid Value Over Training')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig('adaptive_alpha_analysis.png', dpi=300)
plt.show()