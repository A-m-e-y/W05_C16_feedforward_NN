import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv("feedforward_layer_scaling.csv")

# Extract unique values
layers = sorted(df["hidden_layers"].unique())
backends = ['cuda', 'pytorch_cpu', 'pytorch_gpu']
colors = {'cuda': '#1f77b4', 'pytorch_cpu': '#ff7f0e', 'pytorch_gpu': '#2ca02c'}

# Create bar positions
x = np.arange(len(layers))  # positions for each group (layer)
bar_width = 0.25

# Plot setup
plt.figure(figsize=(12, 6))

for i, backend in enumerate(backends):
    times = []
    for layer in layers:
        match = df[(df["backend"] == backend) & (df["hidden_layers"] == layer)]
        time = match["avg_time_ms"].values[0] if not match.empty else 0
        times.append(time)

    offset = (i - 1) * bar_width  # center around group
    positions = x + offset
    bars = plt.bar(positions, times, bar_width, label=backend, color=colors[backend])

    # Add text annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                 f"{height:.2f}", ha='center', va='bottom', fontsize=9)

# Final plot formatting
plt.xticks(x, layers)
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Average Time (ms)")
plt.title("Feedforward Inference Time vs Hidden Layers (Batch Size = 1024)")
plt.legend(title="Backend")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/layer_scaling_comparison.png", dpi=300)
plt.show()
