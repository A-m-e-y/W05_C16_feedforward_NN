import matplotlib.pyplot as plt
import csv

# Read data from CSV
csv_file = "feedforward_timing.csv"
backends = []
times = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        backends.append(row['backend'])
        times.append(float(row['avg_time_ms']))

# Define custom colors for each backend
color_map = {
    'cuda': '#1f77b4',         # Blue
    'pytorch_cpu': '#ff7f0e',  # Orange
    'pytorch_gpu': '#2ca02c'   # Green
}
colors = [color_map.get(b, '#888888') for b in backends]

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(backends, times, color=colors)

# Add value labels on top
for bar, time in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{time:.3f} ms", ha='center', va='bottom', fontsize=10)

plt.title("Feedforward Network Forward Pass Timing (Batch Size = 1024)")
plt.xlabel("Backend")
plt.ylabel("Average Time (ms)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/feedforward_benchmark_plot.png", dpi=300)
plt.show()
