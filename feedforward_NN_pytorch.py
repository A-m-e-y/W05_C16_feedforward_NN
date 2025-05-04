import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import csv

# ---- Config from CLI ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1024
BACKEND = 'pytorch_gpu'

if len(sys.argv) >= 2:
    if sys.argv[1].lower() == 'cpu':
        DEVICE = 'cpu'
        BACKEND = 'pytorch_cpu'
    elif sys.argv[1].lower() == 'gpu':
        DEVICE = 'cuda'
        BACKEND = 'pytorch_gpu'

if len(sys.argv) >= 3:
    BATCH_SIZE = int(sys.argv[2])

print(f"Running on {DEVICE.upper()} with batch size {BATCH_SIZE}")

# ---- Model Definition ----
class FeedforwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---- Benchmark ----
times = []

for run in range(5):
    torch.manual_seed(run)

    model = FeedforwardNet().to(DEVICE)
    x = torch.randn(BATCH_SIZE, 4).to(DEVICE)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model(x)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        t0 = time.time()
        _ = model(x)
        elapsed = (time.time() - t0) * 1000

    times.append(elapsed)
    print(f"Run {run+1}: {elapsed:.4f} ms")

avg_time = sum(times[2:]) / 3
print(f"Feedforward {BACKEND} avg time (runs 3–5): {avg_time:.4f} ms")

# ---- Append to CSV ----
with open("feedforward_timing.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([BACKEND, BATCH_SIZE, f"{avg_time:.4f}"])
