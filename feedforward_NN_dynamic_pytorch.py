import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import csv

# ---- Constants
INPUT_SIZE = 4
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
BATCH_SIZE = 1024
REPEAT = 100
CSV_FILE = "feedforward_layer_scaling.csv"

# ---- Parse arguments
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BACKEND = 'pytorch_gpu'

if len(sys.argv) >= 2 and sys.argv[1].lower() == 'cpu':
    DEVICE = 'cpu'
    BACKEND = 'pytorch_cpu'

if len(sys.argv) < 3:
    print("Usage: python3 feedforward_NN_dynamic_pytorch.py cpu/gpu <max_hidden_layers>")
    sys.exit(1)

MAX_LAYERS = int(sys.argv[2])

# ---- Dynamic Feedforward Network Definition
class DynamicFF(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(INPUT_SIZE, HIDDEN_SIZE))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

# ---- Run benchmarks from 1 to MAX_LAYERS
with open(CSV_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    for L in range(1, MAX_LAYERS + 1):
        times = []
        for run in range(REPEAT):
            torch.manual_seed(run)
            model = DynamicFF(L).to(DEVICE)
            x = torch.randn(BATCH_SIZE, INPUT_SIZE).to(DEVICE)

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
                elapsed = (time.time() - t0) * 1000  # ms

            times.append(elapsed)
        avg = sum(times[2:]) / (REPEAT - 2)
        print(f"{BACKEND} | Layers = {L} | Avg Time = {avg:.4f} ms")
        writer.writerow([BACKEND, L, BATCH_SIZE, f"{avg:.4f}"])
