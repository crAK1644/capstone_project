import os
import sys
import time
from thop import profile 

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src", "ssfl_project"))

import torch
from model import build_classifier, build_discriminator

device = torch.device("cpu")
clf = build_classifier(num_classes=11, device=device)
disc = build_discriminator(device=device)

x = torch.randn(4, 23, 5)
clf_out = clf(x)
disc_out = disc(x)

print("Input shape:          ", tuple(x.shape))
print("Classifier output:    ", tuple(clf_out.shape), "  (expected: (4, 11))")
print("Discriminator output: ", tuple(disc_out.shape), "   (expected: (4, 2))")
print("Classifier params:    ", sum(p.numel() for p in clf.parameters()))
print("Discriminator params: ", sum(p.numel() for p in disc.parameters()))

assert clf_out.shape == (4, 11)
assert disc_out.shape == (4, 2)
print("")
print("OK - TrafficCNN is wired correctly.")

# --- MODEL PERFORMANCE & COMPLEXITY ANALYSIS ---

def run_performance_benchmark(model, model_name):
    print(f"\n>>> {model_name} Analysis")
    
    # Standard input shape for IoT traffic: (Batch=1, Features=23, Length=5)
    # This matches our final data specification
    dummy_input = torch.randn(1, 23, 5)
    
    # 1. FLOPs and Parameter Count
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    # 2. Memory Footprint (Float32 = 4 bytes per parameter)
    size_mb = (params * 4) / (1024 * 1024)
    
    # 3. Inference Latency (Timing the forward pass)
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):  # 100 iterations for a stable average
            _ = model(dummy_input)
    end_time = time.perf_counter()
    avg_latency = ((end_time - start_time) / 100) * 1000  # Convert to milliseconds

    print(f"  - Parameters:      {params/1e3:.2f} K")
    print(f"  - Operations:      {flops/1e6:.4f} MFLOPs")
    print(f"  - Memory Usage:    {size_mb:.4f} MB")
    print(f"  - Inference Time:  {avg_latency:.4f} ms")

# Apply analysis to our model heads
run_performance_benchmark(clf, "Classifier")
run_performance_benchmark(disc, "Discriminator")