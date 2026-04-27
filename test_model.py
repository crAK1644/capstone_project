import os
import sys

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
