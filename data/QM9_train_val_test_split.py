import sys
import os
import ase.io
import random

SEED = 0
random.seed(SEED)


frames = ase.io.read("../data/qm9.xyz", index=":")

frames_filtered = []
frames_problematic = []

for frame in frames:
    if frame.info['problematic'] == "OK":
        frames_filtered.append(frame)
    elif frame.info['problematic'] == "PROBLEMATIC":
        frames_problematic.append(frame)
    else:
        raise ValueError("problematic value not recognized")

print("Number of frames: ", len(frames))
print("Number of filtered frames: ", len(frames_filtered))
print("Number of problematic frames: ", len(frames_problematic))

random.shuffle(frames_filtered)

frames_train = frames_filtered[:100000]
frames_val = frames_filtered[100000:110000]
frames_test = frames_filtered[120000:]

print("Number of training frames: ", len(frames_train))
print("Number of validation frames: ", len(frames_val))
print("Number of test frames: ", len(frames_test))

ase.io.write("../data/qm9_train.xyz", frames_train)
ase.io.write("../data/qm9_val.xyz", frames_val)
ase.io.write("../data/qm9_test.xyz", frames_test)

ase.io.write("../data/qm9_problematic.xyz", frames_problematic)