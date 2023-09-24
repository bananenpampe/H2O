import sys
import os
import ase.io

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))

from load import get_qm9_w_problematic


frames = get_qm9_w_problematic()
dir_path = os.path.dirname(os.path.realpath(__file__))
ase.io.write(os.path.join(dir_path, "..", "data", "qm9.xyz"), frames)
