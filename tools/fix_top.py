import os
import sys
import subprocess

# ── Install dependencies offline ──────────────────────────────────────────────
wheels = [
    "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
    "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
]
for w in wheels:
    if os.path.exists(w):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", w])
    else:
        print(f"Warning: Wheel not found {w}, skipping installation.")

import pandas as pd
