import json
import re

with open("notebooks/ribonanza/script.py", "r") as f:
    ribo_code = f.read()

# Discovery logic for RibonanzaNet datasets
discovery_code = """
import os, glob, sys
from pathlib import Path

print("--- STEP 1: AUTO-RESOLVING RIBONANZANET PATHS ---")
def find_path(name, is_dir=False):
    for p in Path("/kaggle/input").rglob(name):
        if is_dir and p.is_dir(): return str(p)
        if not is_dir and p.is_file(): return str(p)
    return None

test_csv_path = find_path("test_sequences.csv")
DATA_BASE = os.path.dirname(test_csv_path) if test_csv_path else "/kaggle/input/competitions/stanford-rna-3d-folding-2"

# Resolve Ribonanza Weights
ribo_2d_final = find_path("ribonanzanet2d-final", is_dir=True) or "/kaggle/input/ribonanzanet2d-final"
ribo_weights = find_path("RibonanzaNet.pt")
if ribo_weights:
    ribo_weights_dir = os.path.dirname(ribo_weights)
else:
    ribo_weights_dir = "/kaggle/input/ribonanzanet-pretrained-checkpoint"

ribo_3d_weights = find_path("RibonanzaNet-3D.pt")
if ribo_3d_weights:
    ribo_3d_dir = os.path.dirname(ribo_3d_weights)
else:
    ribo_3d_dir = "/kaggle/input/ribonanzanet-3d-finetune"

print(f"DATA_BASE: {DATA_BASE}")
print(f"RIBO_2D: {ribo_2d_final}")
print(f"RIBO_WEIGHTS: {ribo_weights_dir}")
print(f"RIBO_3D: {ribo_3d_dir}")
"""

# Replace hardcoded paths in the original script
ribo_code = ribo_code.replace('"/kaggle/input/stanford-ribonanza-2-rna-folding-in-3-d/test_sequences.csv"', 'f"{DATA_BASE}/test_sequences.csv"')
ribo_code = ribo_code.replace('"/kaggle/input/ribonanzanet2d-final"', 'ribo_2d_final')
ribo_code = ribo_code.replace('"/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt"', 'f"{ribo_weights_dir}/RibonanzaNet.pt"')
ribo_code = ribo_code.replace('"/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"', 'f"{ribo_2d_final}/configs/pairwise.yaml"')
ribo_code = ribo_code.replace('"/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt"', 'f"{ribo_3d_dir}/RibonanzaNet-3D.pt"')

final_script = discovery_code + "\n\n" + ribo_code

nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": [final_script]}],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}
with open("notebooks/rna_folding_v25_ribonanza.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
