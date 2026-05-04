import json
import re
import os

# Read the original winner script
with open("notebooks/top_kernels/winner_script.py", "r") as f:
    winner_code = f.read()

discovery_code = """
import os, glob, subprocess, sys
from pathlib import Path

print("--- STEP 1: INSTALLING DEPENDENCIES ---")
wheels_dir = "/kaggle/input/stanford-rna-protenix-wheels-3-12/protenix_wheels"
if not os.path.exists(wheels_dir):
    for p in Path("/kaggle/input").rglob("protenix_wheels"):
        if "3-12" in str(p):
            wheels_dir = str(p)
            break

wheels = glob.glob(os.path.join(wheels_dir, "*.whl"))
if wheels:
    print(f"Found {len(wheels)} wheels. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + wheels + ["--no-index", "--quiet"])

import Bio, biotite, rdkit, biotraj
print("Verification: ALL libraries imported successfully!")

print("--- STEP 2: AUTO-RESOLVING PATHS ---")
def find_path(name, is_dir=False):
    for p in Path("/kaggle/input").rglob(name):
        if is_dir and p.is_dir(): return str(p)
        if not is_dir and p.is_file(): return str(p)
    return None

test_csv_path = find_path("test_sequences.csv")
if test_csv_path:
    DATA_BASE = os.path.dirname(test_csv_path)
else:
    DATA_BASE = "/kaggle/input/competitions/stanford-rna-3d-folding-2"

DEFAULT_TEST_CSV   = find_path("test_sequences.csv") or f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV  = find_path("train_sequences.csv") or f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS = find_path("train_labels.csv") or f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV    = find_path("validation_sequences.csv") or f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS   = find_path("validation_labels.csv") or f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT     = "/kaggle/working/submission.csv"

protenix_base = find_path("protenix-v1-adjusted", is_dir=True)
if protenix_base:
    protenix_dirs = list(Path(protenix_base).rglob("Protenix-v1"))
    DEFAULT_CODE_DIR = str(protenix_dirs[-1]) if protenix_dirs else protenix_base
else:
    DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"

DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR
print(f"Resolved DATA_BASE: {DATA_BASE}")
print(f"Resolved CODE_DIR: {DEFAULT_CODE_DIR}")
"""

# Rename conflicting variables
winner_code = winner_code.replace('DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"', 'DATA_BASE_OLD = ""')
winner_code = winner_code.replace('DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"', 'DEFAULT_TEST_CSV_OLD = ""')
winner_code = winner_code.replace('DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"', 'DEFAULT_TRAIN_CSV_OLD = ""')
winner_code = winner_code.replace('DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"', 'DEFAULT_TRAIN_LBLS_OLD = ""')
winner_code = winner_code.replace('DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"', 'DEFAULT_VAL_CSV_OLD = ""')
winner_code = winner_code.replace('DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"', 'DEFAULT_VAL_LBLS_OLD = ""')
winner_code = winner_code.replace('DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"', 'DEFAULT_OUTPUT_OLD = ""')
winner_code = re.sub(r"DEFAULT_CODE_DIR\s*=\s*\(.*?\)", "DEFAULT_CODE_DIR_OLD = \"\"", winner_code, flags=re.DOTALL)
winner_code = winner_code.replace("DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR", "DEFAULT_ROOT_DIR_OLD = \"\"")

final_script = discovery_code + "\n\n" + winner_code

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [final_script]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/rna_folding_v18.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
