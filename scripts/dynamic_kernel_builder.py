import json
import re
import os
import sys

def build(n_sample, min_similarity, chunk_overlap, version):
    with open("notebooks/top_kernels/winner_script.py", "r") as f:
        code = f.read()

    # 1. Parameter Injection
    code = code.replace('N_SAMPLE      = 5', f'N_SAMPLE      = {n_sample}')
    code = code.replace('MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))', f'MIN_SIMILARITY       = {min_similarity}')
    code = code.replace('CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "64"))', f'CHUNK_OVERLAP = {chunk_overlap}')
    
    # 2. Fixes (No Hash, etc.)
    code = code.replace('    os.environ["PYTHONHASHSEED"] = str(seed)', '    pass')
    code = code.replace('rng     = np.random.default_rng((abs(hash(tid)) + i * 10007) % (2**32))', 'rng     = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))')
    code = code.replace('seed_val = hash(tid) % 10000 + len(combined) * 1000', 'seed_val = row.name * 1000000 + len(combined) * 1000')

    # 3. Rename conflicting variables for our discovery logic
    code = code.replace('DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"', 'DATA_BASE_OLD = ""')
    code = code.replace('DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"', 'DEFAULT_TEST_CSV_OLD = ""')
    code = code.replace('DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"', 'DEFAULT_TRAIN_CSV_OLD = ""')
    code = code.replace('DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"', 'DEFAULT_TRAIN_LBLS_OLD = ""')
    code = code.replace('DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"', 'DEFAULT_VAL_CSV_OLD = ""')
    code = code.replace('DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"', 'DEFAULT_VAL_LBLS_OLD = ""')
    code = code.replace('DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"', 'DEFAULT_OUTPUT_OLD = ""')
    code = re.sub(r"DEFAULT_CODE_DIR\s*=\s*\(.*?\)", "DEFAULT_CODE_DIR_OLD = \"\"", code, flags=re.DOTALL)
    code = code.replace("DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR", "DEFAULT_ROOT_DIR_OLD = \"\"")

    discovery_code = """
import os, glob, subprocess, sys
from pathlib import Path

print("--- STEP 1: INSTALLING SPECIFIC DEPENDENCIES ---")
target_packages = ["biopython", "biotite", "rdkit", "biotraj"]
wheels_dir = "/kaggle/input/stanford-rna-protenix-wheels-3-12/protenix_wheels"
if not os.path.exists(wheels_dir):
    for p in Path("/kaggle/input").rglob("protenix_wheels"):
        if "3-12" in str(p): wheels_dir = str(p); break

wheels = glob.glob(os.path.join(wheels_dir, "*.whl"))
to_install = [w for w in wheels if any(pkg in os.path.basename(w).lower() for pkg in target_packages)]
if to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + to_install + ["--no-index", "--quiet"])

import Bio, biotite, rdkit, biotraj
print("Verification: Targeted libraries imported successfully!")

os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
print("FORCE: KAGGLE_IS_COMPETITION_RERUN set to 1")

print("--- STEP 2: AUTO-RESOLVING PATHS ---")
def find_path(name, is_dir=False):
    for p in Path("/kaggle/input").rglob(name):
        if is_dir and p.is_dir(): return str(p)
        if not is_dir and p.is_file(): return str(p)
    return None

test_csv_path = find_path("test_sequences.csv")
DATA_BASE = os.path.dirname(test_csv_path) if test_csv_path else "/kaggle/input/competitions/stanford-rna-3d-folding-2"
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
"""

    final_script = discovery_code + "\n\n" + code
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [], "source": [final_script]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    
    nb_name = f"rna_folding_v{version}.ipynb"
    with open(f"notebooks/{nb_name}", "w") as f:
        json.dump(nb, f, indent=1)
    
    kernel_id = f"softkleenex/rna-folding-v{version}"
    meta = {
        "id": kernel_id,
        "title": f"RNA Folding v{version} Auto",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "dataset_sources": ["qiweiyin/protenix-v1-adjusted", "yuriygreben/stanford-rna-protenix-wheels-3-12"],
        "competition_sources": ["stanford-rna-3d-folding-2"]
    }
    with open("notebooks/kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=1)
    
    return kernel_id

if __name__ == "__main__":
    n = int(sys.argv[1])
    s = float(sys.argv[2])
    o = int(sys.argv[3])
    v = int(sys.argv[4])
    print(build(n, s, o, v))
