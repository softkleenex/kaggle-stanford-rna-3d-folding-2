import re
from pathlib import Path

content = open('notebooks/script_v333.py', 'r').read()

# 1. Fix !pip install -> syntactically valid Python with offline installers
pip_pattern = re.compile(r"(!pip install[^\n]+(?:\n\s*!pip install[^\n]+)*)", re.MULTILINE)
new_install = """import os
import sys
import subprocess
from pathlib import Path

def install_wheels():
    potential_base = Path("/kaggle/input")
    wheels = list(potential_base.rglob("*.whl"))
    
    priority = ["biopython", "biotite", "rdkit"]
    to_install = []
    for p in priority:
        for w in wheels:
            if p in w.name.lower():
                to_install.append(str(w))
                break
                
    if not to_install:
        to_install = [
            "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
            "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
            "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
        ]

    for wheel in to_install:
        if os.path.exists(wheel):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
            except Exception as e:
                print(f"Failed to install {wheel}: {e}")

if os.path.exists("/kaggle/input"):
    install_wheels()
"""
content = pip_pattern.sub(new_install, content, count=1)

# 2. Fix static paths to dynamic paths
content = content.replace(
'''DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"''',
'''def find_path(name, is_dir=False):
    for p in Path("/kaggle/input").rglob(name):
        if is_dir and p.is_dir(): return str(p)
        if not is_dir and p.is_file(): return str(p)
    return None

test_csv_path = find_path("test_sequences.csv")
DATA_BASE = os.path.dirname(test_csv_path) if test_csv_path else "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV   = find_path("test_sequences.csv") or f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV  = find_path("train_sequences.csv") or f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS = find_path("train_labels.csv") or f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV    = find_path("validation_sequences.csv") or f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS   = find_path("validation_labels.csv") or f"{DATA_BASE}/validation_labels.csv"'''
)

content = content.replace(
'''DEFAULT_CODE_DIR = (
    "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
    "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
)''',
'''protenix_base = find_path("protenix-v1-adjusted", is_dir=True)
if protenix_base:
    protenix_dirs = list(Path(protenix_base).rglob("Protenix-v1"))
    DEFAULT_CODE_DIR = str(protenix_dirs[-1]) if protenix_dirs else protenix_base
else:
    DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"'''
)

# 3. Insert the BOSS strategy constraint into adaptive_rna_constraints
# We'll look for CNBS constraint and insert BOSS right after it.
cnbs_pattern = r"""        # ── Consecutive Nucleobase Stacking \(CNBS\) Constraint ────────────────
        cnbs_strength = 0\.35 \* strength
        if L_full >= 2:
            d_cnbs = X\[1:\] - X\[:-1\]
            dist_cnbs = np\.linalg\.norm\(d_cnbs, axis=1\) \+ 1e-12
            adj_cnbs = d_cnbs \* \(\(3\.4 - dist_cnbs\) / dist_cnbs\)\[:, None\] \* cnbs_strength
            X\[:-1\] -= adj_cnbs \* 0\.5
            X\[1:\] \+= adj_cnbs \* 0\.5"""

boss_addition = """

        # ── Base-on-Sugar Stacking (BOSS) Constraint ─────────────────────────
        # Implement a 4.8 Å distance constraint between the O4' atom of residue i 
        # and the centroid of the base of residue i+1 to stabilize the characteristic 
        # A-form base-on-sugar stacking.
        boss_strength = 0.28 * strength
        if L_full >= 2:
            d_boss = X[1:] - X[:-1]
            dist_boss = np.linalg.norm(d_boss, axis=1) + 1e-12
            adj_boss = d_boss * ((4.8 - dist_boss) / dist_boss)[:, None] * boss_strength
            X[:-1] -= adj_boss * 0.5
            X[1:] += adj_boss * 0.5"""

if "cnbs_strength = 0.35 * strength" in content:
    content = re.sub(cnbs_pattern, cnbs_pattern.replace('\\', '') + boss_addition, content)
else:
    # If CNBS is missing, let's insert it at the beginning of the passes loop
    fallback_pattern = r"(for _ in range\(passes\):[\s\S]*?for s, e in segments:\n\s+C = X\[s:e\]; L = e - s\n\s+if L < 3:\n\s+continue)"
    fallback_addition = fallback_pattern.replace('\\', '') + boss_addition.replace('X[', 'C[').replace('X[:-1]', 'C[:-1]').replace('L_full', 'L')
    content = re.sub(fallback_pattern, fallback_addition, content)

open('notebooks/script_v333.py', 'w').write(content)
print("Fix applied.")
