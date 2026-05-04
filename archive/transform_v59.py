import re
import os

with open('notebooks/top1/script.py', 'r') as f:
    code = f.read()

# 1. Fix pip installs
code = re.sub(r'^!pip install (.+)$', r'import os\nos.system("pip install --quiet \1")', code, flags=re.MULTILINE)

# 2. Add Rigorous Geometry Pass
geometry_pass = """
def rigorous_geometry_pass(coords, passes=50, target_bond=5.95, tether_weight=0.05):
    import numpy as np
    X = coords.copy()
    X_orig = coords.copy()
    L = len(X)
    if L < 2: return X
    for _ in range(passes):
        # Tether to prevent global drift
        X += tether_weight * (X_orig - X)
        # Enforce C1'-C1' bonds
        d = X[1:] - X[:-1]
        dist = np.linalg.norm(d, axis=1) + 1e-8
        adj = d * ((target_bond - dist) / dist)[:, None] * 0.5
        X[:-1] -= adj
        X[1:] += adj
    return X
"""

# Insert geometry pass after generate_rna_structure
code = code.replace('def tbm_phase(', geometry_pass + '\n\ndef tbm_phase(')

# 3. Apply geometry pass in Phase 3
old_phase3 = """        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

new_phase3 = """        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        
        # --- PHASE 4: Rigorous Geometry Pass ---
        for s in range(stacked.shape[0]):
            stacked[s] = rigorous_geometry_pass(stacked[s])
            
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

code = code.replace(old_phase3, new_phase3)

# 4. Fix DATA_BASE paths dynamically
old_paths = """# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2\""""

new_paths = """# ─────────────── Paths & Constants ───────────────────────────────────────────
import os
DATA_BASE = "/kaggle/input/stanford-rna-3d-folding-2"
for _cand in ['/kaggle/input/stanford-rna-3d-folding-2', '/kaggle/input/competitions/stanford-rna-3d-folding-2']:
    if os.path.isdir(_cand):
        DATA_BASE = _cand
        break
"""
code = code.replace(old_paths, new_paths)

with open('notebooks/script_v59.py', 'w') as f:
    f.write(code)
print("Saved notebooks/script_v59.py")
