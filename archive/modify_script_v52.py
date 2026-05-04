import re

with open('notebooks/top1/script.py', 'r') as f:
    content = f.read()

# 1. Pip installs -> os.system
def repl_pip(m):
    return 'import os\nos.system("pip install --quiet' + m.group(1) + '")\n'

content = re.sub(r'!pip install(.*?)\n', repl_pip, content)

# Remove redundant import os
content = re.sub(r'(import os\n)+', 'import os\n', content)

# 2. Dynamic paths
path_def = """from pathlib import Path

def get_path(filename: str, default: str) -> str:
    paths = list(Path('/kaggle/input').rglob(filename))
    return str(paths[0]) if paths else default

DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = get_path("test_sequences.csv", f"{DATA_BASE}/test_sequences.csv")
DEFAULT_TRAIN_CSV      = get_path("train_sequences.csv", f"{DATA_BASE}/train_sequences.csv")
DEFAULT_TRAIN_LBLS     = get_path("train_labels.csv", f"{DATA_BASE}/train_labels.csv")
DEFAULT_VAL_CSV        = get_path("validation_sequences.csv", f"{DATA_BASE}/validation_sequences.csv")
DEFAULT_VAL_LBLS       = get_path("validation_labels.csv", f"{DATA_BASE}/validation_labels.csv")
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"
"""

content = re.sub(
    r'DATA_BASE\s*=\s*"/kaggle/input/stanford-rna-3d-folding-2".*?DEFAULT_OUTPUT\s*=\s*"/kaggle/working/submission\.csv"',
    path_def,
    content,
    flags=re.DOTALL
)

# 3. generate_rna_structure
new_gen = """def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    \"\"\"Idealized A-form RNA helix with realistic physical geometry and variation.\"\"\"
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence)
    coords = np.zeros((n, 3))
    twist = np.radians(32.7) + np.random.normal(0, 0.05)
    rise = 2.81 + np.random.normal(0, 0.1)
    radius = 9.5 + np.random.normal(0, 0.2)
    for i in range(n):
        ang = i * twist
        coords[i] = [radius * np.cos(ang), radius * np.sin(ang), i * rise]
    return coords"""

content = re.sub(
    r'def generate_rna_structure\(.*?return coords',
    new_gen,
    content,
    flags=re.DOTALL
)

# 4. enforce_c1_geometry
enforce_c1 = """
def enforce_c1_geometry(coords, target_dist=5.95, lr=0.5, steps=15):
    X = coords.copy()
    for _ in range(steps):
        d = X[1:] - X[:-1]
        dist = np.linalg.norm(d, axis=1) + 1e-8
        diff = (target_dist - dist) / dist
        adj = d * diff[:, None] * lr
        X[:-1] -= adj * 0.5
        X[1:] += adj * 0.5
    return X

# ─────────────── TBM Phase ───────────────────────────────────────────────────"""

content = content.replace(
    '# ─────────────── TBM Phase ───────────────────────────────────────────────────',
    enforce_c1
)

# 5. Phase 3 combined logic
old_phase3 = """        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

new_phase3 = """        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                # Apply physics-based smoothing and self-avoidance to Protenix output
                refined_ptx = adaptive_rna_constraints(ptx[j], tid, segments_map, confidence=0.8, passes=3)
                combined.append(refined_ptx)

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        
        # Enforce exact C1' geometry across all predictions to prevent coordinate shrinkage
        for j in range(stacked.shape[0]):
            stacked[j] = enforce_c1_geometry(stacked[j], target_dist=5.95, lr=0.5, steps=15)
            
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

content = content.replace(old_phase3, new_phase3)

with open('notebooks/script_v52.py', 'w') as f:
    f.write(content)
print("Updated script_v52.py")
