import os

with open('notebooks/top1/script.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('!pip '):
        cmd = line.strip()[1:] # remove !
        new_lines.append(f'import os\nos.system("{cmd}")\n')
    else:
        new_lines.append(line)

content = "".join(new_lines)

# 1. Add Dynamic DATA_BASE
data_base_target = 'DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"'
data_base_replacement = """def get_dataset_base():
    from pathlib import Path
    for p in Path("/kaggle/input").rglob("test_sequences.csv"):
        return str(p.parent)
    return "/kaggle/input/stanford-rna-3d-folding-2"

DATA_BASE              = get_dataset_base()"""

content = content.replace(data_base_target, data_base_replacement)

# 2. Add apply_physics_constraints
physics_func = """
def apply_physics_constraints(coords: np.ndarray, sequence: str, passes: int = 2) -> np.ndarray:
    \"\"\"
    Apply global physical constraints to refine final coordinates:
    1. Fix consecutive C1'-C1' distances to ~6.0A (biological RNA backbone).
    2. Prevent severe steric clashes (< 3.8A).
    3. Encourage base-pairing distances (10.5A) for proximal Watson-Crick pairs.
    \"\"\"
    X = coords.copy()
    L = len(sequence)
    if L < 3:
        return X
        
    pairs_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    
    for _ in range(passes):
        # 1. Sequential distance constraints (i to i+1)
        d = X[1:] - X[:-1]
        dist = np.linalg.norm(d, axis=1) + 1e-6
        adj = d * ((6.0 - dist) / dist)[:, None] * 0.3
        X[:-1] -= adj
        X[1:] += adj
        
        # 2. Steric clash prevention
        idx = np.linspace(0, L - 1, min(L, 200)).astype(int) if L > 200 else np.arange(L)
        P = X[idx]
        diff = P[:, None, :] - P[None, :, :]
        dm = np.linalg.norm(diff, axis=2) + 1e-6
        sep = np.abs(idx[:, None] - idx[None, :])
        mask = (sep > 2) & (dm < 3.8)
        if np.any(mask):
            vec = (diff * ((3.8 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
            X[idx] += 0.05 * vec
            
        # 3. Base-pairing constraints
        for i in range(L):
            comp = pairs_map.get(sequence[i])
            if not comp: continue
            for j in range(i + 3, min(i + 20, L)):
                if sequence[j] == comp:
                    diff_bp = X[j] - X[i]
                    dist_bp = np.linalg.norm(diff_bp) + 1e-6
                    if 8.0 < dist_bp < 14.0:
                        bp_adj = diff_bp * ((10.5 - dist_bp) / dist_bp) * 0.1
                        X[i] -= bp_adj
                        X[j] += bp_adj
                        break
    return X

# ─────────────── TBM Phase ───────────────────────────────────────────────────
"""

content = content.replace("# ─────────────── TBM Phase ───────────────────────────────────────────────────", physics_func)

# 3. Inject it into main
replace_target = """        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

replacement = """        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        
        # Apply global physics constraints (clash + base-pairing)
        for i_s in range(stacked.shape[0]):
            stacked[i_s] = apply_physics_constraints(stacked[i_s], seq)
            
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

content = content.replace(replace_target, replacement)

with open('notebooks/script_v41.py', 'w') as f:
    f.write(content)
print("Transformation complete.")
