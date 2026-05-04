import os
import re
from pathlib import Path

def update_script():
    with open('notebooks/top1/script.py', 'r') as f:
        code = f.read()

    # 1. Fix magic pip install
    code = code.replace(
        "!pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "import os\nos.system('pip install --quiet --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl')"
    )
    code = code.replace(
        "!pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "os.system('pip install --quiet --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl')"
    )
    code = code.replace(
        "!pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl",
        "os.system('pip install --quiet --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl')"
    )

    # 2. Fix dynamic path resolution
    dynamic_path_code = """
def find_dataset_file(filename: str, default_path: str) -> str:
    if os.path.exists(default_path):
        return default_path
    for p in Path("/kaggle/input").rglob(filename):
        return str(p)
    return default_path

DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = find_dataset_file("test_sequences.csv", f"{DATA_BASE}/test_sequences.csv")
DEFAULT_TRAIN_CSV      = find_dataset_file("train_sequences.csv", f"{DATA_BASE}/train_sequences.csv")
DEFAULT_TRAIN_LBLS     = find_dataset_file("train_labels.csv", f"{DATA_BASE}/train_labels.csv")
DEFAULT_VAL_CSV        = find_dataset_file("validation_sequences.csv", f"{DATA_BASE}/validation_sequences.csv")
DEFAULT_VAL_LBLS       = find_dataset_file("validation_labels.csv", f"{DATA_BASE}/validation_labels.csv")
"""
    code = re.sub(
        r'DATA_BASE\s*=\s*"/kaggle/input/stanford-rna-3d-folding-2".*?DEFAULT_VAL_LBLS\s*=\s*f"\{DATA_BASE\}/validation_labels\.csv"',
        dynamic_path_code.strip(),
        code,
        flags=re.DOTALL
    )

    # 3. Add tethered geometry relaxation function
    tether_func = """
def tethered_geometry_relaxation(coords, target_id, segments_map, passes=5) -> np.ndarray:
    \"\"\"
    Enforce strict 5.95A C1'-C1' bond distances post-stitching,
    while using a harmonic tether to the original coordinates to prevent global drift.
    \"\"\"
    X = coords.copy()
    X_orig = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    
    target_bond = 5.95
    tether_weight = 0.1
    
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 2:
                continue
            
            # tether
            C += tether_weight * (X_orig[s:e] - C)
            
            # bond enforcement
            d = C[1:] - C[:-1]
            dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((target_bond - dist) / dist)[:, None] * 0.5
            
            C[:-1] -= adj
            C[1:] += adj
            
            X[s:e] = C
            
    return X
"""
    code = code.replace("def _rotmat(axis, ang):", tether_func + "\n\ndef _rotmat(axis, ang):")

    # 4. Apply geometry relaxation to protenix outputs
    protenix_append_old = """
        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)
"""
    protenix_append_new = """
        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                refined_ptx = tethered_geometry_relaxation(ptx[j], tid, segments_map)
                combined.append(refined_ptx)  # (seq_len, 3)
"""
    code = code.replace(protenix_append_old, protenix_append_new)

    # 5. Add variation to generate_rna_structure
    gen_rna_old = """
def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    \"\"\"Idealized A-form RNA helix — last-resort de-novo fallback.\"\"\"
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords
"""
    gen_rna_new = """
def generate_rna_structure(sequence: str, seed=None, variation=1) -> np.ndarray:
    \"\"\"Idealized A-form RNA helix — last-resort de-novo fallback with diverse parametric generation.\"\"\"
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    
    # Parametric Variation for A-form helices across 5 models
    rise = 2.81 + (variation - 3) * 0.1
    rotation = 32.7 + (variation - 3) * 2
    radius = 10.0 + (variation - 3) * 0.5
    
    for i in range(n):
        ang = np.radians(rotation * i)
        coords[i] = [radius * np.cos(ang), radius * np.sin(ang), rise * i]
    return coords
"""
    code = code.replace(gen_rna_old.strip(), gen_rna_new.strip())

    # Update call to generate_rna_structure
    call_old = "dn       = generate_rna_structure(seq, seed=seed_val)"
    call_new = "dn       = generate_rna_structure(seq, seed=seed_val, variation=(len(combined) % 5) + 1)"
    code = code.replace(call_old, call_new)

    with open('notebooks/script_v62.py', 'w') as f:
        f.write(code)

if __name__ == '__main__':
    update_script()
