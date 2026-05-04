import re

with open('notebooks/top1/script.py', 'r') as f:
    code = f.read()

# Replace pip installs
pip_old = """!pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl

!pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

!pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"""

pip_new = """import os
import sys

os.system("pip install --quiet --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --quiet --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --quiet --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")
"""
code = code.replace(pip_old, pip_new)

# Add find_dataset_file
import_sys_idx = code.find('import sys')
utils_idx = code.find('def seed_everything')

find_func = """def find_dataset_file(filename: str, default_path: str) -> str:
    if os.path.exists(default_path):
        return default_path
    if IS_KAGGLE:
        found = list(Path('/kaggle/input').rglob(filename))
        if found:
            return str(found[0])
    return default_path

"""
code = code[:utils_idx] + find_func + code[utils_idx:]


# Replace resolve_paths
resolve_old = """def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir"""

resolve_new = """def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    test_csv   = find_dataset_file('test_sequences.csv', test_csv)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir"""

code = code.replace(resolve_old, resolve_new)


# Replace adapt_template_to_query
adapt_old = """def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln        = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs):
            new_coords[qs:qe] = chunk
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0:
                new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0:
                new_coords[i] = new_coords[nv] + [3, 0, 0]
            else:
                new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)"""

adapt_new = """def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln        = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs):
            new_coords[qs:qe] = chunk
            
    c1_dist = 5.9
    
    # Pass 1: Interpolate bounded gaps
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]

    # Pass 2: Forward extension for missing C-terminus
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv == -1:
                if i - 1 > 0 and not np.isnan(new_coords[i-2, 0]):
                    d = new_coords[i-1] - new_coords[i-2]
                    norm = np.linalg.norm(d)
                    direction = d / norm if norm > 1e-4 else np.array([1.0, 0.0, 0.0])
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                
                direction = direction + np.random.randn(3) * 0.15
                direction /= (np.linalg.norm(direction) + 1e-6)
                new_coords[i] = new_coords[i-1] + direction * c1_dist

    # Pass 3: Backward extension for missing N-terminus
    for i in range(len(new_coords)-1, -1, -1):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv == -1 and nv >= 0:
                if i + 1 < len(new_coords) - 1 and not np.isnan(new_coords[i+2, 0]):
                    d = new_coords[i+1] - new_coords[i+2]
                    norm = np.linalg.norm(d)
                    direction = d / norm if norm > 1e-4 else np.array([-1.0, 0.0, 0.0])
                else:
                    direction = np.array([-1.0, 0.0, 0.0])
                    
                direction = direction + np.random.randn(3) * 0.15
                direction /= (np.linalg.norm(direction) + 1e-6)
                new_coords[i] = new_coords[i+1] + direction * c1_dist
                
    # Pass 4: If completely empty
    if np.isnan(new_coords).any():
        for i in range(len(new_coords)):
            if np.isnan(new_coords[i, 0]):
                new_coords[i] = [i * c1_dist, 0, 0]
                
    return np.nan_to_num(new_coords)"""

code = code.replace(adapt_old, adapt_new)

# Replace generate_rna_structure
gen_old = """def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    \"\"\"Idealized A-form RNA helix — last-resort de-novo fallback.\"\"\"
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords"""

gen_new = """def generate_rna_structure(sequence: str, seed=None, variation=1) -> np.ndarray:
    \"\"\"Idealized A-form RNA helix — last-resort de-novo fallback.\"\"\"
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    
    # A-form helix parameters with physical variation
    rise = 2.81 + (variation - 3) * 0.1
    rotation = 32.7 + (variation - 3) * 2
    radius = 10.0 + (variation - 3) * 0.5
    
    for i in range(n):
        ang = np.radians(rotation * i)
        coords[i] = [radius * np.cos(ang), radius * np.sin(ang), i * rise]
    return coords"""

code = code.replace(gen_old, gen_new)

# Update generate_rna_structure call
call_old = "dn       = generate_rna_structure(seq, seed=seed_val)"
call_new = "dn       = generate_rna_structure(seq, seed=seed_val, variation=len(combined) + 1)"
code = code.replace(call_old, call_new)

# Replace the data loading logic in main
load_old = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)"""

load_new = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs_path = find_dataset_file('train_sequences.csv', DEFAULT_TRAIN_CSV)
    val_seqs_path = find_dataset_file('validation_sequences.csv', DEFAULT_VAL_CSV)
    train_labels_path = find_dataset_file('train_labels.csv', DEFAULT_TRAIN_LBLS)
    val_labels_path = find_dataset_file('validation_labels.csv', DEFAULT_VAL_LBLS)

    train_seqs   = pd.read_csv(train_seqs_path)
    val_seqs     = pd.read_csv(val_seqs_path)
    train_labels = pd.read_csv(train_labels_path)
    val_labels   = pd.read_csv(val_labels_path)"""

code = code.replace(load_old, load_new)

with open('notebooks/script_v51.py', 'w') as f:
    f.write(code)

