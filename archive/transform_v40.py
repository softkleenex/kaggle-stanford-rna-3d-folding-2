import re

with open("notebooks/top1/script.py", "r") as f:
    code = f.read()

# 1. Replace !pip with os.system
code = re.sub(r"^!pip install(.*)$", r'os.system("pip install\1")', code, flags=re.MULTILINE)
# Add missing import os at the very beginning
if not code.startswith("import os"):
    code = "import os\n" + code

# 2. Fix hardcoded paths
# Old:
# DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
# DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
# ...
code_paths = """
def find_csv(name):
    from pathlib import Path
    for p in Path("/kaggle/input").rglob(name):
        return str(p)
    return f"/kaggle/input/stanford-rna-3d-folding-2/{name}"

DEFAULT_TEST_CSV       = find_csv("test_sequences.csv")
DEFAULT_TRAIN_CSV      = find_csv("train_sequences.csv")
DEFAULT_TRAIN_LBLS     = find_csv("train_labels.csv")
DEFAULT_VAL_CSV        = find_csv("validation_sequences.csv")
DEFAULT_VAL_LBLS       = find_csv("validation_labels.csv")
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

# Make sure we don't mess up PROTENIX directory paths if they are static
DEFAULT_CODE_DIR = (
    "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
    "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
)
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR
"""
code = re.sub(
    r'DATA_BASE\s*=\s*"/kaggle/input/stanford-rna-3d-folding-2".*?DEFAULT_ROOT_DIR\s*=\s*DEFAULT_CODE_DIR',
    code_paths.strip(),
    code,
    flags=re.DOTALL
)

# 3. Add apply_chain_geometry
geometry_func = """
def apply_chain_geometry(coords: np.ndarray,
                         bond_length: float = 6.1,
                         noise_scale: float = 0.1) -> np.ndarray:
    \"\"\"
    Rescale coordinates so consecutive C1' distances approximate
    the real ~6.1 Å spacing in RNA backbones.
    This greatly improves TM-score for random/learned coords.
    \"\"\"
    if len(coords) < 2:
        return coords

    coords = coords.copy()
    # Center at origin
    coords -= coords.mean(axis=0)

    # Compute current mean bond length
    diffs = np.diff(coords, axis=0)                  # (L-1, 3)
    cur_bond = np.linalg.norm(diffs, axis=1).mean()  # scalar

    if cur_bond > 1e-6:
        scale = bond_length / cur_bond
        coords *= scale

    # Add tiny noise to break degeneracy
    coords += np.random.randn(*coords.shape) * noise_scale

    return coords
"""
code = code.replace("def main() -> None:", geometry_func + "\n\ndef main() -> None:")

# 4. Invoke apply_chain_geometry in main()
invoke_old = """
        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
"""
invoke_new = """
        # Apply chain geometry correction to each prediction
        for pi in range(len(combined)):
            combined[pi] = apply_chain_geometry(combined[pi])

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
"""
code = code.replace(invoke_old, invoke_new)

with open("notebooks/script_v40.py", "w") as f:
    f.write(code)

