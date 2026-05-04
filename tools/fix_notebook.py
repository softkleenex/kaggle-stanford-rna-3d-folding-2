import json
import os

notebook_path = "notebooks/rna_folding_baseline.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)

# The first code cell is where biopython is installed
# Let's replace its content with something more robust
new_install_code = """
import os
import sys
import subprocess

# Robust biopython install for offline Kaggle
biopython_paths = [
    "/kaggle/input/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "/kaggle/input/datasets/ogurtsov/biopython/biopython-1.85-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
]

installed = False
try:
    import Bio
    print("Biopython already installed.")
    installed = True
except ImportError:
    for path in biopython_paths:
        if os.path.exists(path):
            print(f"Installing biopython from {path}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", path, "--no-index", "--find-links=/kaggle/input/biopython"])
            installed = True
            break

if not installed:
    print("WARNING: Could not find biopython .whl file in /kaggle/input.")
    # Fallback to online install IF internet is enabled
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "biopython"])
        print("Installed biopython via pip online.")
    except:
        print("CRITICAL: Failed to install biopython.")
"""

# Find the cell that had biopython install or create a new first cell
found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and ("biopython" in "".join(cell["source"]) or "!pip install" in "".join(cell["source"])):
        cell["source"] = [new_install_code]
        found = True
        break

if not found:
    # Insert at the beginning if not found
    nb["cells"].insert(0, {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "source": [new_install_code],
        "execution_count": None
    })

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)
