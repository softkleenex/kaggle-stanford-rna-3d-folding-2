import json

notebook_path = "notebooks/rna_folding_baseline.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Clear the first cell and put a clean install command
nb["cells"][0] = {
    "cell_type": "code",
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Clean install with internet enabled
",
        "try:
",
        "    import Bio
",
        "    print('Biopython already present')
",
        "except ImportError:
",
        "    !pip install -q biopython
"
    ],
    "execution_count": None
}

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)
