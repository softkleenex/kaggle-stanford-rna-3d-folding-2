import ssl
import urllib.request
from Bio.PDB import PDBList
from Bio.PDB.PDBParser import PDBParser
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

pdbl = PDBList()
pdbl.retrieve_pdb_file("1RNA", pdir=".", file_format="pdb")

parser = PDBParser()
structure = parser.get_structure("1RNA", "pdb1rna.ent")

c1_atoms = []
for model in structure:
    for chain in model:
        for residue in chain:
            if "C1'" in residue:
                c1_atoms.append(residue["C1'"].get_coord())

c1_atoms = np.array(c1_atoms)

distances = {}
counts = {}

for i in range(len(c1_atoms)):
    for j in range(i+1, min(i+12, len(c1_atoms))):
        k = j - i
        d = np.linalg.norm(c1_atoms[i] - c1_atoms[j])
        distances[k] = distances.get(k, 0) + d
        counts[k] = counts.get(k, 0) + 1

for k in sorted(distances.keys()):
    print(f"i to i+{k}: {distances[k]/counts[k]:.2f}")
