import urllib.request
import ssl
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
bps = []
for i in range(len(c1_atoms)):
    for j in range(i+5, len(c1_atoms)):
        d = np.linalg.norm(c1_atoms[i] - c1_atoms[j])
        if 9.5 < d < 11.5:
            bps.append(d)

print(f"Average cross-strand pairing-like distance: {np.mean(bps):.2f}")
print(f"Min: {np.min(bps):.2f}, Max: {np.max(bps):.2f}, Count: {len(bps)}")
