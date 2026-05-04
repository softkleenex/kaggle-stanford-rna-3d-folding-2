from Bio.PDB import PDBParser
import numpy as np

def min_c1_dist(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_file)
    c1_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "C1'" in residue:
                    c1_coords.append(residue["C1'"].get_coord())
    c1_coords = np.array(c1_coords)
    min_dist = 999.0
    for i in range(len(c1_coords)):
        for j in range(i + 2, len(c1_coords)):
            d = np.linalg.norm(c1_coords[i] - c1_coords[j])
            if d < min_dist:
                min_dist = d
    print("Min non-adjacent C1'-C1' distance:", min_dist)

# Download done by curl
min_c1_dist('1EHZ.pdb')
