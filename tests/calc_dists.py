import numpy as np
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
structure = parser.get_structure('157D', '157D.pdb')
c1_coords = []
for model in structure:
    for chain in model:
        c1_chain = []
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname in ['A', 'U', 'C', 'G', 'RA', 'RU', 'RC', 'RG'] or len(resname) == 1:
                for atom in residue:
                    if atom.get_name() in ["C1'", "C1*"]:
                        c1_chain.append(atom.get_coord())
        if len(c1_chain) > 5: c1_coords.extend(c1_chain)
coords = np.array(c1_coords)
print("d4:", np.mean([np.linalg.norm(coords[i+4]-coords[i]) for i in range(len(coords)-4)]))
print("d5:", np.mean([np.linalg.norm(coords[i+5]-coords[i]) for i in range(len(coords)-5)]))
print("d6:", np.mean([np.linalg.norm(coords[i+6]-coords[i]) for i in range(len(coords)-6)]))
