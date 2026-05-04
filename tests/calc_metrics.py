import numpy as np
from Bio.PDB import PDBParser
import sys

def calc_cross_metrics(coords_a, coords_b):
    # coords_a and coords_b are anti-parallel
    # i in A pairs with L-1-i in B
    L = len(coords_a)
    direct = []
    cross_short = []
    cross_long = []
    for i in range(L):
        j = L - 1 - i
        dist_ij = np.linalg.norm(coords_a[i] - coords_b[i]) # Wait, they are anti-parallel in PDB?
        # In PDB, chain A is 1..N, chain B is 1..N.
        # Often they are paired i, N+1-i?
        pass

    # Let's just find the closest C1' on the other chain
    for i in range(L):
        dists = np.linalg.norm(coords_b - coords_a[i], axis=1)
        idx = np.argmin(dists)
        direct.append(dists[idx])
        if idx + 1 < len(coords_b):
            cross_long.append(np.linalg.norm(coords_b[idx+1] - coords_a[i]))
        if idx - 1 >= 0:
            cross_short.append(np.linalg.norm(coords_b[idx-1] - coords_a[i]))
            
    print(f'Direct (C1\'-C1\'): {np.mean(direct):.2f}')
    print(f'Cross (C1\'-C1\' offset +1): {np.mean(cross_long):.2f}')
    print(f'Cross (C1\'-C1\' offset -1): {np.mean(cross_short):.2f}')
    
    # 2nd neighbors
    cross_long2 = []
    cross_short2 = []
    for i in range(L):
        dists = np.linalg.norm(coords_b - coords_a[i], axis=1)
        idx = np.argmin(dists)
        if idx + 2 < len(coords_b):
            cross_long2.append(np.linalg.norm(coords_b[idx+2] - coords_a[i]))
        if idx - 2 >= 0:
            cross_short2.append(np.linalg.norm(coords_b[idx-2] - coords_a[i]))
    print(f'Cross (C1\'-C1\' offset +2): {np.mean(cross_long2):.2f}')
    print(f'Cross (C1\'-C1\' offset -2): {np.mean(cross_short2):.2f}')

parser = PDBParser(QUIET=True)
structure = parser.get_structure('157D', '157D.pdb')

c1_a = []
c1_b = []

for model in structure:
    for chain in model:
        c1_chain = []
        for residue in chain:
            for atom in residue:
                if atom.get_name() in ["C1'", "C1*"]:
                    c1_chain.append(atom.get_coord())
        if chain.id == 'A': c1_a = np.array(c1_chain)
        if chain.id == 'B': c1_b = np.array(c1_chain)

if len(c1_a) > 0 and len(c1_b) > 0:
    calc_cross_metrics(c1_a, c1_b)
