import numpy as np

def parse_pdb(filename):
    c1_coords = []
    p_coords = []
    lines = open(filename).readlines()
    for line in lines:
        if line.startswith("ATOM  "):
            atom_name = line[12:16].strip()
            chain = line[21]
            res = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if atom_name == "C1'":
                c1_coords.append((chain, res, x, y, z))
            elif atom_name == "P":
                p_coords.append((chain, res, x, y, z))
    return c1_coords, p_coords

c1, p = parse_pdb("pdb1rna.ent")

def get_coords(coords, chain_id):
    return np.array([[c[2], c[3], c[4]] for c in coords if c[0] == chain_id])

cA = get_coords(c1, "A")
cB = get_coords(c1, "B")
pA = get_coords(p, "A")
pB = get_coords(p, "B")

for i in range(5):
    j = len(cB) - 1 - i
    # Minor groove (j-3)
    if j-3 >= 0:
        print(f"A[{i}] to B[{j-3}] P-P (Minor): {np.linalg.norm(pA[i] - pB[j-3]):.2f}")
    # Major groove (j-6 or j-5)
    if j-6 >= 0:
        print(f"A[{i}] to B[{j-6}] P-P (Major): {np.linalg.norm(pA[i] - pB[j-6]):.2f}")
    if j-5 >= 0:
        print(f"A[{i}] to B[{j-5}] P-P (Major/Diagonal): {np.linalg.norm(pA[i] - pB[j-5]):.2f}")
