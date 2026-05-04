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

# Standard A-form RNA helix j = 12 - i?
# 1RNA is a 6-bp duplex? Let's check.
print(f"cA length: {len(cA)}")
print(f"cB length: {len(cB)}")

# 1RNA has 6 bp in chain A and 6 in B?
# Let's check pairing
for i in range(min(len(cA), len(cB))):
    j = len(cB) - 1 - i
    dist = np.linalg.norm(cA[i] - cB[j])
    print(f"Pair {i}-{j} C1'-C1' distance: {dist:.2f}")

# Minor groove (j-3)
i = 0
j = len(cB) - 1 - i
if j-3 >= 0:
    print(f"Minor groove P-P (A[{i}] to B[{j-3}]): {np.linalg.norm(pA[i] - pB[j-3]):.2f}")

# Major groove
# If minor is i to j-3, major is i to j+5?
# Since we only have 6bp, we can't measure 11bp turns.
# But we can measure i to i+2
if len(pA) > 2:
    print(f"i to i+2 P-P (A[0] to A[2]): {np.linalg.norm(pA[0] - pA[2]):.2f}")
    print(f"i to i+3 P-P (A[0] to A[3]): {np.linalg.norm(pA[0] - pA[3]):.2f}")
    print(f"i to i+2 C1'-C1' (A[0] to A[2]): {np.linalg.norm(cA[0] - cA[2]):.2f}")
    print(f"i to i+3 C1'-C1' (A[0] to A[3]): {np.linalg.norm(cA[0] - cA[3]):.2f}")
