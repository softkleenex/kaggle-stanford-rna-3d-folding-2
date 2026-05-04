import numpy as np
import urllib.request

urllib.request.urlretrieve("https://files.rcsb.org/download/1RNA.pdb", "1RNA.pdb")

lines = open("1RNA.pdb").readlines()
c1_coords = []
for line in lines:
    if line.startswith("ATOM  ") and "C1'" in line:
        chain = line[21]
        res = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        c1_coords.append((chain, res, x, y, z))

chainA = [c for c in c1_coords if c[0] == "A"]
chainB = [c for c in c1_coords if c[0] == "B"]

cA = np.array([[c[2], c[3], c[4]] for c in chainA])
cB = np.array([[c[2], c[3], c[4]] for c in chainB])

i = 5
j = 10 - i # B's index for pairing with A[i]
# i+1 pairs with j-1
d1 = np.linalg.norm(cA[i] - cB[j-1])
d2 = np.linalg.norm(cA[i+1] - cB[j])

print(f"Distance (i to j-1) i.e. A[i]-B[j-1]: {d1:.2f}")
print(f"Distance (i+1 to j) i.e. A[i+1]-B[j]: {d2:.2f}")
