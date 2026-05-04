import re

with open("notebooks/script_v277.py", "r") as f:
    content = f.read()

old_block = """            # STRATEGY: 12.8 Å distance constraint P(i)-P(j-2) for base-paired residues (i, j)
            # Maintain A-form helical pitch and stabilize inter-strand minor groove diagonal.
            # (Assuming self-pairing for pseudo-helical stabilization in de-novo/stitching)
            if L >= 10:
                for i in range(L - 4):
                    # We look for potential i, j base-pairs. In canonical A-RNA, P(i)-P(j-2) diagonal
                    # across the minor groove is ~12.8A.
                    j = i + 6 # Heuristic: looking across 6 residues
                    if j < L:
                        v_diag = C[j-2] - C[i]
                        d_diag = np.linalg.norm(v_diag) + 1e-6
                        adj_diag = v_diag * ((12.8 - d_diag) / d_diag) * (0.05 * strength)
                        C[i] -= adj_diag * 0.5
                        C[j-2] += adj_diag * 0.5"""

new_block = """            # STRATEGY: Implement a 18.5 Å distance constraint between phosphorus atoms P(i) and P(j+2) for base-paired residues (i, j) to stabilize the wide minor groove and maintain the characteristic shallow groove architecture of A-form RNA.
            if L >= 10:
                for i in range(L - 8):
                    j = i + 6 # Heuristic: looking across 6 residues for base-pairing
                    if j + 2 < L:
                        v_diag = C[j+2] - C[i]
                        d_diag = np.linalg.norm(v_diag) + 1e-6
                        adj_diag = v_diag * ((18.5 - d_diag) / d_diag) * (0.05 * strength)
                        C[i] -= adj_diag * 0.5
                        C[j+2] += adj_diag * 0.5"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open("notebooks/script_v278.py", "w") as f:
        f.write(content)
    print("Successfully replaced and wrote to notebooks/script_v278.py")
else:
    print("Could not find the block to replace!")
