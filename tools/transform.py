import re

with open("notebooks/script_v98.py", "r") as f:
    code = f.read()

# I will add the BSOC constraint in the `adaptive_rna_constraints` function
# I can just append it before the line `C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])`

bsoc_code = """
            # STRATEGY: Base-Stacking Orientation Constraint (BSOC)
            # Enforce parallel alignment of consecutive base rings (3.4 Å spacing)
            if L >= 4:
                v1 = C[1:-2] - C[:-3]
                v2 = C[2:-1] - C[1:-2]
                v3 = C[3:] - C[2:-1]
                
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                
                n1_u = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-6)
                n2_u = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-6)
                
                cross_n = np.cross(n1_u, n2_u)
                align_force = cross_n * (0.10 * strength)
                
                C[:-3] += align_force * 0.5
                C[1:-2] -= align_force * 0.5
                C[2:-1] += align_force * 0.5
                C[3:] -= align_force * 0.5
                
                avg_n = n1_u + n2_u
                avg_n_u = avg_n / (np.linalg.norm(avg_n, axis=1, keepdims=True) + 1e-6)
                
                proj = np.sum(v2 * avg_n_u, axis=1, keepdims=True)
                rise_adj = (3.4 * np.sign(proj) - proj) * avg_n_u * (0.15 * strength)
                
                C[1:-2] -= rise_adj * 0.5
                C[2:-1] += rise_adj * 0.5
"""

# Find the insertion point
target = "C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])"
if target in code:
    code = code.replace(target, bsoc_code + "\n            " + target)
else:
    print("Could not find insertion point!")

with open("notebooks/script_v99.py", "w") as f:
    f.write(code)

print("Saved to notebooks/script_v99.py")
