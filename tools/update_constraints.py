import re

with open("notebooks/top1/script.py", "r") as f:
    code = f.read()

new_func = """def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            # soft i–i+2  ~10.2 Å
            if L >= 3:
                d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
                adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
                C[:-2] -= adj2; C[2:] += adj2
                
            # soft i-i+3 ~14.4 Å
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((14.4 - d3n) / d3n)[:, None] * (0.08 * strength)
                C[:-3] -= adj3; C[3:] += adj3
                
            # soft i-i+4 ~17.5 Å
            if L >= 5:
                d4 = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((17.5 - d4n) / d4n)[:, None] * (0.06 * strength)
                C[:-4] -= adj4; C[4:] += adj4

            # soft i–i+5  ~19.0 Å
            if L >= 6:
                d5   = C[5:] - C[:-5]; d5n = np.linalg.norm(d5, axis=1) + 1e-6
                adj5 = d5 * ((19.0 - d5n) / d5n)[:, None] * (0.05 * strength)
                C[:-5] -= adj5; C[5:] += adj5
            
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # self-avoidance (upgraded to 4.0 A)
            if L >= 10:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 3) & (dm < 4.0)
                if np.any(mask):
                    vec = (diff * ((4.0 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.02 * strength) * vec
            X[s:e] = C
    return X"""

code = re.sub(r'def adaptive_rna_constraints\(.*?return X', new_func, code, flags=re.DOTALL)

# Fix syntax issues with os.system for pip
code = re.sub(r'!pip install(.*?)\n', r'os.system("pip install\1")\n', code)

with open("notebooks/script_v176.py", "w") as f:
    f.write(code)

print("Created notebooks/script_v176.py successfully!")
