import re

with open('notebooks/top1/script.py', 'r') as f:
    code = f.read()

old_func = """def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
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
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # soft i–i+5  ~19.0 Å
            if L >= 6:
                d5   = C[5:] - C[:-5]; d5n = np.linalg.norm(d5, axis=1) + 1e-6
                adj5 = d5 * ((19.0 - d5n) / d5n)[:, None] * (0.05 * strength)
                C[:-5] -= adj5; C[5:] += adj5
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X"""

new_func = """def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.85 * (1.0 - min(confidence, 0.98)), 0.05)
    
    # Data-driven ENM targets derived from training set median C1' distances
    targets = [
        (1, 5.65, 0.25),
        (2, 10.65, 0.15),
        (3, 14.88, 0.10),
        (4, 18.20, 0.08),
        (5, 20.66, 0.05)
    ]
    
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            
            # Apply Elastic Network Model constraints
            for k, tgt_d, w in targets:
                if L > k:
                    d = C[k:] - C[:-k]
                    dn = np.linalg.norm(d, axis=1) + 1e-6
                    adj = d * ((tgt_d - dn) / dn)[:, None] * (w * strength)
                    C[:-k] -= adj
                    C[k:] += adj
                    
            # 4th-order Biharmonic smoothing for high-frequency noise removal while preserving curvature
            if L >= 5:
                lap2 = -C[:-4] + 4*C[1:-3] - 6*C[2:-2] + 4*C[3:-1] - C[4:]
                C[2:-2] += (0.02 * strength) * lap2
            elif L >= 3:
                C[1:-1] += (0.05 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
                
            # self-avoidance (excluded volume)
            if L >= 20:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 3) & (dm < 3.8) # Increased exclusion radius and separation
                if np.any(mask):
                    vec = (diff * ((3.8 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.02 * strength) * vec
            X[s:e] = C
    return X"""

code = code.replace(old_func, new_func)

with open('notebooks/script_v167.py', 'w') as f:
    f.write(code)

print("Replaced and saved to notebooks/script_v167.py")
