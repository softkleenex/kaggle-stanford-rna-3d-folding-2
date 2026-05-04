import numpy as np

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2):
    X = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]
            L = e - s
            if L < 3:
                continue
            
            # Comprehensive PDB-derived local distance profile
            ideal_dists = {
                1: (5.60, 0.22),
                2: (10.45, 0.10),
                3: (14.40, 0.08),
                4: (17.40, 0.06),
                5: (19.45, 0.05),
                6: (20.60, 0.04),
                7: (21.05, 0.03),
                8: (21.15, 0.02)
            }
            for k, (target_dist, weight) in ideal_dists.items():
                if L > k:
                    dk = C[k:] - C[:-k]
                    dkn = np.linalg.norm(dk, axis=1) + 1e-6
                    adj = dk * ((target_dist - dkn) / dkn)[:, None] * (weight * strength)
                    C[:-k] -= adj
                    C[k:] += adj

            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.5)
                if np.any(mask):
                    vec = (diff * ((3.5 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X

# Test run
coords = np.random.rand(50, 3) * 10
C_new = adaptive_rna_constraints(coords, 'T1', {'T1': [(0, 50)]})
print("Success!")
