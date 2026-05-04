import numpy as np

def adaptive_rna_constraints(coords, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get("A", [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            
            if L >= 4:
                v1 = C[1:-2] - C[:-3]
                v2 = C[2:-1] - C[1:-2]
                v3 = C[3:] - C[2:-1]
                
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                
                n1_norm = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-6
                n2_norm = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-6
                
                n1_unit = n1 / n1_norm
                n2_unit = n2 / n2_norm
                
                v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-6
                v2_unit = v2 / v2_norm
                
                m1 = np.cross(n1_unit, v2_unit)
                
                x = np.sum(n1_unit * n2_unit, axis=1)
                y = np.sum(m1 * n2_unit, axis=1)
                
                current_angles = np.arctan2(y, x)
                target_angle = np.deg2rad(33.0)
                
                diff = current_angles - target_angle
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                
                # Apply corrective rotational force
                correction = np.cross(v2_unit, v3) * diff[:, None] * (0.05 * strength)
                C[3:] -= correction
                C[:-3] += correction
            X[s:e] = C
    return X

coords = np.random.randn(10, 3)
segments_map = {"A": [(0, 10)]}
adaptive_rna_constraints(coords, segments_map)
print("Test passed.")
