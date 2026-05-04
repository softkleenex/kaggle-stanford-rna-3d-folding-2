import numpy as np

def adaptive_rna_constraints(coords, segments_map, confidence=1.0, passes=2):
    X        = coords.copy()
    segments = segments_map.get("test", [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]
            L = e - s
            if L < 3:
                continue
            
            # C1'-C1' Trace Chirality constraint
            if L >= 4:
                v1 = C[1:-2] - C[:-3]
                v2 = C[2:-1] - C[1:-2]
                v3 = C[3:] - C[2:-1]
                T = np.sum(v1 * np.cross(v2, v3), axis=1)
                mask = T < 2.0
                if np.any(mask):
                    c_v2_v3 = np.cross(v2[mask], v3[mask])
                    c_v3_v1 = np.cross(v3[mask], v1[mask])
                    c_v1_v2 = np.cross(v1[mask], v2[mask])
                    
                    rate = 0.005 * strength
                    m_idx = np.where(mask)[0]
                    np.add.at(C, m_idx,   -c_v2_v3 * rate)
                    np.add.at(C, m_idx+1, (c_v2_v3 - c_v3_v1) * rate)
                    np.add.at(C, m_idx+2, (c_v3_v1 - c_v1_v2) * rate)
                    np.add.at(C, m_idx+3,  c_v1_v2 * rate)

            X[s:e] = C
    return X

# generate zigzag
C = np.zeros((10, 3))
for i in range(10):
    C[i] = [i*3, (i%2)*5, 0] # flat zigzag, T should be 0

print("Before:")
v1 = C[1:-2] - C[:-3]
v2 = C[2:-1] - C[1:-2]
v3 = C[3:] - C[2:-1]
T = np.sum(v1 * np.cross(v2, v3), axis=1)
print(T)

C_new = adaptive_rna_constraints(C, {"test": [(0, 10)]}, confidence=0.0)

print("After:")
v1 = C_new[1:-2] - C_new[:-3]
v2 = C_new[2:-1] - C_new[1:-2]
v3 = C_new[3:] - C_new[2:-1]
T = np.sum(v1 * np.cross(v2, v3), axis=1)
print(T)

