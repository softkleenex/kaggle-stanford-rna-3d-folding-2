import numpy as np

def generate_rna_structure(n):
    coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords

C = generate_rna_structure(10)
v1 = C[1:] - C[:-1]
v1_ = v1[:-2]
v2_ = v1[1:-1]
v3_ = v1[2:]

triple = np.sum(v1_ * np.cross(v2_, v3_), axis=1)
print("Triple products:", triple)

def chirality_gradient(C):
    N = len(C)
    grad = np.zeros((N, 3))
    for i in range(N-3):
        p0, p1, p2, p3 = C[i], C[i+1], C[i+2], C[i+3]
        v1 = p1 - p0
        v2 = p2 - p1
        v3 = p3 - p2
        
        grad_p0 = -np.cross(v2, v3)
        grad_p1 = np.cross(v2, v3) - np.cross(v3, v1)
        grad_p2 = np.cross(v3, v1) - np.cross(v1, v2)
        grad_p3 = np.cross(v1, v2)
        
        # Test gradient
        epsilon = 1e-5
        for j, (p, gp) in enumerate([(p0, grad_p0), (p1, grad_p1), (p2, grad_p2), (p3, grad_p3)]):
            for dim in range(3):
                C_pos = C.copy()
                C_pos[i+j, dim] += epsilon
                v1_p = C_pos[i+1] - C_pos[i]
                v2_p = C_pos[i+2] - C_pos[i+1]
                v3_p = C_pos[i+3] - C_pos[i+2]
                T_pos = np.dot(v1_p, np.cross(v2_p, v3_p))
                
                C_neg = C.copy()
                C_neg[i+j, dim] -= epsilon
                v1_n = C_neg[i+1] - C_neg[i]
                v2_n = C_neg[i+2] - C_neg[i+1]
                v3_n = C_neg[i+3] - C_neg[i+2]
                T_neg = np.dot(v1_n, np.cross(v2_n, v3_n))
                
                grad_num = (T_pos - T_neg) / (2 * epsilon)
                assert np.abs(grad_num - gp[dim]) < 1e-4, f"Mismatch: {grad_num} vs {gp[dim]}"
    return "Gradient verified!"

print(chirality_gradient(C))
