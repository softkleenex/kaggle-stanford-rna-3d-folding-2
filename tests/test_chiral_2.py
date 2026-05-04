import numpy as np

def apply_chiral_constraint(C, strength=1.0):
    L = len(C)
    if L >= 4:
        P0 = C[:-3]
        P1 = C[1:-2]
        P2 = C[2:-1]
        P3 = C[3:]
        
        A = P1 - P0
        B = P2 - P0
        C_vec = P3 - P0
        
        B_cross_C = np.cross(B, C_vec)
        C_cross_A = np.cross(C_vec, A)
        A_cross_B = np.cross(A, B)
        
        V = np.sum(A * B_cross_C, axis=1)
        V_ideal = 17.2
        
        dV = V_ideal - V
        mask = dV > 0
        if np.any(mask):
            grad_P1 = B_cross_C
            grad_P2 = C_cross_A
            grad_P3 = A_cross_B
            grad_P0 = -(grad_P1 + grad_P2 + grad_P3)
            
            norm2 = (np.sum(grad_P0**2, axis=1) + np.sum(grad_P1**2, axis=1) + 
                     np.sum(grad_P2**2, axis=1) + np.sum(grad_P3**2, axis=1) + 1e-6)
                     
            alpha = (0.15 * strength) * dV / norm2
            alpha = alpha * mask  # apply only where volume is too small
            
            C[:-3] += grad_P0 * alpha[:, None]
            C[1:-2] += grad_P1 * alpha[:, None]
            C[2:-1] += grad_P2 * alpha[:, None]
            C[3:] += grad_P3 * alpha[:, None]
    return C

C = np.random.rand(10, 3)
C = apply_chiral_constraint(C)
print("Done")
