import numpy as np

def generate_rna_structure(sequence: str) -> np.ndarray:
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords

def apply_chiral_volume_constraint(coords, strength=0.1, V_ideal=17.2):
    X = coords.copy()
    L = len(X)
    if L < 4:
        return X
    
    P0 = X[:-3]
    P1 = X[1:-2]
    P2 = X[2:-1]
    P3 = X[3:]
    
    A = P1 - P0
    B = P2 - P0
    C = P3 - P0
    
    B_cross_C = np.cross(B, C)
    C_cross_A = np.cross(C, A)
    A_cross_B = np.cross(A, B)
    
    V = np.sum(A * B_cross_C, axis=1)
    dV = V_ideal - V
    
    grad_P1 = B_cross_C
    grad_P2 = C_cross_A
    grad_P3 = A_cross_B
    grad_P0 = -(grad_P1 + grad_P2 + grad_P3)
    
    norm2 = (np.sum(grad_P0**2, axis=1) + np.sum(grad_P1**2, axis=1) + 
             np.sum(grad_P2**2, axis=1) + np.sum(grad_P3**2, axis=1) + 1e-6)
             
    alpha = strength * dV / norm2
    
    update_P0 = grad_P0 * alpha[:, None]
    update_P1 = grad_P1 * alpha[:, None]
    update_P2 = grad_P2 * alpha[:, None]
    update_P3 = grad_P3 * alpha[:, None]
    
    np.add.at(X, np.arange(0, L-3), update_P0)
    np.add.at(X, np.arange(1, L-2), update_P1)
    np.add.at(X, np.arange(2, L-1), update_P2)
    np.add.at(X, np.arange(3, L), update_P3)
    
    return X

# Test
c = generate_rna_structure("A" * 10)
print("Original Vols:", np.sum((c[1:-2]-c[:-3]) * np.cross(c[2:-1]-c[:-3], c[3:]-c[:-3]), axis=1))
c_flat = c.copy()
c_flat[:, 2] = 0 # flatten z
print("Flattened Vols:", np.sum((c_flat[1:-2]-c_flat[:-3]) * np.cross(c_flat[2:-1]-c_flat[:-3], c_flat[3:]-c_flat[:-3]), axis=1))

c_rec = apply_chiral_volume_constraint(c_flat, strength=0.5, V_ideal=17.2)
c_rec = apply_chiral_volume_constraint(c_rec, strength=0.5, V_ideal=17.2)
c_rec = apply_chiral_volume_constraint(c_rec, strength=0.5, V_ideal=17.2)
print("Recovered Vols:", np.sum((c_rec[1:-2]-c_rec[:-3]) * np.cross(c_rec[2:-1]-c_rec[:-3], c_rec[3:]-c_rec[:-3]), axis=1))
