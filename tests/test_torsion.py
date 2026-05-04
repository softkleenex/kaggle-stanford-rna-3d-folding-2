import numpy as np

def constrain_twist(C, strength=0.1):
    L = len(C)
    if L < 4:
        return C
    
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
    # Ensure shortest path
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    
    # We want to adjust the coordinates to minimize this difference.
    # A simple pseudo-force on C[i+3] (or C[3:]) is proportional to the cross product n2 x v3?
    # Actually, to change the torsion angle without changing bond lengths/angles,
    # we can rotate C[3:] around the v2 axis.
    
    return C

print("Ready")
