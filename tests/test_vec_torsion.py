import numpy as np

def calculate_pseudo_torsion(C):
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
    
    angles = np.arctan2(y, x)
    return np.rad2deg(angles)

radius = 8.5
rise_per_step = 2.8
twist_per_step = np.deg2rad(33.0)

C = []
for i in range(10):
    angle = i * twist_per_step
    z = i * rise_per_step
    C.append([radius * np.cos(angle), radius * np.sin(angle), z])
C = np.array(C)

print(calculate_pseudo_torsion(C))
