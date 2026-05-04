import numpy as np

C = np.random.randn(6, 3)
L = len(C)
strength = 1.0

if L >= 4:
    v1 = C[1:-2] - C[:-3]
    v2 = C[2:-1] - C[1:-2]
    v3 = C[3:] - C[2:-1]
    
    cross23 = np.cross(v2, v3)
    cross31 = np.cross(v3, v1)
    cross12 = np.cross(v1, v2)
    
    stp = np.sum(v1 * cross23, axis=1)
    print("Initial STP:", stp)
    
    mask = stp < 10.0
    if np.any(mask):
        error = 10.0 - stp[mask]
        
        grad0 = -cross23[mask]
        grad1 = cross23[mask] - cross31[mask]
        grad2 = cross31[mask] - cross12[mask]
        grad3 = cross12[mask]
        
        norm = (np.linalg.norm(grad0, axis=1, keepdims=True) + 
                np.linalg.norm(grad1, axis=1, keepdims=True) + 
                np.linalg.norm(grad2, axis=1, keepdims=True) + 
                np.linalg.norm(grad3, axis=1, keepdims=True) + 1e-6)
        
        step = (error[:, None] * 0.1) / norm
        
        C[:-3][mask] += grad0 * step
        C[1:-2][mask] += grad1 * step
        C[2:-1][mask] += grad2 * step
        C[3:][mask] += grad3 * step
        
        v1 = C[1:-2] - C[:-3]
        v2 = C[2:-1] - C[1:-2]
        v3 = C[3:] - C[2:-1]
        new_stp = np.sum(v1 * np.cross(v2, v3), axis=1)
        print("Updated STP:", new_stp)
