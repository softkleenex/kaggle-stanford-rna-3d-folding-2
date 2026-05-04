import numpy as np

def vectorized_pseudo_torsion_grad(p0, p1, p2, p3, target_angle=0.24):
    pts = [p0, p1, p2, p3]
    grads = [np.zeros_like(p) for p in pts]
    eps = 1e-4
    
    def calc_ang(P):
        b0 = -1.0 * (P[1] - P[0])
        b1 = P[2] - P[1]
        b2 = P[3] - P[2]
        b1_u = b1 / (np.linalg.norm(b1, axis=1, keepdims=True) + 1e-8)
        v = b0 - np.sum(b0 * b1_u, axis=1, keepdims=True) * b1_u
        w = b2 - np.sum(b2 * b1_u, axis=1, keepdims=True) * b1_u
        x = np.sum(v * w, axis=1)
        y = np.sum(np.cross(b1_u, v) * w, axis=1)
        return np.arctan2(y, x)
    
    for i in range(4):
        for j in range(3):
            pts_plus = [p.copy() for p in pts]
            pts_plus[i][:, j] += eps
            
            pts_minus = [p.copy() for p in pts]
            pts_minus[i][:, j] -= eps
            
            a_plus = calc_ang(pts_plus)
            a_minus = calc_ang(pts_minus)
            
            diff_plus = (a_plus - target_angle)
            diff_plus = (diff_plus + np.pi) % (2*np.pi) - np.pi
            
            diff_minus = (a_minus - target_angle)
            diff_minus = (diff_minus + np.pi) % (2*np.pi) - np.pi
            
            err_plus = diff_plus**2
            err_minus = diff_minus**2
            
            grads[i][:, j] = (err_plus - err_minus) / (2 * eps)
            
    return grads

p0 = np.array([[0,0,0]], dtype=float)
p1 = np.array([[5,0,0]], dtype=float)
p2 = np.array([[5,5,0]], dtype=float)
p3 = np.array([[0,5,5]], dtype=float)

grads = vectorized_pseudo_torsion_grad(p0, p1, p2, p3, target_angle=0.24)
print(grads[0])
print(np.linalg.norm(grads[0]))

