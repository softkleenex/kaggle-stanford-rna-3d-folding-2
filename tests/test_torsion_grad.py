import numpy as np
import torch
import time

def apply_torsion_constraint(C, strength=0.05, target_deg=33.0):
    if len(C) < 4:
        return C
    
    T = torch.tensor(C, dtype=torch.float32, requires_grad=True)
    p0 = T[:-3]
    p1 = T[1:-2]
    p2 = T[2:-1]
    p3 = T[3:]
    
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    
    b1_norm = torch.linalg.norm(b1, dim=1, keepdim=True) + 1e-8
    b1_u = b1 / b1_norm
    
    v = torch.cross(b0, b1)
    w = torch.cross(b1, b2)
    
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True) + 1e-8
    w_norm = torch.linalg.norm(w, dim=1, keepdim=True) + 1e-8
    
    v_u = v / v_norm
    w_u = w / w_norm
    
    cos_theta = torch.sum(v_u * w_u, dim=1)
    cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
    
    # Actually, simpler is to just use standard PyTorch operations
    x = torch.cross(v_u, w_u)
    sign = torch.sign(torch.sum(x * b1_u, dim=1))
    sin_theta = torch.sqrt(1 - cos_theta**2) * sign
    
    angle_rad = torch.atan2(sin_theta, cos_theta)
    angle_deg = angle_rad * (180.0 / np.pi)
    
    diff = angle_deg - target_deg
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
    
    loss = torch.sum(diff**2)
    loss.backward()
    
    # Apply gradient descent step
    # We want to minimize loss, so C -= lr * grad
    lr = 0.001 * strength
    return C - lr * T.grad.numpy()

C = np.random.randn(100, 3)
t0 = time.time()
for _ in range(10):
    C = apply_torsion_constraint(C)
print(f"Time: {time.time() - t0:.4f}s")
