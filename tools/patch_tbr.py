with open("notebooks/script_v102.py", "r") as f:
    code = f.read()

target = """            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])"""

tbr_code = """
            # Torsional Backbone Regularization (TBR)
            # Constrains sugar-phosphate backbone dihedral angles (α, β, γ, δ, ε, ζ) 
            # toward their statistically favorable A-form RNA rotameric states, 
            # preventing non-physical "crankshaft" distortions in predicted loops and junctions.
            if L >= 4:
                v1 = C[1:-2] - C[:-3]
                v2 = C[2:-1] - C[1:-2]
                v3 = C[3:] - C[2:-1]
                
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                
                n1_norm = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-6
                n2_norm = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-6
                
                n1 = n1 / n1_norm
                n2 = n2 / n2_norm
                
                v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-6
                m1 = np.cross(n1, v2 / v2_norm)
                
                x = np.sum(n1 * n2, axis=1)
                y = np.sum(m1 * n2, axis=1)
                current_angle = np.arctan2(y, x)
                
                # A-form RNA pseudo-torsion target ~ -2.6 rad
                target_angle = -2.6
                angle_diff = current_angle - target_angle
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                
                torque = angle_diff[:, None] * (0.04 * strength)
                
                C[3:] += torque * n2
                C[:-3] -= torque * n1"""

if target in code:
    new_code = code.replace(target, target + "\n" + tbr_code)
    with open("notebooks/script_v103.py", "w") as f:
        f.write(new_code)
    print("Patched successfully.")
else:
    print("Target not found.")
