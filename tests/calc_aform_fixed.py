import numpy as np

def a_form_duplex_ideal(n=20, r=9.3):
    h = 2.73
    twist = np.deg2rad(32.7)
    theta_bp = np.deg2rad(68.7) # Angle between C1' of a pair in XY plane
    s1 = []
    s2 = []
    for i in range(n):
        # Strand 1
        angle1 = i * twist
        z1 = i * h
        s1.append([r * np.cos(angle1), r * np.sin(angle1), z1])
        # Strand 2 (anti-parallel, same Z level for paired bases)
        angle2 = i * twist + theta_bp
        z2 = i * h
        s2.append([r * np.cos(angle2), r * np.sin(angle2), z2])
    return np.array(s1), np.array(s2)

s1, s2 = a_form_duplex_ideal()
print(f"BP C1'-C1' (i, i): {np.linalg.norm(s1[10] - s2[10]):.3f} A")
print(f"Diagonal (i, i-1): {np.linalg.norm(s1[10] - s2[9]):.3f} A")
print(f"Diagonal (i, i+1): {np.linalg.norm(s1[10] - s2[11]):.3f} A")
print(f"Diagonal (i, i-2): {np.linalg.norm(s1[10] - s2[8]):.3f} A")
print(f"Diagonal (i, i+2): {np.linalg.norm(s1[10] - s2[12]):.3f} A")
