import numpy as np

def a_form_c1_prime(n=20, r=9.3):
    h = 2.73
    twist = np.deg2rad(32.7)
    coords = []
    for i in range(n):
        angle = i * twist
        z = i * h
        coords.append([r * np.cos(angle), r * np.sin(angle), z])
    return np.array(coords)

coords = a_form_c1_prime()
v1 = coords[0] - coords[1]
v2 = coords[2] - coords[1]
cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.rad2deg(np.arccos(cos_angle))
print(f"Angle C1'(i-1)-C1'(i)-C1'(i+1): {angle:.2f} deg")

d2 = np.linalg.norm(coords[2] - coords[0])
print(f"Distance C1'(i-1)-C1'(i+1): {d2:.2f} A")
