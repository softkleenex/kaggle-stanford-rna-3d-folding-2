import numpy as np

def calculate_dihedral(p0, p1, p2, p3):
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2 /= np.linalg.norm(b2)
    m1 = np.cross(n1, b2)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))

R = 9.3
phi = np.radians(32.7)
h = 2.6

coords = []
for i in range(4):
    coords.append([R * np.cos(i * phi), R * np.sin(i * phi), i * h])
coords = np.array(coords)

angle = calculate_dihedral(*coords)
print(f"A-form C1' dihedral (i, i+1, i+2, i+3): {angle:.2f} degrees")
