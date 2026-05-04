import numpy as np

R = 9.3
phi = np.radians(32.7)
h = 2.6

coords = []
for i in range(12):
    coords.append([R * np.cos(i * phi), R * np.sin(i * phi), i * h])
coords = np.array(coords)

d1 = np.linalg.norm(coords[1] - coords[0])
d2 = np.linalg.norm(coords[2] - coords[0])
d3 = np.linalg.norm(coords[3] - coords[0])
d4 = np.linalg.norm(coords[4] - coords[0])
d5 = np.linalg.norm(coords[5] - coords[0])
d11 = np.linalg.norm(coords[11] - coords[0])

v1 = coords[0] - coords[1]
v2 = coords[2] - coords[1]
angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

print(f"i-i+1: {d1:.2f} A")
print(f"i-i+2: {d2:.2f} A")
print(f"i-i+3: {d3:.2f} A")
print(f"i-i+4: {d4:.2f} A")
print(f"i-i+5: {d5:.2f} A")
print(f"i-i+11: {d11:.2f} A")
print(f"Angle i-1, i, i+1: {angle:.2f} deg")
