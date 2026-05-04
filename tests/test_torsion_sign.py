import numpy as np
import Bio.PDB

# approximate A-form RNA C1' coords
# Actually let's just make a small helix and check it
theta = np.linspace(0, 4*np.pi, 20)
z = np.linspace(0, 10, 20)
r = 5.0
x = r * np.cos(theta)
y = r * np.sin(theta)
C = np.column_stack([x, y, z])
v1 = C[1:-2] - C[:-3]
v2 = C[2:-1] - C[1:-2]
v3 = C[3:] - C[2:-1]
cross23 = np.cross(v2, v3)
dot1 = np.sum(v1 * cross23, axis=1)
print("Ideal right-handed helix triple product sign:", np.sign(dot1))
