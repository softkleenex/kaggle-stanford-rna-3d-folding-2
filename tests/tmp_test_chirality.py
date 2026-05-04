import numpy as np

def generate_a_form_c1(n=15):
    coords = []
    r = 9.3 # estimated C1' radius
    twist = np.deg2rad(32.7)
    rise = 2.81
    for i in range(n):
        x = r * np.cos(i * twist)
        y = r * np.sin(i * twist)
        z = i * rise
        coords.append([x, y, z])
    return np.array(coords)

coords = generate_a_form_c1(15)
v = coords[1:] - coords[:-1]
stp = []
for i in range(len(v)-2):
    v1 = v[i]
    v2 = v[i+1]
    v3 = v[i+2]
    val = np.dot(v1, np.cross(v2, v3))
    stp.append(val)
print("Scalar triple products:", stp[:5])
