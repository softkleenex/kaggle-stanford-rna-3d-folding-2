import numpy as np

def generate_a_form_c1(n=15):
    # A-form RNA parameters
    # radius ~ 8.5 A
    # twist ~ 32.7 deg
    # rise ~ 2.81 A
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
dists = []
for k in range(1, 12):
    d = np.linalg.norm(coords[k] - coords[0])
    dists.append(d)

print("A-form C1' distances:")
for k, d in enumerate(dists, 1):
    print(f"i to i+{k}: {d:.2f} A")
