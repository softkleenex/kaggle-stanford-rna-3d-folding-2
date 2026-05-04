import numpy as np
R, h = 10.0, 2.5
theta = np.deg2rad(32.7)
def p(i): return np.array([R*np.cos(i*theta), R*np.sin(i*theta), i*h])
v1 = p(1) - p(0)
v2 = p(2) - p(1)
v3 = p(3) - p(2)
vol = np.dot(v1, np.cross(v2, v3))
print("Volume:", vol)
