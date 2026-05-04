import numpy as np
def test_angle():
    C = np.array([[0,0,0], [5,0,0], [5+5*np.cos(np.deg2rad(65)), 5*np.sin(np.deg2rad(65)), 0]])
    v1 = C[0] - C[1]
    v2 = C[2] - C[1]
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    print(np.rad2deg(np.arccos(np.dot(v1, v2))))

test_angle()
