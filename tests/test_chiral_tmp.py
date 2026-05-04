import numpy as np

def generate_rna_structure(sequence: str) -> np.ndarray:
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        # right-handed helix rotating along Z
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.8]
    return coords

coords = generate_rna_structure("A" * 10)
v1 = coords[1:-2] - coords[:-3]
v2 = coords[2:-1] - coords[1:-2]
v3 = coords[3:] - coords[2:-1]

stp = np.einsum('ij,ij->i', v1, np.cross(v2, v3))
print("Triple product:", stp)

# Volume of tetrahedron
v1_a = coords[1:-2] - coords[:-3]
v2_a = coords[2:-1] - coords[:-3]
v3_a = coords[3:] - coords[:-3]
vol = np.einsum('ij,ij->i', v1_a, np.cross(v2_a, v3_a))
print("Volume:", vol)
