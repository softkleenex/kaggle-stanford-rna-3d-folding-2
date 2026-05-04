import numpy as np

def generate_aform_c1_duplex(n):
    # Two strands, anti-parallel
    coords1 = []
    coords2 = []
    r = 8.5
    twist = np.radians(32.7)
    rise = 2.81
    for i in range(n):
        ang1 = i * twist
        coords1.append([r * np.cos(ang1), r * np.sin(ang1), i * rise])
        
        ang2 = ang1 + np.pi # roughly opposite
        # But wait, A-form major/minor groove means the angle is not exactly pi.
        # Let's say it's pi.
        coords2.append([r * np.cos(ang2), r * np.sin(ang2), i * rise])
    
    # Reverse strand 2 to make it anti-parallel
    coords2.reverse()
    return np.array(coords1), np.array(coords2)

c1, c2 = generate_aform_c1_duplex(15)

# Distance between C1_0 on strand 1 and all C1 on strand 2
for i in range(15):
    dist = np.linalg.norm(c1[7] - c2[i])
    print(f"c1[7] to c2[{i}]: {dist:.2f} A")
