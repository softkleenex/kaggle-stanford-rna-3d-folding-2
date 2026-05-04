import numpy as np

def generate_aform_c1(n):
    # Accurate A-form RNA C1' coordinates.
    # From literature, C1' atoms are located at a radius of ~8.5-8.9 A from the helical axis.
    # The twist is ~32.7 degrees, and the rise is ~2.81 A.
    # Wait, 11 bp/turn, pitch = 30.9 A. So rise = 30.9 / 11 = 2.81 A.
    # Let's adjust radius so that i->i+1 is exactly 5.95 A.
    # D^2 = R^2 + R^2 - 2 R^2 cos(twist) + rise^2
    # 5.95^2 = 2 R^2 (1 - cos(32.7 deg)) + 2.81^2
    # 35.4025 = 2 R^2 (1 - 0.8415) + 7.8961
    # 27.5064 = 2 R^2 (0.1585)
    # R^2 = 27.5064 / 0.317 = 86.77
    # R = 9.31 A
    r = 9.31
    twist = np.radians(32.7)
    rise = 2.81
    
    coords = []
    for i in range(n):
        ang = i * twist
        coords.append([r * np.cos(ang), r * np.sin(ang), i * rise])
    return np.array(coords)

c = generate_aform_c1(15)
for k in range(1, 12):
    dist = np.linalg.norm(c[k] - c[0])
    print(f"{k}: {dist:.2f},")
