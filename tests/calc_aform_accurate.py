import numpy as np

def a_form_duplex(n=30, r=9.3, h=2.73, twist=32.7, theta_bp=68.7):
    twist = np.deg2rad(twist)
    theta_bp = np.deg2rad(theta_bp)
    
    s1 = []
    s2 = []
    for i in range(n):
        a1 = i * twist
        z1 = i * h
        s1.append([r * np.cos(a1), r * np.sin(a1), z1])
        
        a2 = i * twist + theta_bp
        z2 = i * h
        s2.append([r * np.cos(a2), r * np.sin(a2), z2])
    return np.array(s1), np.array(s2)

s1, s2 = a_form_duplex()

def triple_prod(p0, p1, p2, p3):
    return np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))

# Quartet: s1[i], s1[i+1], s2[i], s2[i+1]
# Note: j-1 for s2[i] is s2[i-1]? 
# If i pairs with j (which is i on s2), then i+1 pairs with i-1 on s2 (anti-parallel).
# So quartet is s1[i], s1[i+1], s2[i], s2[i+1]? 
# Let's check s1[15], s1[16], s2[15], s2[16]
tp = triple_prod(s1[15], s1[16], s2[15], s2[16])
print(f"Triple product (s1[i], s1[i+1], s2[i], s2[i+1]): {tp:.2f}")

# How about the standard chirality test p0, p1, p2, p3 on same strand
tp_ss = triple_prod(s1[15], s1[16], s1[17], s1[18])
print(f"Triple product same strand: {tp_ss:.2f}")
