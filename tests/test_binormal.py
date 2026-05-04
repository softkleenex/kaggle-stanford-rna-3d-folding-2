import numpy as np

import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://files.rcsb.org/download/1RNA.pdb"
response = urllib.request.urlopen(url, context=ctx)
lines = response.read().decode('utf-8').split('\n')

C1_coords = []
for line in lines:
    if line.startswith("ATOM  ") and "C1'" in line:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        C1_coords.append([x, y, z])

C = np.array(C1_coords)[:11]

v = C[1:] - C[:-1]
b = np.cross(v[:-1], v[1:])
b_norm = np.linalg.norm(b, axis=1) + 1e-6
b_unit = b / b_norm[:, None]

dot = np.sum(b_unit[:-1] * b_unit[1:], axis=1)
angles = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))

print("Angles between binormals:", angles)
print("Mean angle:", angles.mean())
