import numpy as np

def get_volume(C):
    v1 = C[1:-2] - C[:-3]
    v2 = C[2:-1] - C[1:-2]
    v3 = C[3:] - C[2:-1]
    
    vol = np.sum(v1 * np.cross(v2, v3), axis=1)
    return vol.mean()

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

C1_coords = np.array(C1_coords)
print(get_volume(C1_coords[:11]))
