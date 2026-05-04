import re

with open("notebooks/top1/script.py", "r") as f:
    content = f.read()

new_block = """            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # soft i–i+3  ~14.4 Å
            if L >= 4:
                d3   = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((14.4 - d3n) / d3n)[:, None] * (0.08 * strength)
                C[:-3] -= adj3; C[3:] += adj3"""

content = content.replace("""            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2""", new_block)

with open("notebooks/script_v164.py", "w") as f:
    f.write(content)
