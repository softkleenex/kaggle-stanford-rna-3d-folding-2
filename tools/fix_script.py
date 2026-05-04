with open("notebooks/script_v174.py", "r") as f:
    content = f.read()

content = content.replace("consecutive '$ atoms", "consecutive C1' atoms")

with open("notebooks/script_v174.py", "w") as f:
    f.write(content)
