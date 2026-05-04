with open("notebooks/script_v146.py", "r") as f:
    lines = f.readlines()

with open("notebooks/script_v146.py", "w") as f:
    for line in lines:
        if line.startswith("!pip install"):
            f.write(f'import os; os.system("{line.strip()[1:]}")\n')
        else:
            f.write(line)
