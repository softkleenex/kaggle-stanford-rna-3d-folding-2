with open("notebooks/script_v86.py", "r") as f:
    lines = f.readlines()

new_lines = []
has_sys = False
for line in lines:
    if line.strip().startswith("!pip install"):
        parts = line.strip()[1:].split()
        import_stmt = "import sys\nimport subprocess\n" if not has_sys else ""
        has_sys = True
        
        args_str = ", ".join([f'"{p}"' for p in parts])
        new_lines.append(f"{import_stmt}subprocess.check_call([sys.executable, '-m', {args_str}])\n")
    else:
        new_lines.append(line)

with open("notebooks/script_v86.py", "w") as f:
    f.writelines(new_lines)
