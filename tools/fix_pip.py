import re

def fix_pip():
    with open('notebooks/script_v147.py', 'r') as f:
        content = f.read()
        
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('!pip install'):
            cmd = stripped[1:] # remove !
            cmd = cmd.replace('"', '\\"')
            new_lines.append(f'import os; os.system("{cmd}")')
        else:
            new_lines.append(line)
            
    with open('notebooks/script_v147.py', 'w') as f:
        f.write('\n'.join(new_lines))

if __name__ == '__main__':
    fix_pip()
