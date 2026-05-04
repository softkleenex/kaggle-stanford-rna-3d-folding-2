import sys
import json

def extract_code(ipynb_path, out_path):
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code_cells = []
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = "".join(source)
                code_cells.append(source)
        
        full_code = "\n\n# --- NEXT CELL ---\n\n".join(code_cells)
        # Limit length just in case it's absurdly huge (save LLM context)
        lines = full_code.split('\n')
        if len(lines) > 600:
            full_code = "\n".join(lines[:600]) + "\n\n# ... [TRUNCATED DUE TO LENGTH] ..."
            
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
    except Exception as e:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"# Failed to parse notebook: {e}")

if __name__ == '__main__':
    extract_code(sys.argv[1], sys.argv[2])
