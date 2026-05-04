import sys

with open("notebooks/script_v98.py", "r", encoding="utf-8") as f:
    code = f.read()

old_block = """            # STRATEGY: Base-Stacking Geometry Constraint (BSGC)
            # Enforce parallel orientation and optimal overlap between adjacent nucleobases in helical stems
            if L >= 4:"""

new_block = """            # STRATEGY: Base-Stacking Orientation Constraint (BSOC)
            # Enforcing parallel alignment of consecutive base rings (3.4 \u00c5 spacing) to stabilize helical regions and improve local folding accuracy.
            if L >= 4:"""

if old_block in code:
    code = code.replace(old_block, new_block)
    print("Replaced BSGC with BSOC")
else:
    print("Error: Could not find old_block")
    sys.exit(1)

with open("notebooks/script_v99.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Saved to notebooks/script_v99.py")
