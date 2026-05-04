import numpy as np
import sys
import os

# Add current directory to path to import script
sys.path.append(os.getcwd())

# Mock segments map for testing
segments_map = {"test": [(0, 10)]}

def generate_test_structure(n=10):
    coords = np.zeros((n, 3))
    for i in range(n):
        # Linear structure with 4.0 A spacing
        coords[i] = [0, 0, i * 4.0]
    return coords

# We need to import adaptive_rna_constraints from the script
# Since it's in a notebook-style script, we might need to copy it or import it carefully
# Let's try to import it by relative path or just copy the essential part for verification

from notebooks.script_v253 import adaptive_rna_constraints

def test_ippc_constraint():
    print("Testing IPPC (16.8 A) constraint...")
    
    # Create a structure where P(i)-P(i+3) distance is 12.0 A (4.0 * 3)
    # We want it to move towards 16.8 A
    coords = generate_test_structure(10)
    
    # Calculate initial P(0)-P(3) distance
    initial_dist = np.linalg.norm(coords[3] - coords[0])
    print(f"Initial P(0)-P(3) distance: {initial_dist:.2f} A")
    
    # Apply constraints (confidence=0.0 to maximize strength)
    # We need to pass stems or it won't trigger in_stem check in IPPC
    # Actually, in script_v253, IPPC is guarded by in_stem
    
    # Mock stems to include 0 and 3
    # detect_simple_stems will be called inside, so we rely on it or mock it.
    # In script_v253, stems are detected automatically.
    # 12.0 A is within the dist_threshold (12.5) of detect_simple_stems.
    
    refined = adaptive_rna_constraints(coords, "test", segments_map, confidence=0.0, passes=5)
    
    final_dist = np.linalg.norm(refined[3] - refined[0])
    print(f"Final P(0)-P(3) distance: {final_dist:.2f} A")
    
    if final_dist > initial_dist:
        print("SUCCESS: Distance increased towards 16.8 A target.")
    else:
        print("FAILURE: Distance did not increase.")

if __name__ == "__main__":
    try:
        test_ippc_constraint()
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
