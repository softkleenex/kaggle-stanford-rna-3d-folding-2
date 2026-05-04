import numpy as np
import sys
import os

# Add notebooks to path to import script_v280
sys.path.append(os.path.join(os.getcwd(), "notebooks"))
import script_v280

def test_p_i_i2_constraint():
    print("Testing P(i)-P(i+2) constraint in script_v280.py...")
    
    # Create a dummy sequence and segments map
    sequence = "AAAAAAAAAA"
    target_id = "test_target"
    segments_map = {target_id: [(0, 10)]}
    
    # Create initial coordinates (stretched out)
    coords = np.zeros((10, 3))
    for i in range(10):
        coords[i] = [0, 0, i * 8.0] # 8A step between P atoms, so 16A between P(i) and P(i+2)
        
    # Apply constraints (high confidence to ensure stems are detected)
    # We'll mock the detect_simple_stems to return a full stem for the test
    original_detect = script_v280.detect_simple_stems
    script_v280.detect_simple_stems = lambda c: [[(i, 9-i) for i in range(5)]]
    
    try:
        # Run with many passes and high strength to see the effect clearly
        # Note: adaptive_rna_constraints uses confidence to calculate strength
        # strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
        # So low confidence = high strength.
        constrained_coords = script_v280.adaptive_rna_constraints(coords, target_id, segments_map, confidence=0.0, passes=20)
        
        # Check P(i)-P(i+2) distances in the helical region
        distances = []
        for i in range(8):
            dist = np.linalg.norm(constrained_coords[i] - constrained_coords[i+2])
            distances.append(dist)
            
        avg_dist = np.mean(distances)
        print(f"Average P(i)-P(i+2) distance after constraints: {avg_dist:.2f} A")
        
        # Initial was 16.0A, target is 14.1A.
        if avg_dist < 15.5:
            print("SUCCESS: P(i)-P(i+2) distance shifted towards 14.1 A.")
        else:
            print(f"FAILURE: P(i)-P(i+2) distance ({avg_dist:.2f} A) did not shift enough from 16.0 A.")
            
    finally:
        script_v280.detect_simple_stems = original_detect

if __name__ == "__main__":
    test_p_i_i2_constraint()
