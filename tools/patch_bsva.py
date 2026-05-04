import re

def patch_file():
    with open('candidates/candidate_v90.py', 'r') as f:
        content = f.read()

    bsva_code = """
    # Geometric Backbone Tensor Smoothing (GBTS)
    # Apply local laplacian smoothing specifically around the chunk boundaries
    for i in range(1, len(chunk_ranges)):
        s_curr = chunk_ranges[i][0]
        e_prev = chunk_ranges[i - 1][1]
        mid = (s_curr + e_prev) // 2
        
        win_start = max(0, mid - 6)
        win_end = min(seq_len, mid + 6)
        
        for _ in range(5): 
            for j in range(win_start + 1, win_end - 1):
                prev_vec = full[j] - full[j-1]
                next_vec = full[j+1] - full[j]
                d1 = np.linalg.norm(prev_vec) + 1e-6
                d2 = np.linalg.norm(next_vec) + 1e-6
                
                # C1'-C1' average bond distance is ~5.95
                adj1 = prev_vec * ((5.95 - d1) / d1) * 0.12
                adj2 = next_vec * ((5.95 - d2) / d2) * 0.12
                
                full[j] += adj1 - adj2

    # Base-Stacking Vector Alignment (BSVA)
    # Maintains pi-pi orbital overlap continuity and helical coherence across interfaces
    for i in range(1, len(chunk_ranges)):
        s_curr = chunk_ranges[i][0]
        e_prev = chunk_ranges[i - 1][1]
        mid = (s_curr + e_prev) // 2
        
        win_start = max(0, mid - 10)
        win_end = min(seq_len, mid + 10)
        
        for _ in range(3):
            for j in range(win_start + 1, win_end - 2):
                v1 = full[j] - full[j-1]
                v2 = full[j+1] - full[j]
                v3 = full[j+2] - full[j+1]
                
                # Smooth the progression of successive inter-residue vectors
                # to maintain helical axis coherence and continuous base stacking
                avg_v = (v1 + v3) * 0.5
                diff = avg_v - v2
                
                bsva_strength = 0.15
                correction = diff * bsva_strength
                
                full[j] -= correction * 0.5
                full[j+1] += correction * 0.5
"""
    
    # We replace the GBTS block with GBTS + BSVA
    search_str = """    # Geometric Backbone Tensor Smoothing (GBTS)
    # Apply local laplacian smoothing specifically around the chunk boundaries
    for i in range(1, len(chunk_ranges)):
        s_curr = chunk_ranges[i][0]
        e_prev = chunk_ranges[i - 1][1]
        mid = (s_curr + e_prev) // 2
        
        win_start = max(0, mid - 6)
        win_end = min(seq_len, mid + 6)
        
        for _ in range(5): 
            for j in range(win_start + 1, win_end - 1):
                prev_vec = full[j] - full[j-1]
                next_vec = full[j+1] - full[j]
                d1 = np.linalg.norm(prev_vec) + 1e-6
                d2 = np.linalg.norm(next_vec) + 1e-6
                
                # C1'-C1' average bond distance is ~5.95
                adj1 = prev_vec * ((5.95 - d1) / d1) * 0.12
                adj2 = next_vec * ((5.95 - d2) / d2) * 0.12
                
                full[j] += adj1 - adj2"""
    
    if search_str in content:
        content = content.replace(search_str, bsva_code.strip('\n'))
        with open('notebooks/script_v91.py', 'w') as f:
            f.write(content)
        print("Success")
    else:
        print("Failed to find GBTS block")

patch_file()
