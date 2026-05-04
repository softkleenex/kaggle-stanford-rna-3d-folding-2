import re

def modify_script():
    with open('notebooks/top1/script.py', 'r') as f:
        content = f.read()

    # 1. Modify adaptive_rna_constraints
    old_func = """def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # Laplacian smoothing"""
    
    new_func = """def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            # soft i–i+2  ~10.2 Å
            if L >= 3:
                d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
                adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
                C[:-2] -= adj2; C[2:] += adj2
                
            # NEW: soft i-i+3 ~ 14.0 Å (Helical turn constraint)
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((14.0 - d3n) / d3n)[:, None] * (0.05 * strength)
                C[:-3] -= adj3; C[3:] += adj3
                
            # Laplacian smoothing"""
            
    if old_func in content:
        content = content.replace(old_func, new_func)
    else:
        print("ERROR: old_func not found!")
        
    # 2. Add local validation function before main()
    val_func = """
def do_local_validation(val_df, pool_df, coords_dict, seg_map):
    print("\\n--- RUNNING LOCAL VALIDATION ---")
    rmsds = []
    for _, row in val_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]
        if tid not in coords_dict:
            continue
        gt_coords = coords_dict[tid]
        
        similar = find_similar_sequences_detailed(seq, pool_df, coords_dict, top_n=5)
        if len(similar) > 0:
            _, tmpl_seq, sim, tmpl_coords, _, _, _ = similar[0]
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            pred = adaptive_rna_constraints(adapted, tid, seg_map, confidence=sim)
        else:
            pred = generate_rna_structure(seq)
            pred = adaptive_rna_constraints(pred, tid, seg_map, confidence=0.2)
            
        mlen = min(len(pred), len(gt_coords))
        if mlen < 3: continue
        P = pred[:mlen]
        Q = gt_coords[:mlen]
        valid = ~(np.isnan(P).any(axis=1) | np.isnan(Q).any(axis=1))
        P = P[valid]; Q = Q[valid]
        if len(P) >= 3:
            R, t = kabsch_align(P, Q)
            P_aligned = P @ R.T + t
            diff = P_aligned - Q
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            rmsds.append(rmsd)
            
    final_rmsd = np.mean(rmsds) if rmsds else 9.999
    print(f"LOCAL_VALIDATION_RMSD: {final_rmsd:.5f}")
    print("--------------------------------\\n")

def main() -> None:"""

    content = content.replace("def main() -> None:", val_func)
    
    # 3. Call local validation in main
    main_insertion_old = """    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────"""
    
    main_insertion_new = """    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")
    
    # +++ LOCAL VALIDATION +++
    val_size = min(10, len(train_seqs))
    if val_size > 0:
        local_val_seqs = train_seqs.tail(val_size).copy()
        local_pool_seqs = train_seqs.iloc[:-val_size].copy()
        do_local_validation(local_val_seqs, local_pool_seqs, train_coords, segments_map)

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────"""
    
    if main_insertion_old in content:
        content = content.replace(main_insertion_old, main_insertion_new)
    else:
        print("ERROR: main_insertion_old not found!")

    with open('notebooks/script_v147.py', 'w') as f:
        f.write(content)
    print("Modification successful.")

if __name__ == '__main__':
    modify_script()
