import re

with open("notebooks/top1/script.py", "r") as f:
    code = f.read()

# 1. Replace !pip install with os.system
code = re.sub(r"!pip install (.*)\n", r'os.system("pip install \1")\n', code)

# 2. Fix IS_KAGGLE and LOCAL_N_SAMPLES
code = code.replace(
    'IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))',
    'IS_KAGGLE = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))'
)
code = code.replace(
    'LOCAL_N_SAMPLES = None',
    'LOCAL_N_SAMPLES = 2'
)

# 3. Dynamic DATA_BASE
code = code.replace(
    'DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"',
    '''def get_data_base():
    from pathlib import Path
    for p in Path("/kaggle/input").rglob("test_sequences.csv"):
        return str(p.parent)
    return "/kaggle/input/stanford-rna-3d-folding-2"

DATA_BASE              = get_data_base()'''
)

# 4. Modify stitch_chunk_coords to include seam_mask and TSR
old_stitch = """    # Blend them together
    full = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len, dtype=np.float64)

    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        chunk_len = coords.shape[0]
        actual_end = min(s + chunk_len, seq_len)
        used_len = actual_end - s

        w = np.ones(used_len, dtype=np.float64)

        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)

        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w

    mask = weights > 0
    full[mask] /= weights[mask, None]

    return full"""

new_stitch = """    # Blend them together
    full = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len, dtype=np.float64)
    seam_mask = np.zeros(seq_len, dtype=bool)

    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        chunk_len = coords.shape[0]
        actual_end = min(s + chunk_len, seq_len)
        used_len = actual_end - s

        w = np.ones(used_len, dtype=np.float64)

        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
                seam_mask[ov_start:ov_end] = True

        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)
                seam_mask[s+ramp_start:s+ramp_start+ramp_len] = True

        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w

    mask = weights > 0
    full[mask] /= weights[mask, None]

    # --- Targeted Stitch-Seam Relaxation (TSR) ---
    if np.any(seam_mask):
        # Expand seam mask by 1 to allow neighbors to adjust
        sm_expanded = np.copy(seam_mask)
        for j in range(1, seq_len):
            if seam_mask[j]: sm_expanded[j-1] = True
        for j in range(seq_len - 1):
            if seam_mask[j]: sm_expanded[j+1] = True
            
        for _ in range(15):
            d = full[1:] - full[:-1]
            dist = np.linalg.norm(d, axis=1) + 1e-6
            # Relax towards 5.95A
            adj = d * ((5.95 - dist) / dist)[:, None] * 0.25
            for j in range(seq_len - 1):
                if sm_expanded[j]:   full[j] -= adj[j]
                if sm_expanded[j+1]: full[j+1] += adj[j]

    return full"""
code = code.replace(old_stitch, new_stitch)

# 5. Local Validation Splits
old_load_data = """    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))

    # Truncated copy for Protenix (Protenix has token limits)
    test_df_trunc = test_df.copy()
    test_df_trunc["sequence"] = test_df_trunc["sequence"].str[:MAX_SEQ_LEN]

    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)"""

new_load_data = """    # ── Load training data for TBM & Local Validation ───────────────────────
    print("\\nLoading training data for TBM …")
    
    train_seqs_full   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs          = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels_full = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels        = pd.read_csv(DEFAULT_VAL_LBLS)
    
    true_coords = None
    
    if not IS_KAGGLE:
        print(f"Running in LOCAL VALIDATION mode with {LOCAL_N_SAMPLES} targets.")
        test_df = train_seqs_full.tail(LOCAL_N_SAMPLES).reset_index(drop=True)
        train_seqs = train_seqs_full.head(len(train_seqs_full) - LOCAL_N_SAMPLES).reset_index(drop=True)
        true_coords = process_labels(train_labels_full)
    else:
        test_df_full = pd.read_csv(test_csv)
        test_df = test_df_full.reset_index(drop=True)
        train_seqs = train_seqs_full
        print(f"Test targets : {len(test_df)} (KAGGLE MODE)")

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))

    # Truncated copy for Protenix (Protenix has token limits)
    test_df_trunc = test_df.copy()
    test_df_trunc["sequence"] = test_df_trunc["sequence"].str[:MAX_SEQ_LEN]

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels_full, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)"""

code = code.replace(old_load_data, new_load_data)

# 6. final_predictions_dict & RMSD Calculation
code = code.replace("all_rows = []", "all_rows = []\n    final_predictions_dict = {}")
code = code.replace(
    "all_rows.extend(coords_to_rows(tid, seq, stacked))",
    "final_predictions_dict[tid] = stacked\n        all_rows.extend(coords_to_rows(tid, seq, stacked))"
)

old_save = """    # ── Save ───────────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]
    coord_cols = [c for c in cols if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub[cols].to_csv(output_csv, index=False)

    print(f"\\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")"""

new_save = """    # ── Save ───────────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]
    coord_cols = [c for c in cols if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub[cols].to_csv(output_csv, index=False)

    print(f"\\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")

    # ── Evaluate Local Validation RMSD ─────────────────────────────────────
    if not IS_KAGGLE and true_coords is not None:
        rmsd_list = []
        for tid, stacked_preds in final_predictions_dict.items():
            if tid in true_coords:
                truth = true_coords[tid]
                valid_mask = ~np.isnan(truth[:, 0])
                if valid_mask.sum() < 3: 
                    continue
                
                best_rmsd = float('inf')
                for p_idx in range(stacked_preds.shape[0]):
                    pred = stacked_preds[p_idx]
                    if len(pred) != len(truth): 
                        continue
                        
                    p_valid = pred[valid_mask]
                    t_valid = truth[valid_mask]
                    
                    P = p_valid - p_valid.mean(axis=0)
                    Q = t_valid - t_valid.mean(axis=0)
                    H = P.T @ Q
                    U, _, Vt = np.linalg.svd(H)
                    d = np.linalg.det(Vt.T @ U.T)
                    S = np.eye(3)
                    if d < 0: S[2, 2] = -1
                    R = Vt.T @ S @ U.T
                    P_aligned = P @ R.T
                    
                    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))
                    best_rmsd = min(best_rmsd, rmsd)
                
                if best_rmsd != float('inf'):
                    rmsd_list.append(best_rmsd)
        
        if rmsd_list:
            mean_rmsd = np.mean(rmsd_list)
            print(f"\\nLOCAL_VALIDATION_RMSD: {mean_rmsd:.5f}")"""

code = code.replace(old_save, new_save)

with open("notebooks/script_v66.py", "w") as f:
    f.write(code)

print("Created notebooks/script_v66.py")
