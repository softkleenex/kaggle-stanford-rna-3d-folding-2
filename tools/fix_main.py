import re

with open('notebooks/script_v70_temp.py', 'r') as f:
    content = f.read()

# 1. Fix data loading in main() to support local validation
old_loading = """    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))

    # Truncated copy for Protenix (Protenix has token limits)
    test_df_trunc = test_df.copy()
    test_df_trunc["sequence"] = test_df_trunc["sequence"].str[:MAX_SEQ_LEN]"""

new_loading = """    # ── Load test data / Validation Split ────────────────────────────────────
    train_seqs_full = pd.read_csv(DEFAULT_TRAIN_CSV)
    if not IS_KAGGLE:
        print("LOCAL MODE: creating a validation split from train_sequences.csv")
        test_df = train_seqs_full.head(15).copy() # Use 15 targets for validation
        test_df = test_df.reset_index(drop=True)
        # We also need train_seqs without these 15
        train_seqs = train_seqs_full.iloc[15:].copy().reset_index(drop=True)
    else:
        test_df_full = pd.read_csv(test_csv)
        test_df = test_df_full.reset_index(drop=True)
        train_seqs = train_seqs_full.copy()
        
    print(f"Test targets : {len(test_df)}" + (" (LOCAL MODE)" if not IS_KAGGLE else ""))"""
content = content.replace(old_loading, new_loading)

# 2. Fix TBM training data loading since we redefined train_seqs above
old_tbm_load = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)"""

new_tbm_load = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    # train_seqs already defined above
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)"""
content = content.replace(old_tbm_load, new_tbm_load)


# 3. Add relaxation to stitched Protenix outputs
old_stitch = """                stitched_samples = []
                for s_idx in range(n_needed):
                    items = chunk_results_per_sample[s_idx]
                    coords_list = [c for c, _ in items]
                    ranges_list = [r for _, r in items]
                    full_coords = stitch_chunk_coords(coords_list, ranges_list, seq_len)
                    stitched_samples.append(full_coords)
                
                result = np.stack(stitched_samples, axis=0)
                protenix_preds[target_id] = result"""

new_stitch = """                stitched_samples = []
                for s_idx in range(n_needed):
                    items = chunk_results_per_sample[s_idx]
                    coords_list = [c for c, _ in items]
                    ranges_list = [r for _, r in items]
                    full_coords = stitch_chunk_coords(coords_list, ranges_list, seq_len)
                    # Apply Tethered Geometry Relaxation to fix stitching shrinkage
                    full_coords = tethered_geometry_relaxation(full_coords, passes=50, k_tether=0.15, k_bond=1.0)
                    stitched_samples.append(full_coords)
                
                result = np.stack(stitched_samples, axis=0)
                protenix_preds[target_id] = result"""
content = content.replace(old_stitch, new_stitch)

# 4. Also add relaxation to single-pass Protenix outputs just in case
old_single_pass = """            if len(chunks) == 1:
                # Single pass
                coords = raw_predictions.get(target_id)
                protenix_preds[target_id] = coords"""

new_single_pass = """            if len(chunks) == 1:
                # Single pass
                coords = raw_predictions.get(target_id)
                if coords is not None:
                    # Apply Tethered Geometry Relaxation to single-pass to ensure 5.95A bond lengths universally
                    rel_coords = []
                    for s_idx in range(coords.shape[0]):
                        rel_coords.append(tethered_geometry_relaxation(coords[s_idx], passes=30, k_tether=0.2, k_bond=1.0))
                    coords = np.stack(rel_coords, axis=0)
                protenix_preds[target_id] = coords"""
content = content.replace(old_single_pass, new_single_pass)


# 5. Local RMSD evaluation at the end
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
    
    # ── Local Validation ───────────────────────────────────────────────────
    if not IS_KAGGLE:
        print("\\nCalculating local validation RMSD...")
        rmsds = []
        for tid in test_df["target_id"].unique():
            # Get predicted coords (mean of N_SAMPLE or just the 1st sample)
            pred_rows = sub[sub["ID"].str.startswith(tid + "_")].sort_values("resid")
            if len(pred_rows) == 0: continue
            
            # Using 1st model prediction for validation RMSD
            pred_coords = pred_rows[["x_1", "y_1", "z_1"]].values
            
            # Get true coords
            if tid in train_coords:
                true_coords = train_coords[tid]
                if len(pred_coords) == len(true_coords):
                    score = compute_rmsd(pred_coords, true_coords)
                    rmsds.append(score)
        
        if rmsds:
            final_rmsd = np.mean(rmsds)
            print(f"LOCAL_VALIDATION_RMSD: {final_rmsd:.5f}")
        else:
            print("LOCAL_VALIDATION_RMSD: NaN")"""
content = content.replace(old_save, new_save)

with open('notebooks/script_v70.py', 'w') as f:
    f.write(content)
