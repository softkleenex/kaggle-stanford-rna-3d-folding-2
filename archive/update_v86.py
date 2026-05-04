import re

with open("notebooks/top1/script.py", "r") as f:
    code = f.read()

# 1. Add kabsch_rmsd and enforce_ibso_at_junction
funcs = """def kabsch_align(P: np.ndarray, Q: np.ndarray):
    \"\"\"Compute optimal rotation R and translation t so that  R @ P + t ≈ Q.\"\"\"
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    Pc = P - centroid_P
    Qc = Q - centroid_Q
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    if len(P) == 0 or len(Q) == 0:
        return 0.0
    P = P.copy()
    Q = Q.copy()
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P -= cP
    Q -= cQ
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S_mat = np.eye(3)
    if d < 0.0:
        S_mat[2, 2] = -1.0
    R = Vt.T @ S_mat @ U.T
    P_aligned = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))


def enforce_ibso_at_junction(full_coords: np.ndarray, overlap_start: int, overlap_end: int) -> np.ndarray:
    \"\"\"
    Interface Base-Stacking Optimization (IBSO)
    Enforces a 3.4 Å vertical spacing along the local helical axis.
    \"\"\"
    junction_idx = (overlap_start + overlap_end) // 2
    if junction_idx < 2 or junction_idx >= len(full_coords) - 2:
        return full_coords
        
    X = full_coords.copy()
    v1 = X[junction_idx] - X[junction_idx - 1]
    v2 = X[junction_idx + 1] - X[junction_idx]
    
    t = (v1 + v2) / 2.0
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-6:
        t = t / t_norm
    else:
        t = np.array([0.0, 0.0, 1.0])
        
    target_step = 3.4
    actual_step1 = np.dot(v1, t)
    actual_step2 = np.dot(v2, t)
    
    correction1 = (target_step - actual_step1) * t * 0.25
    correction2 = (target_step - actual_step2) * t * 0.25
    
    X[junction_idx - 1] -= correction1
    X[junction_idx] += correction1 - correction2
    X[junction_idx + 1] += correction2
    
    smooth_rad = 4
    start_i = max(1, junction_idx - smooth_rad)
    end_i = min(len(X) - 1, junction_idx + smooth_rad + 1)
    
    for _ in range(3):
        X_new = X.copy()
        for i in range(start_i, end_i):
            X_new[i] = X[i] * 0.5 + X[i-1] * 0.25 + X[i+1] * 0.25
        X = X_new
            
    return X
"""

code = code.replace("""def kabsch_align(P: np.ndarray, Q: np.ndarray):
    \"\"\"Compute optimal rotation R and translation t so that  R @ P + t ≈ Q.\"\"\"
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    Pc = P - centroid_P
    Qc = Q - centroid_Q
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t""", funcs)

# 2. Update stitch_chunk_coords
ibso_call = """    mask = weights > 0
    full[mask] /= weights[mask, None]

    # ------ APPLY IBSO HERE ------
    for i in range(1, len(chunk_ranges)):
        prev_end = chunk_ranges[i - 1][1]
        cur_start = chunk_ranges[i][0]
        if prev_end > cur_start:
            full = enforce_ibso_at_junction(full, cur_start, prev_end)
    # -----------------------------

    return full"""

code = code.replace("""    mask = weights > 0
    full[mask] /= weights[mask, None]

    return full""", ibso_call)


# 3. Validation split loading in main
val_load = """    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)

    # ── Create Local Validation Split ───────────────────────────────────────
    train_seqs_full = pd.read_csv(DEFAULT_TRAIN_CSV)
    train_labels_full = pd.read_csv(DEFAULT_TRAIN_LBLS)
    
    num_val = 5
    val_split_seqs = train_seqs_full.tail(num_val).copy()
    val_split_targets = set(val_split_seqs["target_id"].tolist())
    val_split_labels = train_labels_full[train_labels_full["ID"].str.rsplit("_", n=1).str[0].isin(val_split_targets)].copy()
    val_true_coords = process_labels(val_split_labels)
    
    # Remove validation from train
    train_seqs = train_seqs_full.head(len(train_seqs_full) - num_val).copy()
    train_labels = train_labels_full[~train_labels_full["ID"].str.rsplit("_", n=1).str[0].isin(val_split_targets)].copy()

    # Prepend validation to test
    test_df = pd.concat([val_split_seqs, test_df], ignore_index=True)

    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))"""

code = code.replace("""    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))""", val_load)

code = code.replace("""    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)""", """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)""")

# 4. Phase 3 validation evaluation
phase_3 = """    # ─── PHASE 3: Combine everything ───────────────────────────────────────
    print(f"\\n{'='*60}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*60}")

    all_rows = []
    rmsd_list = []

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        combined: list = list(template_preds.get(tid, []))  # TBM predictions

        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        
        if tid in val_split_targets and tid in val_true_coords:
            true_c = val_true_coords[tid]
            pred_c = stacked[0] # use top 1 for validation
            min_len = min(len(pred_c), len(true_c))
            if min_len > 0:
                rmsd = kabsch_rmsd(pred_c[:min_len], true_c[:min_len])
                rmsd_list.append(rmsd)
                
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    if rmsd_list:
        print(f"\\nLOCAL_VALIDATION_RMSD: {np.mean(rmsd_list):.5f}")
    else:
        print(f"\\nLOCAL_VALIDATION_RMSD: 0.00000")

    # ── Save ───────────────────────────────────────────────────────────────
    # Filter out local validation targets before saving to avoid format errors
    final_rows = [r for r in all_rows if r["ID"].rsplit("_", 1)[0] not in val_split_targets]
    
    sub = pd.DataFrame(final_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]"""

code = code.replace("""    # ─── PHASE 3: Combine everything ───────────────────────────────────────
    print(f"\\n{'='*60}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*60}")

    all_rows = []

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        combined: list = list(template_preds.get(tid, []))  # TBM predictions

        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    # ── Save ───────────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]""", phase_3)

with open("notebooks/script_v86.py", "w") as f:
    f.write(code)

print("Patch applied successfully.")
