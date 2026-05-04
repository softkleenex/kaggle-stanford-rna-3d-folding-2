import re

with open('notebooks/top1/script.py', 'r') as f:
    content = f.read()

# 1. Fix pip install syntax
content = re.sub(
    r'!pip install(.*?)\.whl',
    r'os.system("pip install\1.whl")',
    content
)

# 2. Add compute_rmsd function
rmsd_func = """
def compute_rmsd(pred_coords, true_coords):
    P = pred_coords.copy()
    Q = true_coords.copy()
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P -= centroid_P
    Q -= centroid_Q
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        S[-1] = -S[-1]
        Vt[-1, :] = -Vt[-1, :]
    R = Vt.T @ U.T
    P_aligned = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1))))

def get_c1_mask
"""
content = content.replace("def get_c1_mask", rmsd_func, 1)

# 3. Replace adaptive_rna_constraints
old_constraints = """            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec"""

new_constraints = """            # self-avoidance and electrostatic repulsion
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                
                # 1. Short-range steric clash repulsion (< 3.2 A)
                mask_clash = (sep > 2) & (dm < 3.2)
                if np.any(mask_clash):
                    vec_clash = (diff * ((3.2 - dm) / dm)[:, :, None] * mask_clash[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec_clash

                # 2. Distance-dependent Dielectric Electrostatic Repulsion (Phosphate-Phosphate emulation)
                # Phosphate groups repel each other, screened by counterions. 
                # We model a soft long-range repulsion between 3.2 and 12.0 A.
                mask_elec = (sep > 4) & (dm >= 3.2) & (dm < 12.0)
                if np.any(mask_elec):
                    force_mag = 1.0 / (dm**2 + 1.0)
                    vec_elec = (diff * (force_mag / dm)[:, :, None] * mask_elec[:, :, None]).sum(axis=1)
                    C[idx] += (0.005 * strength) * vec_elec"""

if old_constraints in content:
    content = content.replace(old_constraints, new_constraints)
else:
    print("Warning: old constraints not found!")

# 4. Modify main function
old_main_start = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        test_df, combined_seqs, train_coords, segments_map
    )"""

new_main_start = """    # ── Load training data for TBM & Local Validation Split ─────────────────
    print("\\nLoading training data for TBM …")
    train_seqs_full   = pd.read_csv(DEFAULT_TRAIN_CSV)
    train_labels_full = pd.read_csv(DEFAULT_TRAIN_LBLS)
    
    # 5 targets for local validation
    val_targets = train_seqs_full['target_id'].tail(5).values
    local_val_seqs = train_seqs_full[train_seqs_full['target_id'].isin(val_targets)].reset_index(drop=True)
    train_seqs = train_seqs_full[~train_seqs_full['target_id'].isin(val_targets)].reset_index(drop=True)
    
    local_val_labels = train_labels_full[train_labels_full['ID'].str.split('_').str[0].isin(val_targets)].reset_index(drop=True)
    train_labels = train_labels_full[~train_labels_full['ID'].str.split('_').str[0].isin(val_targets)].reset_index(drop=True)
    
    local_val_coords = process_labels(local_val_labels)

    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    
    # Combine test and local_val for processing
    process_df = pd.concat([test_df, local_val_seqs], ignore_index=True)
    segments_map, _ = build_segments_map(process_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        process_df, combined_seqs, train_coords, segments_map
    )"""

if old_main_start in content:
    content = content.replace(old_main_start, new_main_start)
else:
    print("Warning: old main start not found!")


old_phase3 = """    # ─── PHASE 3: Combine everything ───────────────────────────────────────
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
        all_rows.extend(coords_to_rows(tid, seq, stacked))"""

new_phase3 = """    # ─── PHASE 3: Combine everything ───────────────────────────────────────
    print(f"\\n{'='*60}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*60}")

    all_rows = []
    val_rmsd_list = []
    test_target_ids = set(test_df["target_id"])
    val_target_ids = set(local_val_seqs["target_id"])

    for _, row in process_df.iterrows():
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

        # Stack to (N_SAMPLE, seq_len, 3)
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        
        if tid in test_target_ids:
            all_rows.extend(coords_to_rows(tid, seq, stacked))
            
        if tid in val_target_ids:
            true = local_val_coords.get(tid)
            if true is not None and len(combined) > 0:
                pred = combined[0] # top-1
                if len(pred) == len(true):
                    rmsd = compute_rmsd(pred, true)
                    val_rmsd_list.append(rmsd)

    if val_rmsd_list:
        mean_rmsd = np.mean(val_rmsd_list)
        print(f"LOCAL_VALIDATION_RMSD: {mean_rmsd:.5f}")
    else:
        print("LOCAL_VALIDATION_RMSD: 99.99999")"""

if old_phase3 in content:
    content = content.replace(old_phase3, new_phase3)
else:
    print("Warning: old phase3 not found!")


with open('notebooks/script_v118.py', 'w') as f:
    f.write(content)

