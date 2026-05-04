import os
import re

with open("notebooks/top1/script.py", "r") as f:
    code = f.read()

# 0. Fix Jupyter Magic syntax errors (!pip -> os.system)
lines = code.splitlines()
for i, line in enumerate(lines):
    if line.strip().startswith("!pip install"):
        lines[i] = "import os; os.system('" + line.strip()[1:] + "')"
code = "\n".join(lines)

# 1. Modify adaptive_rna_constraints to add soft i-i+4
old_constraint = """            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2"""

new_constraint = """            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            
            # soft i–i+4  ~16.8 Å
            if L >= 5:
                d4   = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((16.8 - d4n) / d4n)[:, None] * (0.05 * strength)
                C[:-4] -= adj4; C[4:] += adj4"""

if old_constraint not in code:
    print("Failed to find old constraint")
code = code.replace(old_constraint, new_constraint)

# 2. Modify data loading to implement try/except for validation and split if necessary
old_data_loading = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)"""

new_data_loading = """    # ── Load training data for TBM ──────────────────────────────────────────
    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    
    # ── Local Validation Split ──────────────────────────────────────────────
    try:
        val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
        val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)
    except FileNotFoundError:
        print("Validation files not found, splitting train data for local validation.")
        unique_targets = train_seqs["target_id"].unique()
        val_targets = unique_targets[-5:]
        train_targets = unique_targets[:-5]
        
        val_seqs = train_seqs[train_seqs["target_id"].isin(val_targets)].copy()
        val_labels = train_labels[train_labels["ID"].str.rsplit("_", n=1).str[0].isin(val_targets)].copy()
        
        train_seqs = train_seqs[train_seqs["target_id"].isin(train_targets)].copy()
        train_labels = train_labels[train_labels["ID"].str.rsplit("_", n=1).str[0].isin(train_targets)].copy()

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)"""

if old_data_loading not in code:
    print("Failed to find data loading block")
code = code.replace(old_data_loading, new_data_loading)

# 3. Add local validation computation at the end of main()
old_end_main = """    print(f"\\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")"""

new_end_main = """    print(f"\\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")

    # ── Local Validation Execution ─────────────────────────────────────────
    print("\\nEvaluating LOCAL_VALIDATION_RMSD...")
    val_segments_map, _ = build_segments_map(val_seqs)
    val_preds, _ = tbm_phase(val_seqs, train_seqs, process_labels(train_labels), val_segments_map)
    val_coords_true = process_labels(val_labels)
    
    rmsd_list = []
    for _, row in val_seqs.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]
        preds = val_preds.get(tid, [])
        while len(preds) < 1:
            preds.append(adaptive_rna_constraints(generate_rna_structure(seq), tid, val_segments_map, confidence=0.2))
        
        pred_coords = preds[0]
        true_coords = val_coords_true.get(tid)
        
        if true_coords is not None and len(pred_coords) == len(true_coords):
            P = pred_coords
            Q = true_coords
            centroid_P = P.mean(axis=0)
            centroid_Q = Q.mean(axis=0)
            Pc = P - centroid_P
            Qc = Q - centroid_Q
            H = Pc.T @ Qc
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            S = np.eye(3)
            if d < 0: S[2, 2] = -1
            R = Vt.T @ S @ U.T
            P_aligned = Pc @ R.T + centroid_Q
            rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))
            rmsd_list.append(rmsd)
            
    if rmsd_list:
        print(f"LOCAL_VALIDATION_RMSD: {np.mean(rmsd_list):.5f}")
    else:
        print("LOCAL_VALIDATION_RMSD: 0.00000")"""

if old_end_main not in code:
    print("Failed to find end of main block")
code = code.replace(old_end_main, new_end_main)

with open("notebooks/script_v148.py", "w") as f:
    f.write(code)

print("Modification complete.")
