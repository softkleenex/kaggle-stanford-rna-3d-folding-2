import re
import os

with open('notebooks/top1/script.py', 'r') as f:
    content = f.read()

# 1. pip install
content = re.sub(r'!pip install (.*?\.whl)', r"os.system('pip install \1')", content)

# 2. IS_KAGGLE
content = content.replace(
    'IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))',
    'IS_KAGGLE = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))'
)

# 3. Add kabsch_rmsd and pytorch_energy_minimization
tbm_core_idx = content.find('# ─────────────── TBM Core Functions')

new_funcs = """
def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_aligned = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1))))

def pytorch_energy_minimization(coords: np.ndarray, max_iter=150, tether_weight=0.5) -> np.ndarray:
    if len(coords) < 3: return coords
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Original coordinates as tether
    X0 = torch.tensor(coords, dtype=torch.float32, device=device)
    X = torch.tensor(coords, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([X], lr=0.05)
    
    L = len(X)
    
    # Create mask for steric clash (i and j separated by more than 3 residues)
    idx = torch.arange(L, device=device)
    mask = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)) > 3
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        
        # 1. Bond length (5.95 Å)
        d1 = torch.norm(X[1:] - X[:-1], dim=1)
        E_bond = torch.sum((d1 - 5.95)**2) * 5.0
        
        # 2. Angle proxy (10.2 Å for i to i+2)
        E_angle = 0.0
        if L >= 3:
            d2 = torch.norm(X[2:] - X[:-2], dim=1)
            E_angle = torch.sum((d2 - 10.2)**2) * 1.0
            
        # 3. Tether to original prediction
        E_tether = torch.sum((X - X0)**2) * tether_weight
        
        # 4. Steric clash (repulsion) -> C1' distances shouldn't be < 3.2 A
        E_rep = 0.0
        if L > 10:
            diff = X.unsqueeze(1) - X.unsqueeze(0)
            dist = torch.norm(diff, dim=2)
            clash_dist = dist[mask]
            clashes = torch.relu(3.2 - clash_dist)
            E_rep = torch.sum(clashes**2) * 2.0
            
        loss = E_bond + E_angle + E_tether + E_rep
        loss.backward()
        optimizer.step()
        
    return X.detach().cpu().numpy()

"""

content = content[:tbm_core_idx] + new_funcs + content[tbm_core_idx:]

# 4. Apply in Phase 2
stitch_call = "full_coords = stitch_chunk_coords(coords_list, ranges_list, seq_len)"
replacement_stitch = """full_coords = stitch_chunk_coords(coords_list, ranges_list, seq_len)
                    full_coords = pytorch_energy_minimization(full_coords, max_iter=150, tether_weight=0.5)"""
content = content.replace(stitch_call, replacement_stitch)

# 5. Modify validation split and calculate RMSD
old_test_load = """    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))"""

new_test_load = """    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    train_seqs_full = pd.read_csv(DEFAULT_TRAIN_CSV)
    
    if not IS_KAGGLE:
        val_split_df = train_seqs_full.sample(n=min(5, len(train_seqs_full)), random_state=42).copy()
        train_seqs_full = train_seqs_full.drop(val_split_df.index).reset_index(drop=True)
        test_df = val_split_df.reset_index(drop=True)
        print(f"LOCAL MODE: Using {len(test_df)} targets from train.csv as test set for validation.")
    else:
        test_df = test_df_full.reset_index(drop=True)
        print(f"KAGGLE MODE: Processing all {len(test_df)} test targets.")"""

content = content.replace(old_test_load, new_test_load)

old_train_load = """    print("\\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)"""

new_train_load = """    print("\\nLoading training data for TBM …")
    train_seqs   = train_seqs_full"""

content = content.replace(old_train_load, new_train_load)

validation_block = """
    print(f"\\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")

    # ── Validation RMSD ────────────────────────────────────────────────────
    if not IS_KAGGLE:
        all_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
        true_coords_dict = process_labels(all_labels)
        
        rmsd_list = []
        for _, row in test_df.iterrows():
            tid = row["target_id"]
            if tid in true_coords_dict:
                pred_coords = sub[sub["ID"].str.startswith(f"{tid}_")].sort_values("resid")[["x_1", "y_1", "z_1"]].values
                true_coords = true_coords_dict[tid]
                if len(pred_coords) == len(true_coords):
                    rmsd = kabsch_rmsd(pred_coords, true_coords)
                    rmsd_list.append(rmsd)
        
        if rmsd_list:
            mean_rmsd = np.mean(rmsd_list)
            print(f"LOCAL_VALIDATION_RMSD: {mean_rmsd:.5f}")
"""

content = content.replace('    print(f"\\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")', validation_block)

with open('notebooks/script_v68.py', 'w') as f:
    f.write(content)
print("v68 generated successfully")
