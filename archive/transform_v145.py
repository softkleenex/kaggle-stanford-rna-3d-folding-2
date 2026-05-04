import re

with open("notebooks/top1/script.py", "r") as f:
    content = f.read()

# Fix !pip install
content = re.sub(r"!pip install (.*\.whl)", r'import os; os.system("pip install \1")', content)

# Add Rg constraint
constraint_code = """            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # C1' Radius of Gyration (Rg) compaction constraint
            # Encourages globular folding and prevents unphysically extended linear conformations
            if L >= 10:
                center = C.mean(axis=0)
                diff = C - center
                r = np.linalg.norm(diff, axis=1) + 1e-6
                ideal_rg = 3.0 * (L ** 0.33)
                current_rg = np.sqrt(np.mean(r**2))
                if current_rg > ideal_rg:
                    C -= (0.02 * strength) * (diff / r[:, None]) * ((current_rg - ideal_rg) / current_rg)
"""
content = content.replace("            # Laplacian smoothing\n            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])", constraint_code)

# Add evaluate_local_validation function
eval_code = """
def evaluate_local_validation(train_seqs, train_coords_dict, val_seqs, val_coords_dict, segments_map):
    print("\\nEvaluating Local Validation RMSD...")
    val_subset = val_seqs.head(20).copy()
    template_preds, _ = tbm_phase(val_subset, train_seqs, train_coords_dict, segments_map)
    rmsds = []
    for _, row in val_subset.iterrows():
        tid = row["target_id"]
        if tid in val_coords_dict and tid in template_preds and len(template_preds[tid]) > 0:
            pred = template_preds[tid][0]
            true = val_coords_dict[tid]
            mlen = min(len(pred), len(true))
            if mlen > 0:
                R, t = kabsch_align(pred[:mlen], true[:mlen])
                aligned_pred = (pred[:mlen] @ R.T) + t
                rmsd = np.sqrt(np.mean(np.sum((aligned_pred - true[:mlen])**2, axis=1)))
                rmsds.append(rmsd)
    
    final_rmsd = np.mean(rmsds) if rmsds else 0.0
    print(f"LOCAL_VALIDATION_RMSD: {final_rmsd:.5f}")

"""
content = content.replace("def main() -> None:", eval_code + "\ndef main() -> None:")

# Inject call to evaluate_local_validation
main_inject = """    # ─── LOCAL VALIDATION ──────────────────────────────────────────────────
    evaluate_local_validation(combined_seqs, train_coords, val_seqs, train_coords, segments_map)

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────"""
content = content.replace("    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────", main_inject)

with open("notebooks/script_v145.py", "w") as f:
    f.write(content)
print("Transformation completed.")
