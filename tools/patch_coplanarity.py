import sys

with open("candidates/candidate_v119.py", "r") as f:
    content = f.read()

old_block = """                # Apply weak attraction towards 10.5 A
                active_pairs = pair_mask & (dm > 8.0) & (dm < 13.0) & (sep > 3)
                if np.any(active_pairs):
                    vec = (diff * ((10.5 - dm) / dm)[:, :, None] * active_pairs[:, :, None]).sum(axis=1)
                    C += (0.02 * strength) * vec"""

new_block = """                # Apply weak attraction towards 10.5 A
                active_pairs = pair_mask & (dm > 8.0) & (dm < 13.0) & (sep > 3)
                if np.any(active_pairs):
                    vec = (diff * ((10.5 - dm) / dm)[:, :, None] * active_pairs[:, :, None]).sum(axis=1)
                    C += (0.02 * strength) * vec

                    # Base-Pair Coplanarity constraint
                    pairs_i, pairs_j = np.where(np.triu(active_pairs))
                    if len(pairs_i) > 0:
                        valid = (pairs_i > 0) & (pairs_i < L - 1) & (pairs_j > 0) & (pairs_j < L - 1)
                        pi = pairs_i[valid]
                        pj = pairs_j[valid]
                        if len(pi) > 0:
                            v_ij = C[pj] - C[pi]
                            t_i = C[pi+1] - C[pi-1]
                            t_j = C[pj+1] - C[pj-1]
                            
                            plane_dir = t_i - t_j
                            n = np.cross(v_ij, plane_dir)
                            n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
                            n = n / n_norm
                            
                            out_pi_plus = np.sum((C[pi+1] - C[pi]) * n, axis=1, keepdims=True) * n
                            out_pi_minus = np.sum((C[pi-1] - C[pi]) * n, axis=1, keepdims=True) * n
                            out_pj_plus = np.sum((C[pj+1] - C[pj]) * n, axis=1, keepdims=True) * n
                            out_pj_minus = np.sum((C[pj-1] - C[pj]) * n, axis=1, keepdims=True) * n
                            
                            coplanar_strength = 0.05 * strength
                            np.add.at(C, pi+1, -out_pi_plus * coplanar_strength)
                            np.add.at(C, pi-1, -out_pi_minus * coplanar_strength)
                            np.add.at(C, pj+1, -out_pj_plus * coplanar_strength)
                            np.add.at(C, pj-1, -out_pj_minus * coplanar_strength)
                            
                            np.add.at(C, pi, (out_pi_plus + out_pi_minus) * (coplanar_strength * 0.5))
                            np.add.at(C, pj, (out_pj_plus + out_pj_minus) * (coplanar_strength * 0.5))"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open("notebooks/script_v119.py", "w") as f:
        f.write(content)
    print("Successfully applied coplanarity constraint and saved to notebooks/script_v119.py")
else:
    print("Error: Could not find the target block to replace.")
    sys.exit(1)
