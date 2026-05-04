import os
import sys
import subprocess
import pandas as pd
import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm
import gc
import json
import time
from pathlib import Path

# ── Install dependencies (Offline) ───────────────────────────────────────────
def install_wheels():
    wheels = [
        "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
    ]
    for wheel in wheels:
        if os.path.exists(wheel):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
            except Exception as e:
                print(f"Error installing {wheel}: {e}")

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") != "Interactive":
    # On Kaggle, we might need to install them if they're not in the environment
    # But usually, in a competition submission, we do this at the start.
    pass

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
LOCAL_N_SAMPLES = 2

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                return torch.from_numpy(m).bool()
        except: pass
    return torch.ones(len(data.get("sequence", "")), dtype=torch.bool)

def detect_simple_stems(X, threshold=11.5):
    L = len(X); stems = []; used = np.zeros(L, dtype=bool)
    for i in range(L):
        if used[i]: continue
        for j in range(L - 1, i + 3, -1):
            if used[j]: continue
            if np.linalg.norm(X[i] - X[j]) < threshold:
                curr_i, curr_j = i, j; stem = []
                while curr_i < curr_j and curr_i < L and curr_j >= 0:
                    if np.linalg.norm(X[curr_i] - X[curr_j]) < threshold:
                        stem.append((curr_i, curr_j))
                        curr_i += 1; curr_j -= 1
                    else: break
                if len(stem) >= 3:
                    stems.append(stem)
                    for si, sj in stem: used[si] = used[sj] = True
                    break
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        if stems:
            bpp_strength = 0.15 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if i_s > 0 and i_s < len(X)-1 and j_s > 0 and j_s < len(X)-1:
                        v1_i = X[i_s+1] - X[i_s]; v2_i = X[i_s-1] - X[i_s]
                        n_i = np.cross(v1_i, v2_i); dn_i = np.linalg.norm(n_i) + 1e-6
                        n_i /= dn_i
                        v1_j = X[j_s+1] - X[j_s]; v2_j = X[j_s-1] - X[j_s]
                        n_j = np.cross(v1_j, v2_j); dn_j = np.linalg.norm(n_j) + 1e-6
                        n_j /= dn_j
                        dot = np.dot(n_i, n_j)
                        if abs(dot) < 0.98:
                            target_n = n_i if dot > 0 else -n_i
                            adj_bpp = (target_n - n_j) * bpp_strength
                            X[j_s+1] += adj_bpp; X[j_s-1] += adj_bpp
                            X[i_s+1] -= adj_bpp; X[i_s-1] -= adj_bpp

        if len(stems) >= 2:
            chs_strength = 0.12 * strength
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    stem_a = stems[i]; stem_b = stems[j]
                    terms_a = [stem_a[0], stem_a[-1]]
                    terms_b = [stem_b[0], stem_b[-1]]
                    for (ai, aj) in terms_a:
                        for (bi, bj) in terms_b:
                            dist_ij = min(abs(ai - bi), abs(ai - bj), abs(aj - bi), abs(aj - bj))
                            if dist_ij <= 5: 
                                vec_a = X[ai] - X[aj]; d_a = np.linalg.norm(vec_a) + 1e-6
                                vec_b = X[bi] - X[bj]; d_b = np.linalg.norm(vec_b) + 1e-6
                                center_a = (X[ai] + X[aj]) * 0.5
                                center_b = (X[bi] + X[bj]) * 0.5
                                d_centers = np.linalg.norm(center_a - center_b) + 1e-6
                                if d_centers < 15.0:
                                    adj_c = (center_a - center_b) * ((10.0 - d_centers) / d_centers) * chs_strength
                                    X[ai] += adj_c; X[aj] += adj_c
                                    X[bi] -= adj_c; X[bj] -= adj_c
                                cos_sim = np.dot(vec_a, vec_b) / (d_a * d_b)
                                if cos_sim < 0.95:
                                    target_vec = (vec_a / d_a + vec_b / d_b) * 0.5
                                    target_vec /= np.linalg.norm(target_vec) + 1e-6
                                    X[ai] += (target_vec * d_a * 0.5 + center_a - X[ai]) * chs_strength
                                    X[aj] += (-target_vec * d_a * 0.5 + center_a - X[aj]) * chs_strength
                                    X[bi] += (target_vec * d_b * 0.5 + center_b - X[bi]) * chs_strength
                                    X[bj] += (-target_vec * d_b * 0.5 + center_b - X[bj]) * chs_strength

        if len(stems) >= 2:
            csc_strength = 0.20 * strength
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    ends_a = [stems[i][0], stems[i][-1]]
                    ends_b = [stems[j][0], stems[j][-1]]
                    for (ai, aj) in ends_a:
                        for (bi, bj) in ends_b:
                            if abs(ai - bi) <= 3 or abs(ai - bj) <= 3 or abs(aj - bi) <= 3 or abs(aj - bj) <= 3:
                                axis_a = X[ai] - X[aj]; axis_b = X[bi] - X[bj]
                                d_a = np.linalg.norm(axis_a) + 1e-6
                                d_b = np.linalg.norm(axis_b) + 1e-6
                                u_a = axis_a / d_a; u_b = axis_b / d_b
                                cos_val = np.dot(u_a, u_b)
                                if abs(cos_val) < 0.95:
                                    target_u = u_a if cos_val > 0 else -u_a
                                    adj_csc = (target_u - u_b) * csc_strength
                                    X[bi] += adj_csc * d_b * 0.5; X[bj] -= adj_csc * d_b * 0.5

        if stems:
            cspp_strength = 0.25 * strength
            for stem in stems:
                for pair in stem:
                    i, j = pair
                    tj_minor = j - 3
                    if 0 <= tj_minor < len(X) and i < len(X):
                        dv_m = X[i] - X[tj_minor]
                        dl_m = np.linalg.norm(dv_m) + 1e-6
                        if 11.0 < dl_m < 21.0:
                            adj_m = dv_m * ((15.5 - dl_m) / dl_m) * cspp_strength
                            X[i] += adj_m * 0.5
                            X[tj_minor] -= adj_m * 0.5
                    tj_major = j + 2
                    if 0 <= tj_major < len(X) and i < len(X):
                        dv_M = X[i] - X[tj_major]
                        dl_M = np.linalg.norm(dv_M) + 1e-6
                        if 7.5 < dl_M < 14.5:
                            adj_M = dv_M * ((10.2 - dl_M) / dl_M) * cspp_strength
                            X[i] += adj_M * 0.5
                            X[tj_major] -= adj_M * 0.5

        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            in_stem = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem[i_s - s] = True
                        if s <= j_s < e: in_stem[j_s - s] = True
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            if L >= 2:
                ohpi_mask = in_stem[:-1] & in_stem[1:]
                if np.any(ohpi_mask):
                    ohpi_strength = 0.32 * strength
                    adj_ohpi = d[ohpi_mask] * ((5.90 - dist[ohpi_mask]) / dist[ohpi_mask])[:, None] * ohpi_strength
                    C[:-1][ohpi_mask] -= adj_ohpi * 0.5; C[1:][ohpi_mask] += adj_ohpi * 0.5
                stretch_mask = (~ohpi_mask) & (dist > 6.2)
                if np.any(stretch_mask):
                    adj_rr = d[stretch_mask] * ((6.1 - dist[stretch_mask]) / dist[stretch_mask])[:, None] * (0.3 * strength)
                    C[:-1][stretch_mask] -= adj_rr * 0.5; C[1:][stretch_mask] += adj_rr * 0.5

            if L >= 2:
                d_vbs = C[1:] - C[:-1]; dist_vbs = np.linalg.norm(d_vbs, axis=1) + 1e-6
                vbs_mask = (~in_stem[:-1]) | (~in_stem[1:])
                vbs_strength = 0.18 * strength
                if np.any(vbs_mask):
                    adj_vbs = d_vbs[vbs_mask] * ((3.4 - dist_vbs[vbs_mask]) / dist_vbs[vbs_mask])[:, None] * vbs_strength
                    C[:-1][vbs_mask] -= adj_vbs * 0.5; C[1:][vbs_mask] += adj_vbs * 0.5

            if L >= 4:
                bpt_strength = 0.14 * strength
                p1, p2, p3, p4 = C[:-3], C[1:-2], C[2:-1], C[3:]
                v12 = p2 - p1; v23 = p3 - p2; v34 = p4 - p3
                n1 = np.cross(v12, v23); n2 = np.cross(v23, v34)
                d1 = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-9
                d2 = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9
                n1 /= d1; n2 /= d2
                cos_phi = np.sum(n1 * n2, axis=1); cos_phi = np.clip(cos_phi, -1.0, 1.0)
                phi = np.arccos(cos_phi)
                sign = np.sum(v12 * n2, axis=1)
                phi = np.where(sign < 0, -phi, phi)
                target_phi = 0.61 
                diff_phi = target_phi - phi
                axis = v23 / (np.linalg.norm(v23, axis=1, keepdims=True) + 1e-9)
                force = np.cross(n2, axis) * diff_phi[:, None] * bpt_strength
                C[3:] += force * 0.5; C[:-3] -= force * 0.5

            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((13.5 - d3n) / d3n)[:, None] * (0.05 * strength)
                C[:-3] -= adj3; C[3:] += adj3
            if L >= 5:
                d4 = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((16.5 - d4n) / d4n)[:, None] * (0.03 * strength)
                C[:-4] -= adj4; C[4:] += adj4
            if L >= 4:
                idx_posr = np.arange(L); P_posr = C[idx_posr]
                diff_posr = P_posr[:, None, :] - P_posr[None, :, :]
                dm_posr = np.linalg.norm(diff_posr, axis=2) + 1e-6
                sep_posr = np.abs(idx_posr[:, None] - idx_posr[None, :])
                mask_posr = (sep_posr > 2) & (dm_posr < 5.6)
                if np.any(mask_posr):
                    vec_posr = (diff_posr * ((5.6 - dm_posr) / dm_posr)[:, :, None] * mask_posr[:, :, None]).sum(axis=1)
                    C[idx_posr] += (0.08 * strength) * vec_posr
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            if L >= 25:
                idx = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm = np.linalg.norm(diff, axis=2) + 1e-6
                sep = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
    return X

def main():
    test_csv = "/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv"
    if not os.path.exists(test_csv):
        print("Test CSV not found, exiting.")
        return

    test_df = pd.read_csv(test_csv)
    if not IS_KAGGLE and LOCAL_N_SAMPLES:
        test_df = test_df.head(LOCAL_N_SAMPLES)

    submission = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        tid, seq = row["target_id"], row["sequence"]
        L = len(seq)
        coords = np.cumsum(np.random.normal(0, 3, (L, 3)), axis=0)
        refined = adaptive_rna_constraints(coords, tid, {}, confidence=0.5)
        for i in range(L):
            submission.append({"id": f"{tid}_{i}", "x": refined[i, 0], "y": refined[i, 1], "z": refined[i, 2]})

    pd.DataFrame(submission).to_csv("submission.csv", index=False)
    print("Finished. submission.csv generated.")

if __name__ == "__main__":
    main()
