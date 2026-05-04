import os
import sys
import subprocess
import traceback
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Install dependencies offline ──────────────────────────────────────────────
# STRATEGY: Implement a 12.5 Å distance constraint between the O2' atoms of 
# base-paired residues (i, j) to stabilize the wide minor groove width and 
# reinforce the characteristic A-form helical diameter.
# Implementation: Version 330 adds the O2'-O2' minor groove stabilization (MGS) 
# constraint within the adaptive refinement loop.

wheels = [
    "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
    "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
]
for w in wheels:
    if os.path.exists(w):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", w])
        except Exception as e:
            print(f"Failed to install {w}: {e}")
            traceback.print_exc()

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True
LOCAL_N_SAMPLES = None

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ─────────────── Paths & Constants ───────────────────────────────────────────
def get_dataset_path(pattern):
    try:
        paths = list(Path('/kaggle/input').rglob(pattern))
        if paths: return str(paths[0])
    except Exception: pass
    return None

DEFAULT_TEST_CSV   = get_dataset_path("test_sequences.csv") or "/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv"
DEFAULT_TRAIN_CSV  = get_dataset_path("train_sequences.csv") or "/kaggle/input/stanford-rna-3d-folding-2/train_sequences.csv"
DEFAULT_TRAIN_LBLS = get_dataset_path("train_labels.csv") or "/kaggle/input/stanford-rna-3d-folding-2/train_labels.csv"
DEFAULT_VAL_CSV    = get_dataset_path("validation_sequences.csv") or "/kaggle/input/stanford-rna-3d-folding-2/validation_sequences.csv"
DEFAULT_VAL_LBLS   = get_dataset_path("validation_labels.csv") or "/kaggle/input/stanford-rna-3d-folding-2/validation_labels.csv"
DEFAULT_OUTPUT     = "/kaggle/working/submission.csv"

# Protenix paths
DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "128"))
MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0

def seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

# ─────────────── Core Logic ──────────────────────────────────────────────────
def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── MGS: Minor Groove Stabilization Constraint ──────────────────────
        # Implement a 12.5 Å distance constraint between the O2' atoms of 
        # base-paired residues (i, j) to stabilize the wide minor groove width 
        # and reinforce the characteristic A-form helical diameter.
        mgs_strength = 0.25 * strength
        if stems:
            for stem in stems:
                # stems is a list of [(i, j), (i+1, j-1), ...]
                for k in range(len(stem)):
                    i, j = stem[k]
                    diff = X[j] - X[i]; dist = np.linalg.norm(diff) + 1e-6
                    # Target 12.5 Å for O2'-O2' distance across the minor groove
                    adj = diff * ((12.50 - dist) / dist) * mgs_strength
                    X[i] -= adj * 0.5
                    X[j] += adj * 0.5

        # ── MRBC: Medium-Range Backbone Curvature Constraint ────────────────
        # Reinforce the characteristic 11-residue-per-turn A-form helical twist.
        mrbc_strength = 0.28 * strength
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 4: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            
            mrbc_mask = in_stem_seg[:-3] & in_stem_seg[3:]
            if np.any(mrbc_mask):
                d_mrbc = C[3:] - C[:-3]; dist_mrbc = np.linalg.norm(d_mrbc, axis=1) + 1e-6
                adj_mrbc = d_mrbc[mrbc_mask] * ((16.50 - dist_mrbc[mrbc_mask]) / dist_mrbc[mrbc_mask])[:, None] * mrbc_strength
                C[:-3][mrbc_mask] -= adj_mrbc * 0.5
                C[3:][mrbc_mask] += adj_mrbc * 0.5
            X[s:e] = C

        # ── SPBO: Sugar-Phosphate Backbone Orientation Constraint ──────────
        spbo_strength = 0.38 * strength
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 2: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            
            spbo_mask = in_stem_seg[:-1] & in_stem_seg[1:]
            if np.any(spbo_mask):
                d_spbo = C[1:] - C[:-1]; dist_spbo = np.linalg.norm(d_spbo, axis=1) + 1e-6
                adj_spbo = d_spbo[spbo_mask] * ((6.00 - dist_spbo[spbo_mask]) / dist_spbo[spbo_mask])[:, None] * spbo_strength
                C[:-1][spbo_mask] -= adj_spbo * 0.5
                C[1:][spbo_mask] += adj_spbo * 0.5
            X[s:e] = C

        # ── Standard Refinement ──────────────────────────────────────────────
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            # P-P Bond (5.95 Å)
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            # self-avoidance & smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            X[s:e] = C

        # ── Chain-Continuity Elastic ────────────────────────────────────
        for i in range(len(X) - 1):
            diff = X[i+1] - X[i]; dist = np.linalg.norm(diff) + 1e-12
            if dist > 1.61:
                adj = diff * ((1.61 - dist) / dist) * 0.5
                X[i] -= adj; X[i+1] += adj
    return X

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    pairs = []
    for i in range(L):
        for j in range(i + 4, L):
            # Using 12.5 as the threshold to identify potential base pairs across minor groove
            if np.linalg.norm(coords[i] - coords[j]) < 12.5: pairs.append((i, j))
    stems = []
    if not pairs: return stems
    pairs.sort()
    used = [False] * len(pairs)
    for i in range(len(pairs)):
        if used[i]: continue
        stem = [pairs[i]]; used[i] = True
        ci, cj = pairs[i]
        for k in range(i + 1, len(pairs)):
            if not used[k]:
                ni, nj = pairs[k]
                if ni == ci + 1 and nj == cj - 1:
                    stem.append(pairs[k]); used[k] = True; ci, cj = ni, nj
        if len(stem) >= 3: stems.append(stem)
    return stems

# ─────────────── Integration Logic ───────────────────────────────────────────
def kabsch_align(P, Q):
    centroid_P, centroid_Q = P.mean(axis=0), Q.mean(axis=0)
    H = (P - centroid_P).T @ (Q - centroid_Q)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    return R, centroid_Q - R @ centroid_P

def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len) -> np.ndarray:
    if len(chunk_coords_list) == 1: return chunk_coords_list[0][:seq_len]
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        ps, pe = chunk_ranges[i - 1]; cs, ce = chunk_ranges[i]
        os, oe = cs, min(pe, ce)
        if (oe - os) < 3: aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i-1][os-ps:oe-ps]; cur_ov = chunk_coords_list[i][os-cs:oe-cs]
        R, t = kabsch_align(cur_ov, prev_ov)
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full = np.zeros((seq_len, 3)); weights = np.zeros(seq_len)
    for (s, e), coords in zip(chunk_ranges, aligned):
        L = min(coords.shape[0], seq_len - s)
        w = np.ones(L); full[s:s+L] += coords[:L] * w[:, None]; weights[s:s+L] += w
    mask = weights > 0; full[mask] /= weights[mask, None]
    return adaptive_rna_constraints(full, "stitched", {}, confidence=0.5)

def find_similar_sequences_detailed(query_seq, train_df, train_coords, top_n=30):
    al = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1.5, open_gap_score=-8, extend_gap_score=-0.4)
    res = []
    for _, row in train_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords or abs(len(tseq)-len(query_seq))/max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(al.align(query_seq, tseq)))
        ident = sum(1 for (qs,qe),(ts,te) in zip(*aln.aligned) for qp,tp in zip(range(qs,qe), range(ts,te)) if query_seq[qp]==tseq[tp])
        if (100 * ident / len(query_seq)) >= MIN_PERCENT_IDENTITY:
            res.append((tid, tseq, aln.score, train_coords[tid]))
    return sorted(res, key=lambda x: x[2], reverse=True)[:top_n]

def process_labels(labels_df):
    coords = {}
    for tid, grp in labels_df.groupby(labels_df["ID"].str.rsplit("_", n=1).str[0]):
        coords[tid] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    seed_everything(SEED)
    test_df = pd.read_csv(test_csv)
    train_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    train_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    segments_map = {r["target_id"]: [(0, len(r["sequence"]))] for _, r in test_df.iterrows()}

    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences_detailed(seq, train_seqs, train_coords)
        preds = []
        for tid_t, seq_t, score, coords_t in similar:
            if len(preds) >= N_SAMPLE: break
            aln = next(iter(PairwiseAligner().align(seq, seq_t)))
            adapted = np.zeros((len(seq), 3))
            for (qs,qe),(ts,te) in zip(*aln.aligned): adapted[qs:qe] = coords_t[ts:te]
            preds.append(adaptive_rna_constraints(adapted, tid, segments_map, confidence=0.8))
        
        while len(preds) < N_SAMPLE:
            coords = np.zeros((len(seq), 3))
            for i in range(len(seq)): coords[i] = [i*3, 0, 0]
            preds.append(adaptive_rna_constraints(coords, tid, segments_map, confidence=0.2))
            
        for i in range(len(seq)):
            r = {"ID": f"{tid}_{i+1}", "resname": seq[i], "resid": i+1}
            for s in range(N_SAMPLE):
                r[f"x_{s+1}"], r[f"y_{s+1}"], r[f"z_{s+1}"] = preds[s][i]
            all_rows.append(r)

    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    print(f"✓ Saved submission to {output_csv}")

if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc()
