import os
import sys

# Convert !pip commands to os.system calls for compatibility
os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""))
LOCAL_N_SAMPLES = 5 if not IS_KAGGLE else None

if IS_KAGGLE:
    print("Running in KAGGLE mode.")
else:
    print(f"Running in LOCAL mode — capping test to {LOCAL_N_SAMPLES} samples.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
if not os.path.exists(DATA_BASE):
    DATA_BASE = "data/raw" # Local fallback

DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "submission.csv"

DEFAULT_CODE_DIR = (
    "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
    "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
)
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = 512
CHUNK_OVERLAP = 128

MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0
USE_PROTENIX = os.path.exists(DEFAULT_CODE_DIR)

# ─────────────── General Utilities ───────────────────────────────────────────
def seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def kabsch_align(P: np.ndarray, Q: np.ndarray):
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

def calculate_rmsd(P, Q):
    L = min(len(P), len(Q))
    if L < 3: return 0.0
    P_sub, Q_sub = P[:L], Q[:L]
    R, t = kabsch_align(P_sub, Q_sub)
    P_aligned = (P_sub @ R.T) + t
    return np.sqrt(np.mean(np.sum((P_aligned - Q_sub)**2, axis=-1)))

# ─────────────── Structural Constraints ──────────────────────────────────────
def find_potential_wc_pairs(X, seq):
    n = len(X)
    pairs = []
    comp = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    for i in range(n):
        for j in range(i + 4, n):
            # Watson-Crick or Wobble
            if seq[j] == comp.get(seq[i]) or (seq[i] == 'G' and seq[j] == 'U') or (seq[i] == 'U' and seq[j] == 'G'):
                dist = np.linalg.norm(X[i] - X[j])
                if 8.0 < dist < 13.5:
                    pairs.append((i, j))
    return pairs

def adaptive_rna_constraints(coords, target_id, segments_map, seq="", confidence=1.0, passes=3) -> np.ndarray:
    X = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.8 * (1.0 - min(confidence, 0.98)), 0.05)
    
    pairs = find_potential_wc_pairs(X, seq) if seq else []
    
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # 1. i, i+1 Stacking (~5.95 Å)
            d1 = C[1:] - C[:-1]; dist1 = np.linalg.norm(d1, axis=1) + 1e-6
            adj1 = d1 * ((5.95 - dist1) / dist1)[:, None] * (0.25 * strength)
            C[:-1] -= adj1; C[1:] += adj1
            
            # 2. i, i+2 Backbone (~10.2 Å)
            if L > 2:
                d2 = C[2:] - C[:-2]; dist2 = np.linalg.norm(d2, axis=1) + 1e-6
                adj2 = d2 * ((10.2 - dist2) / dist2)[:, None] * (0.12 * strength)
                C[:-2] -= adj2; C[2:] += adj2
                
            # 3. i, i+3 A-form Trace (~13.0 Å) - NEW
            if L > 3:
                d3 = C[3:] - C[:-3]; dist3 = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((13.0 - dist3) / dist3)[:, None] * (0.08 * strength)
                C[:-3] -= adj3; C[3:] += adj3

            # 4. i, i+5 A-form Diagonal (~16.5 Å) - NEW
            if L > 5:
                d5 = C[5:] - C[:-5]; dist5 = np.linalg.norm(d5, axis=1) + 1e-6
                adj5 = d5 * ((16.5 - dist5) / dist5)[:, None] * (0.05 * strength)
                C[:-5] -= adj5; C[5:] += adj5
            
            # 5. Laplacian smoothing
            C[1:-1] += (0.05 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            X[s:e] = C

        # 6. Watson-Crick Base Pair Distance (~10.5 Å) - NEW
        for i, j in pairs:
            d = X[i] - X[j]; dist = np.linalg.norm(d) + 1e-6
            adj = d * ((10.5 - dist) / dist) * (0.15 * strength)
            X[i] -= adj; X[j] += adj
            
    return X

# ─────────────── TBM Phase ───────────────────────────────────────────────────
_aligner = PairwiseAligner()
_aligner.mode = "global"
_aligner.match_score = 2
_aligner.mismatch_score = -1.5
_aligner.open_gap_score = -8
_aligner.extend_gap_score = -0.4

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar(query_seq, train_seqs_df, train_coords_dict, top_n=5):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict: continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        norm_s = aln.score / (2 * min(len(query_seq), len(tseq)))
        results.append((tid, tseq, norm_s, train_coords_dict[tid]))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template(query_seq, template_seq, template_coords):
    aln = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs): new_coords[qs:qe] = chunk
    # Interpolate gaps
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i+1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0: new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0: new_coords[i] = new_coords[nv] + [3, 0, 0]
            else: new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)

def generate_fallback_structure(sequence: str, seed=42) -> np.ndarray:
    np.random.seed(seed)
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.8]
    return coords

# ─────────────── Main ────────────────────────────────────────────────────────
def main():
    seed_everything(SEED)
    
    # Load test
    test_df = pd.read_csv(DEFAULT_TEST_CSV)
    if LOCAL_N_SAMPLES: test_df = test_df.head(LOCAL_N_SAMPLES)
    
    # Load templates
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV)
    train_lbls = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_seqs = pd.read_csv(DEFAULT_VAL_CSV) if os.path.exists(DEFAULT_VAL_CSV) else pd.DataFrame()
    val_lbls = pd.read_csv(DEFAULT_VAL_LBLS) if os.path.exists(DEFAULT_VAL_LBLS) else pd.DataFrame()
    
    combined_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True)
    combined_lbls = pd.concat([train_lbls, val_lbls], ignore_index=True)
    train_coords_dict = process_labels(combined_lbls)
    
    segments_map = {tid: [(0, len(seq))] for tid, seq in zip(test_df["target_id"], test_df["sequence"])}
    
    # Run TBM
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="TBM Phase"):
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar(seq, combined_seqs, train_coords_dict, top_n=N_SAMPLE)
        
        preds = []
        for i in range(N_SAMPLE):
            if i < len(similar):
                tmpl_id, tmpl_seq, sim, tmpl_coords = similar[i]
                X = adapt_template(seq, tmpl_seq, tmpl_coords)
                conf = sim
            else:
                X = generate_fallback_structure(seq, seed=SEED+i)
                conf = 0.2
            
            refined = adaptive_rna_constraints(X, tid, segments_map, seq=seq, confidence=conf)
            preds.append(refined)
        
        stacked = np.stack(preds, axis=0)
        for i in range(len(seq)):
            res_row = {"ID": f"{tid}_{i+1}", "resname": seq[i], "resid": i+1}
            for s in range(N_SAMPLE):
                res_row[f"x_{s+1}"] = stacked[s, i, 0]
                res_row[f"y_{s+1}"] = stacked[s, i, 1]
                res_row[f"z_{s+1}"] = stacked[s, i, 2]
            results.append(res_row)
            
    sub = pd.DataFrame(results)
    sub.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"Saved submission to {DEFAULT_OUTPUT}")
    
    # ─────────────── LOCAL VALIDATION ─────────────────────────────────────────
    if os.path.exists(DEFAULT_VAL_CSV) and os.path.exists(DEFAULT_VAL_LBLS):
        print("\nRunning Local Validation...")
        v_seqs = pd.read_csv(DEFAULT_VAL_CSV).head(10)
        v_lbls = pd.read_csv(DEFAULT_VAL_LBLS)
        v_coords = process_labels(v_lbls)
        v_seg_map = {tid: [(0, len(seq))] for tid, seq in zip(v_seqs["target_id"], v_seqs["sequence"])}
        
        rmsd_list = []
        for _, row in v_seqs.iterrows():
            tid, seq = row["target_id"], row["sequence"]
            if tid not in v_coords: continue
            target = v_coords[tid]
            
            # Use TBM pool excluding self if present
            pool = combined_seqs[combined_seqs["target_id"] != tid]
            similar = find_similar(seq, pool, train_coords_dict, top_n=1)
            
            if similar:
                _, tseq, sim, tcoords = similar[0]
                X = adapt_template(seq, tseq, tcoords)
                refined = adaptive_rna_constraints(X, tid, v_seg_map, seq=seq, confidence=sim)
            else:
                X = generate_fallback_structure(seq)
                refined = adaptive_rna_constraints(X, tid, v_seg_map, seq=seq, confidence=0.2)
            
            rmsd = calculate_rmsd(refined, target)
            rmsd_list.append(rmsd)
        
        if rmsd_list:
            print(f"LOCAL_VALIDATION_RMSD: {np.mean(rmsd_list):.5f}")

if __name__ == "__main__":
    main()
