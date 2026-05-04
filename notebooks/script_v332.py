import subprocess
import os
import sys
import pandas as pd
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

def install_wheels():
    # Attempt to find wheels dynamically if possible, otherwise use known paths
    potential_base = Path("/kaggle/input")
    wheels = list(potential_base.rglob("*.whl"))
    
    # Priority wheels based on requirements
    priority = ["biopython", "biotite", "rdkit"]
    to_install = []
    for p in priority:
        for w in wheels:
            if p in w.name.lower():
                to_install.append(str(w))
                break
    
    # Fallback to hardcoded if none found (for stability in known environments)
    if not to_install:
        to_install = [
            "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
            "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
            "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
        ]

    for wheel in to_install:
        if os.path.exists(wheel):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
            except Exception as e:
                print(f"Failed to install {wheel}: {e}")

# In Kaggle competition environment, wheels are often pre-provided in specific datasets
if os.path.exists("/kaggle/input"):
    install_wheels()

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True # Defaulting to True for competition script
LOCAL_N_SAMPLES = None

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

def resolve_paths():
    # Dynamic path resolution using rglob as requested
    input_path = Path("/kaggle/input")
    
    # Find test_sequences.csv
    test_files = list(input_path.rglob("test_sequences.csv"))
    if test_files:
        TEST_CSV = str(test_files[0])
        DATA_BASE = str(test_files[0].parent)
    else:
        # Fallback for local or unexpected structure
        DATA_BASE = "./data"
        TEST_CSV = f"{DATA_BASE}/test_sequences.csv"
        
    OUTPUT_CSV = "/kaggle/working/submission.csv" if os.path.exists("/kaggle/working") else "./submission.csv"
    
    # Find train/val files
    train_seq = list(input_path.rglob("train_sequences.csv"))
    train_lbl = list(input_path.rglob("train_labels.csv"))
    val_seq = list(input_path.rglob("validation_sequences.csv"))
    val_lbl = list(input_path.rglob("validation_labels.csv"))
    
    paths = {
        "TEST_CSV": TEST_CSV,
        "OUTPUT_CSV": OUTPUT_CSV,
        "TRAIN_CSV": str(train_seq[0]) if train_seq else None,
        "TRAIN_LBL": str(train_lbl[0]) if train_lbl else None,
        "VAL_CSV": str(val_seq[0]) if val_seq else None,
        "VAL_LBL": str(val_lbl[0]) if val_lbl else None
    }
    return paths

# Hyperparameters
N_SAMPLE      = 40
SEED          = 42
MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score = 2
    al.mismatch_score = -1.5
    al.open_gap_score = -8
    al.extend_gap_score = -0.4
    return al

_aligner = _make_aligner()

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    if "ID" not in labels_df.columns: return coords
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict: continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3: continue
        try:
            aln = next(iter(_aligner.align(query_seq, tseq)))
            norm_s = aln.score / (2 * min(len(query_seq), len(tseq)))
            identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned)
                          for qp, tp in zip(range(qs, qe), range(ts, te))
                          if query_seq[qp] == tseq[tp])
            pct_id = 100 * identical / len(query_seq)
            results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id))
        except: continue
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    try:
        aln = next(iter(_aligner.align(query_seq, template_seq)))
    except:
        return np.zeros((len(query_seq), 3))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs):
            new_coords[qs:qe] = chunk
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0: new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0: new_coords[i] = new_coords[nv] + [3, 0, 0]
            else: new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            d = np.linalg.norm(coords[i] - coords[j])
            if 10.0 < d < 22.0: pairs.append((i, j))
    stems = []
    used = set()
    for i, j in pairs:
        if (i, j) in used: continue
        stem = [(i, j)]; used.add((i, j))
        for k in range(1, 12):
            ni, nj = i + k, j - k
            if ni < L and nj >= 0 and ni < nj and (ni, nj) in pairs:
                stem.append((ni, nj)); used.add((ni, nj))
            else: break
        if len(stem) >= 3: stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, confidence=1.0, passes=2) -> np.ndarray:
    X = coords.copy()
    L_full = len(X)
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── Helical Backbone Periodicity (HBP) Constraint ────────────────────
        # Implement a 11.2 Å distance constraint between phosphorus atoms 
        # P(i) and P(i+2) to accurately reflect the 2.8 Å helical rise and 
        # stabilize the intra-strand backbone periodicity of A-form RNA.
        hbp_strength = 0.30 * strength
        if L_full >= 3:
            d_hbp = X[2:] - X[:-2]
            dist_hbp = np.linalg.norm(d_hbp, axis=1) + 1e-12
            # Target 11.2 Å for i to i+2 phosphorus atoms (backbone rise)
            adj_hbp = d_hbp * ((11.2 - dist_hbp) / dist_hbp)[:, None] * hbp_strength
            X[:-2] -= adj_hbp * 0.5
            X[2:] += adj_hbp * 0.5

        # ── Consecutive Nucleobase Stacking (CNBS) Constraint ────────────────
        cnbs_strength = 0.35 * strength
        if L_full >= 2:
            d_cnbs = X[1:] - X[:-1]
            dist_cnbs = np.linalg.norm(d_cnbs, axis=1) + 1e-12
            adj_cnbs = d_cnbs * ((3.4 - dist_cnbs) / dist_cnbs)[:, None] * cnbs_strength
            X[:-1] -= adj_cnbs * 0.5
            X[1:] += adj_cnbs * 0.5

        # ── Cross-Strand P–P Distance (CSPP) Constraint ──────────────────────
        if stems:
            cspp_strength = 0.25 * strength
            for stem in stems:
                for i, j in stem:
                    tj_minor = j - 3
                    if 0 <= tj_minor < L_full:
                        dv_m = X[i] - X[tj_minor]; dl_m = np.linalg.norm(dv_m) + 1e-6
                        if 11.0 < dl_m < 21.0:
                            adj_m = dv_m * ((15.5 - dl_m) / dl_m) * cspp_strength
                            X[i] += adj_m * 0.5; X[tj_minor] -= adj_m * 0.5
                    tj_major = j + 2
                    if 0 <= tj_major < L_full:
                        dv_M = X[i] - X[tj_major]; dl_M = np.linalg.norm(dv_M) + 1e-6
                        if 7.5 < dl_M < 14.5:
                            adj_M = dv_M * ((10.2 - dl_M) / dl_M) * cspp_strength
                            X[i] += adj_M * 0.5; X[tj_major] -= adj_M * 0.5

        # ── Asymmetric P-P Distance (APDC) Constraint ────────────────────────
        if stems:
            apdc_strength = 0.23 * strength
            for stem in stems:
                for i, j in stem:
                    # P(i) to P(j-1) approx 14.1 Å
                    if 0 <= j-1 < L_full:
                        dv_m = X[i] - X[j-1]; dl_m = np.linalg.norm(dv_m) + 1e-6
                        if 10.0 < dl_m < 18.0:
                            adj = dv_m * ((14.1 - dl_m) / dl_m) * apdc_strength
                            X[i] += adj * 0.5; X[j-1] -= adj * 0.5
                    # P(i) to P(j+1) approx 16.8 Å
                    if 0 <= j+1 < L_full:
                        dv_p = X[i] - X[j+1]; dl_p = np.linalg.norm(dv_p) + 1e-6
                        if 12.0 < dl_p < 21.0:
                            adj = dv_p * ((16.8 - dl_p) / dl_p) * apdc_strength
                            X[i] += adj * 0.5; X[j+1] -= adj * 0.5

        # ── Simple Bond Constraint ───────────────────────────────────────────
        if L_full >= 2:
            d = X[1:] - X[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.3 * strength)
            X[:-1] -= adj * 0.5; X[1:] += adj * 0.5

    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.8] # 2.8 Å helical rise
    return coords

def main() -> None:
    seed_everything(SEED)
    paths = resolve_paths()
    if not paths["TEST_CSV"] or not os.path.exists(paths["TEST_CSV"]):
        print("Test file not found. Exiting.")
        return
        
    test_df = pd.read_csv(paths["TEST_CSV"])
    
    # Load templates
    train_dfs = []
    for k in ["TRAIN_CSV", "VAL_CSV"]:
        if paths[k] and os.path.exists(paths[k]): train_dfs.append(pd.read_csv(paths[k]))
    
    label_dfs = []
    for k in ["TRAIN_LBL", "VAL_LBL"]:
        if paths[k] and os.path.exists(paths[k]): label_dfs.append(pd.read_csv(paths[k]))
        
    train_seqs = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    train_coords = process_labels(pd.concat(label_dfs, ignore_index=True)) if label_dfs else {}

    all_rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Targets"):
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences_detailed(seq, train_seqs, train_coords, top_n=N_SAMPLE) if not train_seqs.empty else []
        
        preds = []
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id) in enumerate(similar):
            if len(preds) >= N_SAMPLE: break
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            # Add some jitter if multiple samples from same template are needed (though find_similar handles uniqueness)
            preds.append(adaptive_rna_constraints(adapted, confidence=sim))
            
        while len(preds) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=len(preds))
            preds.append(adaptive_rna_constraints(dn, confidence=0.1))
            
        for s_idx, coords in enumerate(preds[:N_SAMPLE]):
            for r_idx in range(len(seq)):
                all_rows.append({
                    "target_id": tid, "resid": r_idx + 1,
                    "x_1": coords[r_idx, 0], "y_1": coords[r_idx, 1], "z_1": coords[r_idx, 2],
                    "sample_id": s_idx
                })

    pd.DataFrame(all_rows).to_csv(paths["OUTPUT_CSV"], index=False)
    print(f"Submission saved to {paths['OUTPUT_CSV']}")

if __name__ == "__main__":
    main()
