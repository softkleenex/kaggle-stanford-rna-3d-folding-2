import os

os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

import sys
import pandas as pd
import numpy as np
import torch
import gc
import json
import time
from pathlib import Path
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")
LOCAL_N_SAMPLES = 20 # Small subset for local validation speed

if IS_KAGGLE:
    print("Running in KAGGLE mode.")
else:
    print(f"Running in LOCAL mode — test set will be validation subset.")

# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE_KAG = "/kaggle/input/stanford-rna-3d-folding-2"
DATA_BASE_LOC = "data/raw"
DATA_BASE = DATA_BASE_KAG if os.path.exists(DATA_BASE_KAG) else DATA_BASE_LOC

DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

# Protenix paths
DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = 512
CHUNK_OVERLAP = 128

# TBM quality thresholds
MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0

USE_PROTENIX = True
USE_MSA      = "false"
USE_TEMPLATE = "false"
USE_RNA_MSA  = "true"

# ─────────────── General Utilities ───────────────────────────────────────────
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
    test_csv   = os.environ.get("TEST_CSV", DEFAULT_TEST_CSV)
    # If local, run on validation set for RMSD check
    if not IS_KAGGLE:
        test_csv = DEFAULT_VAL_CSV
    output_csv = os.environ.get("SUBMISSION_CSV", DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR", DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

def kabsch_align(P: np.ndarray, Q: np.ndarray):
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
    t = centroid_Q - R @ centroid_P
    return R, t

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
    """Refined A-form Physical Refinement layer (v183 improvement)"""
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.85 * (1.0 - min(confidence, 0.98)), 0.05)
    
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 4: continue
            
            # 1. Right-handed Helical Chirality Constraint (v183 core)
            # Scalar triple product of virtual bond vectors should be positive
            v1 = C[1:-2] - C[:-3]
            v2 = C[2:-1] - C[1:-2]
            v3 = C[3:] - C[2:-1]
            stp = np.sum(np.cross(v1, v2) * v3, axis=1)
            # Target STP for A-form is ~15.0, enforce positive 8.0
            mask_chir = (stp < 8.0)
            if np.any(mask_chir):
                # Push vectors to increase volume
                push = (8.0 - stp[mask_chir])[:, None] * (v1[mask_chir] + v2[mask_chir] + v3[mask_chir]) * (0.008 * strength)
                C[3:][mask_chir] += push
                C[:-3][mask_chir] -= push

            # 2. Refined A-form Distance Constraints
            # i->i+1: 5.95 Å
            d1 = C[1:] - C[:-1]; dist1 = np.linalg.norm(d1, axis=1) + 1e-6
            C[:-1] -= d1 * ((5.95 - dist1) / dist1)[:, None] * (0.25 * strength)
            C[1:] += d1 * ((5.95 - dist1) / dist1)[:, None] * (0.25 * strength)
            
            # i->i+2: 10.2 Å
            if L >= 3:
                d2 = C[2:] - C[:-2]; dist2 = np.linalg.norm(d2, axis=1) + 1e-6
                C[:-2] -= d2 * ((10.2 - dist2) / dist2)[:, None] * (0.15 * strength)
                C[2:] += d2 * ((10.2 - dist2) / dist2)[:, None] * (0.15 * strength)
            
            # i->i+3: 16.3 Å (v183)
            if L >= 4:
                d3 = C[3:] - C[:-3]; dist3 = np.linalg.norm(d3, axis=1) + 1e-6
                C[:-3] -= d3 * ((16.3 - dist3) / dist3)[:, None] * (0.10 * strength)
                C[3:] += d3 * ((16.3 - dist3) / dist3)[:, None] * (0.10 * strength)

            # i->i+4: 20.3 Å (v183)
            if L >= 5:
                d4 = C[4:] - C[:-4]; dist4 = np.linalg.norm(d4, axis=1) + 1e-6
                C[:-4] -= d4 * ((20.3 - dist4) / dist4)[:, None] * (0.08 * strength)
                C[4:] += d4 * ((20.3 - dist4) / dist4)[:, None] * (0.08 * strength)

            # i->i+5: 23.1 Å (v183 corrected)
            if L >= 6:
                d5 = C[5:] - C[:-5]; dist5 = np.linalg.norm(d5, axis=1) + 1e-6
                C[:-5] -= d5 * ((23.1 - dist5) / dist5)[:, None] * (0.06 * strength)
                C[5:] += d5 * ((23.1 - dist5) / dist5)[:, None] * (0.06 * strength)

            # 3. Laplacian Smoothing & Self-Avoidance
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # self-avoidance (v183 increased to 4.5 Å)
            idx = np.arange(L)
            P = C; diff = P[:, None, :] - P[None, :, :]
            dm = np.linalg.norm(diff, axis=2) + 1e-6
            sep = np.abs(idx[:, None] - idx[None, :])
            mask_sa = (sep > 2) & (dm < 4.5)
            if np.any(mask_sa):
                vec = (diff * ((4.5 - dm) / dm)[:, :, None] * mask_sa[:, :, None]).sum(axis=1)
                C += (0.02 * strength) * vec
            
            X[s:e] = C
    return X

def build_input_json(df, json_path):
    data = [{"name": r["target_id"], "covalent_bonds": [], "sequences": [{"rnaSequence": {"sequence": r["sequence"], "count": 1}}]} for _, r in df.iterrows()]
    with open(json_path, "w") as f: json.dump(data, f)

def build_configs(input_json_path, dump_dir, model_name):
    try:
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        base = {**configs_base, "data": data_configs, **inference_configs}
        def deep_update(t, p):
            for k, v in p.items():
                if isinstance(v, dict) and k in t and isinstance(t[k], dict): deep_update(t[k], v)
                else: t[k] = v
        deep_update(base, model_configs[model_name])
        arg_str = f"--model_name {model_name} --input_json_path {input_json_path} --dump_dir {dump_dir} --use_msa {USE_MSA} --use_template {USE_TEMPLATE} --use_rna_msa {USE_RNA_MSA} --sample_diffusion.N_sample {N_SAMPLE} --seeds {SEED}"
        return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)
    except: return None

def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
        for s in range(N_SAMPLE):
            x, y, z = coords[s, i] if s < coords.shape[0] else (0,0,0)
            row[f"x_{s + 1}"], row[f"y_{s + 1}"], row[f"z_{s + 1}"] = float(x), float(y), float(z)
        rows.append(row)
    return rows

def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    seed_everything(SEED)
    
    test_df = pd.read_csv(test_csv)
    if not IS_KAGGLE:
        test_df = test_df.head(LOCAL_N_SAMPLES)
    
    print(f"Processing {len(test_df)} targets...")

    # Load TBM Pool
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs   = pd.read_csv(DEFAULT_VAL_CSV)
    train_lbls = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_lbls   = pd.read_csv(DEFAULT_VAL_LBLS)
    
    combined_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True)
    combined_lbls = pd.concat([train_lbls, val_lbls], ignore_index=True)
    
    def process_labels(labels_df):
        coords = {}
        prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
        for prefix, grp in labels_df.groupby(prefixes):
            coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
        return coords

    train_coords_dict = process_labels(combined_lbls)
    
    segments_map = {r["target_id"]: [(0, len(r["sequence"]))] for _, r in test_df.iterrows()}
    
    all_rows = []
    aligner = PairwiseAligner()
    aligner.mode = "global"

    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        similar = []
        for _, tr in combined_seqs.iterrows():
            t_tid, t_seq = tr["target_id"], tr["sequence"]
            if t_tid not in train_coords_dict or abs(len(t_seq)-len(seq))/max(len(t_seq),len(seq)) > 0.2: continue
            score = aligner.score(seq, t_seq) / (2*min(len(seq), len(t_seq)))
            if score > 0.5: similar.append((t_tid, score))
        similar.sort(key=lambda x: x[1], reverse=True)
        
        preds = []
        for t_tid, sim in similar[:N_SAMPLE]:
            t_coords = train_coords_dict[t_tid]
            # Simple adaptation
            new_c = np.full((len(seq), 3), 0.0)
            ml = min(len(seq), len(t_coords))
            new_c[:ml] = t_coords[:ml]
            # Refine
            preds.append(adaptive_rna_constraints(new_c, tid, segments_map, confidence=sim))
        
        # Fallback de-novo helix
        while len(preds) < N_SAMPLE:
            dn = np.zeros((len(seq), 3))
            for i in range(len(seq)):
                ang = i * 0.6
                dn[i] = [9.3*np.cos(ang), 9.3*np.sin(ang), i*2.8]
            preds.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.1))
            
        stacked = np.stack(preds[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    sub = pd.DataFrame(all_rows)
    sub.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    # Local Validation RMSD
    try:
        merged = sub.merge(combined_lbls, on='ID', suffixes=('_pred', '_true'))
        if len(merged) > 0:
            rmsd_list = []
            for tid, group in merged.groupby(merged['ID'].str.rsplit('_', 1).str[0]):
                P = group[['x_1_pred', 'y_1_pred', 'z_1_pred']].values
                Q = group[['x_1_true', 'y_1_true', 'z_1_true']].values
                if len(P) < 3: continue
                R, t = kabsch_align(P, Q)
                rmsd = np.sqrt(np.mean(np.sum((P @ R + t - Q)**2, axis=1)))
                rmsd_list.append(rmsd)
            if rmsd_list: print(f"LOCAL_VALIDATION_RMSD: {np.mean(rmsd_list):.5f}")
    except Exception as e: print(f"Validation error: {e}")

if __name__ == "__main__":
    main()
