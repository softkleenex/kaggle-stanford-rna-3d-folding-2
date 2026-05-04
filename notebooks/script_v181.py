import os
import sys

def install_dependencies():
    if os.path.exists("/kaggle/input"):
        # Use os.system to avoid SyntaxError from !pip in .py scripts
        os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
        os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
        os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

install_dependencies()

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
IS_KAGGLE = True
LOCAL_N_SAMPLES = 2

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

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
USE_PROTENIX = True

# ─────────────── Utilities ───────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)

def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

def ensure_required_files(root_dir: str) -> None:
    for p, name in [
        (Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt",          "checkpoint"),
        (Path(root_dir) / "common" / "components.cif",                "CCD file"),
        (Path(root_dir) / "common" / "components.cif.rdkit_mol.pkl",  "CCD cache"),
    ]:
        if not p.exists():
            print(f"Warning: Missing {name} at {p}")

def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [
        {
            "name": row["target_id"],
            "covalent_bonds": [],
            "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}],
        }
        for _, row in df.iterrows()
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def build_configs(input_json_path: str, dump_dir: str, model_name: str):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict):
                deep_update(t[k], v)
            else:
                t[k] = v

    deep_update(base, model_configs[model_name])
    arg_str = " ".join([
        f"--model_name {model_name}",
        f"--input_json_path {input_json_path}",
        f"--dump_dir {dump_dir}",
        "--use_msa false",
        "--use_template false",
        "--use_rna_msa true",
        f"--sample_diffusion.N_sample {N_SAMPLE}",
        f"--seeds {SEED}",
    ])
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0] and i < coords.shape[1]:
                x, y, z = coords[s, i]
            else:
                x, y, z = 0.0, 0.0, 0.0
            row[f"x_{s + 1}"] = float(x)
            row[f"y_{s + 1}"] = float(y)
            row[f"z_{s + 1}"] = float(z)
        rows.append(row)
    return rows

def split_into_chunks(seq_len: int, max_len: int, overlap: int) -> list:
    if seq_len <= max_len: return [(0, seq_len)]
    chunks = []; step = max_len - overlap; pos = 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P, Q):
    cP = P.mean(axis=0); cQ = Q.mean(axis=0)
    Pc = P - cP; Qc = Q - cQ
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = cQ - R @ cP
    return R, t

def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len: return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:coords.shape[0]] = coords
        return out
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        prev_s, prev_e = chunk_ranges[i-1]; cur_s, cur_e = chunk_ranges[i]
        ov_s = cur_s; ov_e = min(prev_e, cur_e); ov_len = ov_e - ov_s
        if ov_len < 3: aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i-1][ov_s-prev_s : ov_e-prev_s]
        cur_ov = chunk_coords_list[i][ov_s-cur_s : ov_e-cur_s]
        R, t = kabsch_align(cur_ov, prev_ov)
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full = np.zeros((seq_len, 3)); weights = np.zeros(seq_len)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        c_len = coords.shape[0]; act_e = min(s + c_len, seq_len); used_len = act_e - s
        w = np.ones(used_len)
        if i > 0:
            ramp_len = min(chunk_ranges[i-1][1], e) - s
            if ramp_len > 0: w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i+1][0]; ramp_len = act_e - next_s
            if ramp_len > 0: w[act_e-s-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)
        full[s:act_e] += coords[:used_len] * w[:, None]; weights[s:act_e] += w
    full[weights > 0] /= weights[weights > 0, None]
    return full

# ─────────────── TBM Functions ───────────────────────────────────────────────
_aligner = PairwiseAligner()
_aligner.mode = "global"

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, top_n=10):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        score = aln.score / (2 * min(len(query_seq), len(tseq)))
        results.append((tid, tseq, score, train_coords_dict[tid]))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords):
    aln = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        new_coords[qs:qe] = template_coords[ts:te]
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i+1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv); new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0: new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0: new_coords[i] = new_coords[nv] + [3, 0, 0]
            else: new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)

# ── STRATEGY: IMPROVED CONSTRAINTS & CHIRALITY ───────────────────────────────
def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
    X = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.85 * (1.0 - min(confidence, 0.98)), 0.05)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # i -> i+1: 5.90 A (A-form P-P distance)
            d1 = C[1:] - C[:-1]; dist1 = np.linalg.norm(d1, axis=1) + 1e-6
            adj1 = d1 * ((5.90 - dist1) / dist1)[:, None] * (0.25 * strength)
            C[:-1] -= adj1; C[1:] += adj1
            
            # i -> i+2: 10.20 A
            d2 = C[2:] - C[:-2]; dist2 = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.20 - dist2) / dist2)[:, None] * (0.15 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            
            # i -> i+3: 13.50 A
            if L >= 4:
                d3 = C[3:] - C[:-3]; dist3 = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((13.50 - dist3) / dist3)[:, None] * (0.10 * strength)
                C[:-3] -= adj3; C[3:] += adj3
                
            # Right-handed chirality (Scalar Triple Product)
            # Standard A-form P virtual torsion and chirality nudge
            if L >= 4:
                v1 = C[1:-2] - C[:-3]; v2 = C[2:-1] - C[1:-2]; v3 = C[3:] - C[2:-1]
                stp = np.sum(v1 * np.cross(v2, v3), axis=1)
                # A-form RNA STP for P atoms is consistently positive (~25-30)
                mask = stp < 20.0
                if np.any(mask):
                    nudge = np.cross(v1[mask], v2[mask])
                    nudge /= (np.linalg.norm(nudge, axis=1, keepdims=True) + 1e-6)
                    # Correctly update in-place using indices
                    idx_mask = np.where(mask)[0]
                    C[idx_mask + 2] += 0.1 * strength * nudge * (25.0 - stp[mask])[:, None]

            # Steric avoidance (A-form core radius is ~4.5A)
            idx = np.arange(L); diff = C[:, None, :] - C[None, :, :]
            dm = np.linalg.norm(diff, axis=2) + 1e-6
            sep = np.abs(idx[:, None] - idx[None, :])
            mask_steric = (sep > 4) & (dm < 4.8)
            if np.any(mask_steric):
                vec = (diff * ((4.8 - dm) / dm)[:, :, None] * mask_steric[:, :, None]).sum(axis=1)
                C += (0.02 * strength) * vec
            X[s:e] = C
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    """Idealized A-form RNA helix fallback using physical parameters."""
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    radius = 9.3; rise = 2.81; twist = 0.57  # ~32.7 degrees
    for i in range(n):
        ang = i * twist
        coords[i] = [radius * np.cos(ang), radius * np.sin(ang), i * rise]
    return coords

# ─────────────── Main Logic ──────────────────────────────────────────────────
def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    sys.path.append(code_dir)
    seed_everything(SEED)

    test_df_full = pd.read_csv(test_csv)
    test_df = test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE else test_df_full
    
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs = pd.read_csv(DEFAULT_VAL_CSV)
    train_lbls = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_lbls = pd.read_csv(DEFAULT_VAL_LBLS)
    
    combined_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True)
    combined_lbls = pd.concat([train_lbls, val_lbls], ignore_index=True)
    train_coords_dict = process_labels(combined_lbls)
    segments_map = {tid: [(0, len(seq))] for tid, seq in zip(test_df["target_id"], test_df["sequence"])}

    results_dict = {}

    print("PHASE 1: TBM Processing...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences(seq, combined_seqs, train_coords_dict, top_n=N_SAMPLE)
        preds = []
        for s_idx, (tmpl_id, tmpl_seq, sim, tmpl_coords) in enumerate(similar):
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            refined = adaptive_rna_constraints(adapted, tid, segments_map, confidence=sim)
            preds.append(refined)
        
        while len(preds) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=SEED + len(preds))
            preds.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        results_dict[tid] = np.stack(preds[:N_SAMPLE], axis=0)

    all_rows = []
    for tid, coords in results_dict.items():
        all_rows.extend(coords_to_rows(tid, test_df[test_df["target_id"]==tid]["sequence"].values[0], coords))

    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]]
    sub.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

    # ── Local Validation Split ───────────────────────────────────────────────
    print("\nCalculating Local Validation RMSD...")
    val_sample = val_seqs.head(5)
    val_preds = []
    val_rmsd_list = []
    for _, row in val_sample.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences(seq, train_seqs, process_labels(train_lbls), top_n=1)
        if not similar: continue
        adapted = adapt_template_to_query(seq, similar[0][1], similar[0][3])
        refined = adaptive_rna_constraints(adapted, tid, {tid: [(0, len(seq))]}, confidence=similar[0][2])
        
        # Calculate RMSD against val_lbls
        true_coords = val_lbls[val_lbls["ID"].str.startswith(tid)].sort_values("resid")[["x_1", "y_1", "z_1"]].values
        if len(refined) == len(true_coords):
            # Kabsch for evaluation
            P = refined - refined.mean(axis=0)
            Q = true_coords - true_coords.mean(axis=0)
            H = P.T @ Q
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            rmsd = np.sqrt(np.mean(np.sum((P @ R.T - Q)**2, axis=1)))
            val_rmsd_list.append(rmsd)
    
    if val_rmsd_list:
        print(f"LOCAL_VALIDATION_RMSD: {np.mean(val_rmsd_list):.5f}")
    else:
        print("LOCAL_VALIDATION_RMSD: 0.00000 (No validation possible)")

if __name__ == "__main__":
    main()
