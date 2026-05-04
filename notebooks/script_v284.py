import subprocess
import sys
import os

# ── Dependency Installation ──────────────────────────────────────────────────
# Using provided offline wheels for Biopython, Biotite, and RDKit
wheels = [
    "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
    "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
]
for wheel in wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheel], check=False)

import pandas as pd
import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets "
          f"will be processed to save time.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ─────────────── Paths & Constants ───────────────────────────────────────────
# Use dynamic path resolution as requested
def get_dataset_path(pattern):
    paths = list(Path('/kaggle/input').rglob(pattern))
    return str(paths[0].parent) if paths else None

DATA_BASE = get_dataset_path('test_sequences.csv') or "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

# Protenix Adjusted Model Paths
PROTENIX_BASE = get_dataset_path('protenix_base_20250630_v1.0.0.pt') or "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
DEFAULT_CODE_DIR = PROTENIX_BASE
DEFAULT_ROOT_DIR = PROTENIX_BASE

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "128"))

MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

USE_PROTENIX = True

def parse_bool(value: str, default: bool = False) -> str:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return "true"
    if v in {"0", "false", "f", "no", "n", "off"}:
        return "false"
    return "true" if default else "false"

USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))

MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

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
    torch.use_deterministic_algorithms(True)

def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

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
        f"--use_msa {USE_MSA}",
        f"--use_template {USE_TEMPLATE}",
        f"--use_rna_msa {USE_RNA_MSA}",
        f"--sample_diffusion.N_sample {MODEL_N_SAMPLE}",
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
    if seq_len <= max_len:
        return [(0, seq_len)]
    chunks = []
    step = max_len - overlap
    pos = 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len:
            break
        pos += step
    return chunks

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

def prsb_steric_enforcement(coords: np.ndarray, iterations: int = 5) -> np.ndarray:
    X = coords.copy()
    n = len(X)
    if n < 2:
        return X
    for _ in range(iterations):
        dists = np.linalg.norm(X[1:] - X[:-1], axis=1)
        too_close = dists < 3.5
        if np.any(too_close):
            diff = X[1:] - X[:-1]
            scale = (3.5 - dists[too_close]) / (dists[too_close] + 1e-6)
            adj = diff[too_close] * scale[:, None] * 0.5
            X[:-1][too_close] -= adj
            X[1:][too_close] += adj
        too_far = dists > 6.5
        if np.any(too_far):
            diff = X[1:] - X[:-1]
            scale = (dists[too_far] - 6.5) / (dists[too_far] + 1e-6)
            adj = diff[too_far] * scale[:, None] * 0.5
            X[:-1][too_far] += adj
            X[1:][too_far] -= adj
        if n > 2:
            d2 = np.linalg.norm(X[2:] - X[:-2], axis=1)
            too_close_2 = d2 < 5.0
            if np.any(too_close_2):
                diff2 = X[2:] - X[:-2]
                scale2 = (5.0 - d2[too_close_2]) / (d2[too_close_2] + 1e-6)
                adj2 = diff2[too_close_2] * scale2[:, None] * 0.4
                X[:-2][too_close_2] -= adj2
                X[2:][too_close_2] += adj2
    return X

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len:
            return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:coords.shape[0]] = coords
        return out
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]
        ov_start = cur_start
        ov_end = min(prev_end, cur_end)
        ov_len = ov_end - ov_start
        if ov_len < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue
        prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
        cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]
        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        transformed = (chunk_coords_list[i] @ R.T) + t
        aligned.append(transformed)
    full = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        chunk_len = coords.shape[0]
        actual_end = min(s + chunk_len, seq_len)
        used_len = actual_end - s
        w = np.ones(used_len, dtype=np.float64)
        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)
        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w
    mask = weights > 0
    full[mask] /= weights[mask, None]
    full = prsb_steric_enforcement(full)
    return full

def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode                           = "global"
    al.match_score                    = 2
    al.mismatch_score                 = -1.5
    al.open_gap_score                 = -8
    al.extend_gap_score               = -0.4
    return al

_aligner = _make_aligner()

def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or str(stoich).strip() == "":
        return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";") for ch, cnt in [part.split(":")]]

def parse_fasta(fasta_content: str) -> dict:
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if cur is not None: out[cur] = "".join(parts)
            cur = line[1:].split()[0]; parts = []
        else:
            parts.append(line.replace(" ", ""))
    if cur is not None: out[cur] = "".join(parts)
    return out

def get_chain_segments(row) -> list:
    seq    = row["sequence"]
    stoich = row.get("stoichiometry", "")
    all_sq = row.get("all_sequences", "")
    if (pd.isna(stoich) or pd.isna(all_sq) or str(stoich).strip() == "" or str(all_sq).strip() == ""):
        return [(0, len(seq))]
    try:
        chain_dict = parse_fasta(all_sq); order = parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None: return [(0, len(seq))]
            for _ in range(cnt):
                segs.append((pos, pos + len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except Exception:
        return [(0, len(seq))]

def build_segments_map(df: pd.DataFrame) -> tuple:
    seg_map = {}
    for _, r in df.iterrows():
        seg_map[r["target_id"]] = get_chain_segments(r)
    return seg_map

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    if labels_df.empty: return coords
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
        aln       = next(iter(_aligner.align(query_seq, tseq)))
        norm_s    = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
        pct_id = 100 * identical / len(query_seq)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln        = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs): new_coords[qs:qe] = chunk
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv); new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0: new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0: new_coords[i] = new_coords[nv] + [3, 0, 0]
            else: new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    
    # Estimate base-pairing (Dummy for P-P distance enforcement)
    # In a real scenario, we'd use secondary structure prediction
    # Here we assume a simple heuristic for antiparallel strands in the same target
    
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # Basic geometry
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            C[:-1] -= d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[1:] += d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            
            # STRATEGY: 12.1 Å distance constraint between P(i) and P(j-2) for base-paired residues (i, j)
            # This reinforces the characteristic deep, narrow major groove of A-form RNA.
            # We use a simple pairing heuristic: residues i and j pair if they are roughly opposite in the sequence
            # for single strands that might fold back, or across segments for multi-chain.
            
            # Heuristic for internal pairing (simplified for global stabilization)
            if L >= 10:
                for i in range(L):
                    # Look for potential pairing partner j
                    # Standard A-form helix has ~11 residues per turn.
                    # We look for a partner roughly across the helix.
                    for j in range(i + 4, L):
                        d_ij = np.linalg.norm(C[i] - C[j])
                        if 15.0 < d_ij < 20.0: # Rough distance for P-P across helix
                            # Apply 12.1 Å constraint between P(i) and P(j-2)
                            if j >= 2:
                                d_p_pj2 = np.linalg.norm(C[i] - C[j-2])
                                if d_p_pj2 > 0.1:
                                    vec = C[i] - C[j-2]
                                    adj = vec * ((12.1 - d_p_pj2) / d_p_pj2) * (0.05 * strength)
                                    C[i] += adj * 0.5
                                    C[j-2] -= adj * 0.5

            # RPBC, AHTC, BSOC, AHRC from previous iterations
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                C[:-3] -= d3 * ((13.8 - d3n) / d3n)[:, None] * (0.05 * strength)
                C[3:] += d3 * ((13.8 - d3n) / d3n)[:, None] * (0.05 * strength)
                
                v1 = C[1:-2] - C[:-3]; v2 = C[2:-1] - C[1:-2]; v3 = C[3:] - C[2:-1]
                n1 = np.cross(v1, v2); n2 = np.cross(v2, v3)
                n1_u = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-6)
                n2_u = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-6)
                
                # A-form twist
                cos_theta = np.clip(np.sum(n1_u * n2_u, axis=1), -1.0, 1.0)
                current_angle = np.arccos(cos_theta)
                target_angle = 32.7 * np.pi / 180.0
                d_angle = (target_angle - current_angle) * (0.08 * strength)
                # (Rotational adjustment omitted for brevity, keeping axial/stacking)
                
                # BSOC Stacking
                b_avg_u = (n1_u + n2_u) / (np.linalg.norm(n1_u + n2_u, axis=1, keepdims=True) + 1e-6)
                proj_v2 = np.sum(v2 * b_avg_u, axis=1, keepdims=True)
                adj_rise = (3.4 * np.sign(proj_v2) - proj_v2) * b_avg_u * (0.12 * strength)
                C[1:-2] -= adj_rise * 0.5; C[2:-1] += adj_rise * 0.5

            X[s:e] = C
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6; coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords

def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    seed_everything(SEED)
    
    try: test_df_full = pd.read_csv(test_csv)
    except Exception: test_df_full = pd.DataFrame()
    test_df = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE else test_df_full).reset_index(drop=True)
    
    try:
        combined_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
        train_coords  = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    except Exception:
        combined_seqs, train_coords = pd.DataFrame(), {}
    
    segments_map = build_segments_map(test_df)
    all_rows = []

    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences_detailed(seq, combined_seqs, train_coords, top_n=5)
        preds = []
        for i in range(N_SAMPLE):
            if i < len(similar):
                tid_t, seq_t, sim, coords_t, _ = similar[i]
                X = adapt_template_to_query(seq, seq_t, coords_t)
                preds.append(adaptive_rna_constraints(X, tid, segments_map, confidence=sim))
            else:
                dn = generate_rna_structure(seq, seed=row.name*1000+i)
                preds.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        
        stacked = np.stack(preds, axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    sub = pd.DataFrame(all_rows)
    if not sub.empty:
        sub.to_csv(output_csv, index=False)
        print(f"Saved submission to {output_csv}")

if __name__ == "__main__":
    main()
