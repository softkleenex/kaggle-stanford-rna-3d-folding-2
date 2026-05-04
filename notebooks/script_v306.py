# !pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
# !pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# !pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl

import os
import sys
import subprocess
import traceback
from pathlib import Path

# ── Install dependencies offline ──────────────────────────────────────────────
# STRATEGY: Implement a 12.8 Å distance constraint between phosphorus atoms P(i) and P(j-2) for base-paired residues (i, j) to stabilize the minor groove width and reinforce the characteristic A-form helical geometry.
# Implementation: Version 306 implements the 12.8 Å distance constraint between P(i) and P(j-2) for base-paired residues (i, j) to stabilize the minor groove width and reinforce the characteristic A-form helical geometry.

def find_wheel(prefix):
    for p in Path('/kaggle/input').rglob(f'{prefix}*.whl'):
        return str(p)
    return None

wheels = [
    find_wheel("biopython"),
    find_wheel("biotite"),
    find_wheel("rdkit")
]
for w in wheels:
    if w and os.path.exists(w):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", w])
        except Exception as e:
            print(f"Failed to install {w}: {e}")
            traceback.print_exc()
    else:
        print(f"Warning: Wheel not found for {w}, skipping installation.")

import pandas as pd

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))

# How many test samples to use when running locally
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets "
          f"will be processed to save time.")

import gc
import json
import time

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    # 1. Try atom_array attributes first
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"):
                    m = m & atom_array.is_rna
                return torch.from_numpy(m).bool()
            
            if hasattr(atom_array, "atom_name"):
                base = atom_array.atom_name == "C1'"
                if hasattr(atom_array, "is_rna"):
                    base = base & atom_array.is_rna
                return torch.from_numpy(base).bool()
        except Exception:
            pass

    f = data["input_feature_dict"]
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
        
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
    mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
    
    c11 = mask11.sum().item()
    c12 = mask12.sum().item()
    
    if abs(c11 - n_tokens) < abs(c12 - n_tokens):
        return mask11
    else:
        return mask12

# ─────────────── Paths & Constants ───────────────────────────────────────────
def find_data_base():
    # Try to find the competition dataset dynamically
    for p in Path('/kaggle/input').rglob('test_sequences.csv'):
        return str(p.parent)
    return "."

def find_protenix_code():
    # Try to find the protenix code directory dynamically
    for p in Path('/kaggle/input').rglob('runner/inference.py'):
        return str(p.parent.parent)
    return "."

DATA_BASE              = find_data_base()
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS     = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

DEFAULT_CODE_DIR       = find_protenix_code()
DEFAULT_ROOT_DIR       = DEFAULT_CODE_DIR

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
    if v in {"1", "true", "t", "yes", "y", "off"}:
        return "false"
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

def ensure_required_files(root_dir: str) -> bool:
    for p, name in [
        (Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt",          "checkpoint"),
        (Path(root_dir) / "common" / "components.cif",                "CCD file"),
        (Path(root_dir) / "common" / "components.cif.rdkit_mol.pkl",  "CCD cache"),
    ]:
        if not p.exists():
            print(f"Missing {name}: {p}")
            return False
    return True

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

    # ── Backbone Pseudo-Torsion (BPT) Global Regularization ─────────────────
    # Enforce A-form η/θ distributions to stabilize the stitched global shape.
    full = adaptive_rna_constraints(full, "stitched", {}, confidence=0.5, passes=2)

    # ── Chain-Continuity Elastic (CCE) Constraint ───────────────────────────
    # Penalize O3' ... P distances exceeding 1.61 Å to prevent unphysical 
    # covalent bond stretching during distal chunk stitching.
    for i in range(seq_len - 1):
        diff = full[i+1] - full[i]
        dist = np.linalg.norm(diff) + 1e-12
        if dist > 1.61:
            # Shift towards each other to satisfy the covalent bond constraint
            adj = diff * ((1.61 - dist) / dist) * 0.5
            full[i] -= adj
            full[i+1] += adj

    return full

def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode                           = "global"
    al.match_score                    = 2
    al.mismatch_score                 = -1.5
    al.open_gap_score                 = -8
    al.extend_gap_score               = -0.4
    al.query_left_open_gap_score      = -8
    al.query_left_extend_gap_score    = -0.4
    al.query_right_open_gap_score     = -8
    al.query_right_extend_gap_score   = -0.4
    al.target_left_open_gap_score     = -8
    al.target_left_extend_gap_score   = -0.4
    al.target_right_open_gap_score    = -8
    al.target_right_extend_gap_score  = -0.4
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
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                out[cur] = "".join(parts)
            cur = line[1:].split()[0]
            parts = []
        else:
            parts.append(line.replace(" ", ""))
    if cur is not None:
        out[cur] = "".join(parts)
    return out

def get_chain_segments(row) -> list:
    seq    = row["sequence"]
    stoich = row.get("stoichiometry", "")
    all_sq = row.get("all_sequences", "")
    if (pd.isna(stoich) or pd.isna(all_sq) or str(stoich).strip() == "" or str(all_sq).strip() == ""):
        return [(0, len(seq))]
    try:
        chain_dict = parse_fasta(all_sq)
        order = parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None:
                return [(0, len(seq))]
            for _ in range(cnt):
                segs.append((pos, pos + len(base)))
                pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except Exception:
        return [(0, len(seq))]

def build_segments_map(df: pd.DataFrame) -> tuple:
    seg_map, stoich_map = {}, {}
    for _, r in df.iterrows():
        tid               = r["target_id"]
        seg_map[tid]      = get_chain_segments(r)
        raw_s             = r.get("stoichiometry", "")
        stoich_map[tid]   = "" if pd.isna(raw_s) else str(raw_s)
    return seg_map, stoich_map

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segs, t_segs = alignment.aligned
    aq, at, qi, ti = [], [], 0, 0
    for (qs, qe), (ts, te) in zip(q_segs, t_segs):
        while qi < qs: aq.append(query_seq[qi]);    at.append("-");              qi += 1
        while ti < ts: aq.append("-");              at.append(template_seq[ti]); ti += 1
        for qp, tp in zip(range(qs, qe), range(ts, te)):
            aq.append(query_seq[qp]); at.append(template_seq[tp])
        qi, ti = qe, te
    while qi < len(query_seq):    aq.append(query_seq[qi]);    at.append("-");              qi += 1
    while ti < len(template_seq): aq.append("-");              at.append(template_seq[ti]); ti += 1
    return "".join(aq), "".join(at)

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict:
            continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3:
            continue
        aln       = next(iter(_aligner.align(query_seq, tseq)))
        norm_s    = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(
            1 for (qs, qe), (ts, te) in zip(*aln.aligned)
            for qp, tp in zip(range(qs, qe), range(ts, te))
            if query_seq[qp] == tseq[tp]
        )
        pct_id = 100 * identical / len(query_seq)
        aq, at = _build_aligned_strings(query_seq, tseq, aln)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln        = next(iter(_aligner.align(query_seq, template_seq)))
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
            elif pv >= 0:
                new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0:
                new_coords[i] = new_coords[nv] + [3, 0, 0]
            else:
                new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    # Simplified stem detection based on spatial proximity (C1'-C1' distance for pairs)
    pairs = []
    dist_threshold = 12.5
    for i in range(L):
        for j in range(i + 4, L):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < dist_threshold:
                pairs.append((i, j))
    
    stems = []
    if not pairs: return stems
    
    # Group contiguous pairs into stems
    pairs.sort()
    used = [False] * len(pairs)
    for i in range(len(pairs)):
        if used[i]: continue
        stem = [pairs[i]]
        used[i] = True
        curr_i, curr_j = pairs[i]
        for k in range(i + 1, len(pairs)):
            if not used[k]:
                next_i, next_j = pairs[k]
                if next_i == curr_i + 1 and next_j == curr_j - 1:
                    stem.append(pairs[k])
                    used[k] = True
                    curr_i, curr_j = next_i, next_j
        if len(stem) >= 3: # Min stem length
            stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    
    # Pre-detect stems for CHS, VBS, BPP, RPC, CSC, CSPP, BPC, VBPR, BPPD, and BPCR
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── Base-Pair C2' Registration (BPCR) Constraint ────────────────────
        if stems:
            bpcr_strength = 0.32 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
                        v1_i = X[i_s+1] - X[i_s]; v2_i = X[i_s-1] - X[i_s]
                        n_i = np.cross(v1_i, v2_i); dn_i = np.linalg.norm(n_i) + 1e-6
                        c2_i = X[i_s] + (n_i / dn_i) * 3.3
                        v1_j = X[j_s+1] - X[j_s]; v2_j = X[j_s-1] - X[j_s]
                        n_j = np.cross(v1_j, v2_j); dn_j = np.linalg.norm(n_j) + 1e-6
                        c2_j = X[j_s] + (n_j / dn_j) * 3.3
                        dv_bpcr = c2_i - c2_j
                        dl_bpcr = np.linalg.norm(dv_bpcr) + 1e-6
                        if 9.0 < dl_bpcr < 15.0:
                            adj_bpcr = dv_bpcr * ((12.2 - dl_bpcr) / dl_bpcr) * bpcr_strength
                            X[i_s] += adj_bpcr * 0.5
                            X[j_s] -= adj_bpcr * 0.5

        # ── Base-Pair C1' Distance (BPC) Constraint ─────────────────────────
        if stems:
            bpc_strength = 0.35 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
                        v1_i = X[i_s+1] - X[i_s]; v2_i = X[i_s-1] - X[i_s]
                        n_i = np.cross(v1_i, v2_i); dn_i = np.linalg.norm(n_i) + 1e-6
                        c1_i = X[i_s] + (n_i / dn_i) * 2.1
                        v1_j = X[j_s+1] - X[j_s]; v2_j = X[j_s-1] - X[j_s]
                        n_j = np.cross(v1_j, v2_j); dn_j = np.linalg.norm(n_j) + 1e-6
                        c1_j = X[j_s] + (n_j / dn_j) * 2.1
                        dv_bpc = c1_i - c1_j
                        dl_bpc = np.linalg.norm(dv_bpc) + 1e-6
                        if 8.0 < dl_bpc < 13.0:
                            adj_bpc = dv_bpc * ((10.5 - dl_bpc) / dl_bpc) * bpc_strength
                            X[i_s] += adj_bpc * 0.5
                            X[j_s] -= adj_bpc * 0.5

        # ── Base-Pair Phosphorus Distance (BPPD) Constraint ──────────────────
        if stems:
            bppd_strength = 0.38 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if i_s < len(X) and j_s < len(X):
                        dv_bppd = X[i_s] - X[j_s]
                        dl_bppd = np.linalg.norm(dv_bppd) + 1e-6
                        if 12.0 < dl_bppd < 24.0:
                            adj_bppd = dv_bppd * ((17.5 - dl_bppd) / dl_bppd) * bppd_strength
                            X[i_s] += adj_bppd * 0.5
                            X[j_s] -= adj_bppd * 0.5

        # ── Inter-Strand Minor/Major Groove Width Constraint ─────────────────
        if stems:
            ismgw_strength = 0.35 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    # Minor groove stabilization
                    if i_s < len(X) and (j_s + 3) < len(X):
                        dv_ismgw1 = X[i_s] - X[j_s + 3]
                        dl_ismgw1 = np.linalg.norm(dv_ismgw1) + 1e-6
                        if 10.0 < dl_ismgw1 < 25.0:
                            adj_ismgw1 = dv_ismgw1 * ((16.8 - dl_ismgw1) / dl_ismgw1) * ismgw_strength
                            X[i_s] += adj_ismgw1 * 0.5
                            X[j_s + 3] -= adj_ismgw1 * 0.5
                    if j_s < len(X) and (i_s + 3) < len(X):
                        dv_ismgw2 = X[j_s] - X[i_s + 3]
                        dl_ismgw2 = np.linalg.norm(dv_ismgw2) + 1e-6
                        if 10.0 < dl_ismgw2 < 25.0:
                            adj_ismgw2 = dv_ismgw2 * ((16.8 - dl_ismgw2) / dl_ismgw2) * ismgw_strength
                            X[j_s] += adj_ismgw2 * 0.5
                            X[i_s + 3] -= adj_ismgw2 * 0.5

        # ── C4' to P Distance (CPP) Constraint ──────────────────────────────
        cpp_strength = 0.35 * strength
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 2: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            cpp_mask = in_stem_seg[:-1] & in_stem_seg[1:]
            if np.any(cpp_mask):
                d_cpp = C[1:] - C[:-1]; dist_cpp = np.linalg.norm(d_cpp, axis=1) + 1e-6
                adj_cpp = d_cpp[cpp_mask] * ((5.85 - dist_cpp[cpp_mask]) / dist_cpp[cpp_mask])[:, None] * cpp_strength
                C[:-1][cpp_mask] -= adj_cpp * 0.5
                C[1:][cpp_mask] += adj_cpp * 0.5
            X[s:e] = C

        # ── Base-Pair Planarity (BPP) Constraint ─────────────────────────────
        if stems:
            bpp_strength = 0.15 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if i_s > 0 and i_s < len(X)-1 and j_s > 0 and j_s < len(X)-1:
                        v1_i = X[i_s+1] - X[i_s]; v2_i = X[i_s-1] - X[i_s]
                        n_i = np.cross(v1_i, v2_i); dn_i = np.linalg.norm(n_i) + 1e-6; n_i /= dn_i
                        v1_j = X[j_s+1] - X[j_s]; v2_j = X[j_s-1] - X[j_s]
                        n_j = np.cross(v1_j, v2_j); dn_j = np.linalg.norm(n_j) + 1e-6; n_j /= dn_j
                        dot = np.dot(n_i, n_j)
                        if abs(dot) < 0.98:
                            target_n = n_i if dot > 0 else -n_i
                            adj_bpp = (target_n - n_j) * bpp_strength
                            X[j_s+1] += adj_bpp; X[j_s-1] += adj_bpp
                            X[i_s+1] -= adj_bpp; X[i_s-1] -= adj_bpp

        # ── Co-axial Helix Stacking (CHS) Constraint ─────────────────────────
        if len(stems) >= 2:
            chs_strength = 0.12 * strength
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    stem_a = stems[i]; stem_b = stems[j]
                    terms_a = [stem_a[0], stem_a[-1]]; terms_b = [stem_b[0], stem_b[-1]]
                    for (ai, aj) in terms_a:
                        for (bi, bj) in terms_b:
                            dist_ij = min(abs(ai - bi), abs(ai - bj), abs(aj - bi), abs(aj - bj))
                            if dist_ij <= 5: 
                                vec_a = X[ai] - X[aj]; d_a = np.linalg.norm(vec_a) + 1e-6
                                vec_b = X[bi] - X[bj]; d_b = np.linalg.norm(vec_b) + 1e-6
                                center_a = (X[ai] + X[aj]) * 0.5; center_b = (X[bi] + X[bj]) * 0.5
                                d_centers = np.linalg.norm(center_a - center_b) + 1e-6
                                if d_centers < 15.0:
                                    adj_c = (center_a - center_b) * ((10.0 - d_centers) / d_centers) * chs_strength
                                    X[ai] += adj_c; X[aj] += adj_c; X[bi] -= adj_c; X[bj] -= adj_c
                                cos_sim = np.dot(vec_a, vec_b) / (d_a * d_b)
                                if cos_sim < 0.95:
                                    target_vec = (vec_a / d_a + vec_b / d_b) * 0.5
                                    target_vec /= np.linalg.norm(target_vec) + 1e-6
                                    X[ai] += (target_vec * d_a * 0.5 + center_a - X[ai]) * chs_strength
                                    X[aj] += (-target_vec * d_a * 0.5 + center_a - X[aj]) * chs_strength
                                    X[bi] += (target_vec * d_b * 0.5 + center_b - X[bi]) * chs_strength
                                    X[bj] += (-target_vec * d_b * 0.5 + center_b - X[bj]) * chs_strength

        # ── Coaxial Stacking (CSC) Constraint ────────────────────────────────
        if len(stems) >= 2:
            csc_strength = 0.20 * strength
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    ends_a = [stems[i][0], stems[i][-1]]; ends_b = [stems[j][0], stems[j][-1]]
                    for (ai, aj) in ends_a:
                        for (bi, bj) in ends_b:
                            if abs(ai - bi) <= 3 or abs(ai - bj) <= 3 or abs(aj - bi) <= 3 or abs(aj - bj) <= 3:
                                axis_a = X[ai] - X[aj]; axis_b = X[bi] - X[bj]
                                d_a = np.linalg.norm(axis_a) + 1e-6; d_b = np.linalg.norm(axis_b) + 1e-6
                                u_a = axis_a / d_a; u_b = axis_b / d_b; cos_val = np.dot(u_a, u_b)
                                if abs(cos_val) < 0.95:
                                    target_u = u_a if cos_val > 0 else -u_a
                                    adj_csc = (target_u - u_b) * csc_strength
                                    X[bi] += adj_csc * d_b * 0.5; X[bj] -= adj_csc * d_b * 0.5

        # ── Vertical Base-Pair Rise (VBPR) Constraint ───────────────────────
        if stems:
            vbpr_strength = 0.22 * strength
            for stem in stems:
                for k in range(len(stem) - 1):
                    i1, j1 = stem[k]; i2, j2 = stem[k+1]
                    centroid1 = (X[i1] + X[j1]) * 0.5; centroid2 = (X[i2] + X[j2]) * 0.5
                    dv_vbpr = centroid2 - centroid1; dl_vbpr = np.linalg.norm(dv_vbpr) + 1e-6
                    if 1.5 < dl_vbpr < 5.0:
                        adj_vbpr = dv_vbpr * ((2.8 - dl_vbpr) / dl_vbpr) * vbpr_strength
                        X[i1] -= adj_vbpr * 0.25; X[j1] -= adj_vbpr * 0.25
                        X[i2] += adj_vbpr * 0.25; X[j2] += adj_vbpr * 0.25

        # ── Cross-Strand P–P Distance (CSPP) Constraint ──────────────────────
        if stems:
            cspp_strength = 0.25 * strength
            for stem in stems:
                for pair in stem:
                    i, j = pair
                    tj_mgt = j - 1
                    if 0 <= tj_mgt < len(X) and i < len(X):
                        dv_mgt = X[i] - X[tj_mgt]; dl_mgt = np.linalg.norm(dv_mgt) + 1e-6
                        if 10.0 < dl_mgt < 21.0:
                            adj_mgt = dv_mgt * ((15.5 - dl_mgt) / dl_mgt) * cspp_strength
                            X[i] += adj_mgt * 0.5; X[tj_mgt] -= adj_mgt * 0.5
                    tj_mgo = j + 1
                    if 0 <= tj_mgo < len(X) and i < len(X):
                        dv_mgo = X[i] - X[tj_mgo]; dl_mgo = np.linalg.norm(dv_mgo) + 1e-6
                        if 14.0 < dl_mgo < 24.0:
                            adj_mgo = dv_mgo * ((19.1 - dl_mgo) / dl_mgo) * cspp_strength
                            X[i] += adj_mgo * 0.5; X[tj_mgo] -= adj_mgo * 0.5
                    tj_mgs = j - 2
                    if 0 <= tj_mgs < len(X) and i < len(X):
                        dv_mgs = X[i] - X[tj_mgs]; dl_mgs = np.linalg.norm(dv_mgs) + 1e-6
                        if 10.0 < dl_mgs < 18.0:
                            # Strategy Version 306: Implement a 12.8 Å distance constraint between phosphorus atoms 
                            # P(i) and P(j-2) for base-paired residues (i, j) to stabilize the minor groove width 
                            # and reinforce the characteristic A-form helical geometry.
                            adj_mgs = dv_mgs * ((12.8 - dl_mgs) / dl_mgs) * cspp_strength
                            X[i] += adj_mgs * 0.5; X[tj_mgs] -= adj_mgs * 0.5
                    tj_mgn = j - 3
                    if 0 <= tj_mgn < len(X) and i < len(X):
                        dv_mgn = X[i] - X[tj_mgn]; dl_mgn = np.linalg.norm(dv_mgn) + 1e-6
                        if 8.0 < dl_mgn < 18.0:
                            adj_mgn = dv_mgn * ((10.8 - dl_mgn) / dl_mgn) * cspp_strength
                            X[i] += adj_mgn * 0.5; X[tj_mgn] -= adj_mgn * 0.5
                    tj_minor_wide = j + 2
                    if 0 <= tj_minor_wide < len(X) and i < len(X):
                        dv_M = X[i] - X[tj_minor_wide]; dl_M = np.linalg.norm(dv_M) + 1e-6
                        if 10.0 < dl_M < 20.0:
                            adj_M = dv_M * ((15.1 - dl_M) / dl_M) * cspp_strength
                            X[i] += adj_M * 0.5; X[tj_minor_wide] -= adj_M * 0.5
                    tj_mgc = j - 5
                    if 0 <= tj_mgc < len(X) and i < len(X):
                        dv_mgc = X[i] - X[tj_mgc]; dl_mgc = np.linalg.norm(dv_mgc) + 1e-6
                        if 9.0 < dl_mgc < 18.0:
                            adj_mgc = dv_mgc * ((13.2 - dl_mgc) / dl_mgc) * cspp_strength
                            X[i] += adj_mgc * 0.5; X[tj_mgc] -= adj_mgc * 0.5
                    tj_nmg = j + 3
                    if 0 <= tj_nmg < len(X) and i < len(X):
                        dv_nmg = X[i] - X[tj_nmg]; dl_nmg = np.linalg.norm(dv_nmg) + 1e-6
                        if 10.0 < dl_nmg < 19.0:
                            adj_nmg = dv_nmg * ((13.2 - dl_nmg) / dl_nmg) * cspp_strength
                            X[i] += adj_nmg * 0.5; X[tj_nmg] -= adj_nmg * 0.5

        # ── O2'(i) ... O4'(i+1) Backbone Stabilization (OBS) Constraint ─────
        obs_strength = 0.28 * strength
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 2: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            obs_mask = in_stem_seg[:-1] & in_stem_seg[1:]
            if np.any(obs_mask):
                d_obs = C[1:] - C[:-1]; dist_obs = np.linalg.norm(d_obs, axis=1) + 1e-6
                adj_obs = d_obs[obs_mask] * ((6.2 - dist_obs[obs_mask]) / dist_obs[obs_mask])[:, None] * obs_strength
                C[:-1][obs_mask] -= adj_obs * 0.5; C[1:][obs_mask] += adj_obs * 0.5
            X[s:e] = C

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
                rpc_mask = in_stem[:-1] & in_stem[1:]
                if np.any(rpc_mask):
                    rpc_strength = 0.32 * strength
                    adj_rpc = d[rpc_mask] * ((5.90 - dist[rpc_mask]) / dist[rpc_mask])[:, None] * rpc_strength
                    C[:-1][rpc_mask] -= adj_rpc * 0.5; C[1:][rpc_mask] += adj_rpc * 0.5
                stretch_mask = (~rpc_mask) & (dist > 6.2)
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
                d1 = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-9; d2 = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9
                n1 /= d1; n2 /= d2
                cos_phi = np.sum(n1 * n2, axis=1); cos_phi = np.clip(cos_phi, -1.0, 1.0)
                phi = np.arccos(cos_phi); sign = np.sum(v12 * n2, axis=1); phi = np.where(sign < 0, -phi, phi)
                target_phi = 0.61; diff_phi = target_phi - phi
                axis = v23 / (np.linalg.norm(v23, axis=1, keepdims=True) + 1e-9)
                force = np.cross(n2, axis) * diff_phi[:, None] * bpt_strength
                C[3:] += force * 0.5; C[:-3] -= force * 0.5

            # ── Short-Range Helical Periodicity (SRHP) Constraint ─────────────
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            srhp_mask = in_stem[:-2] & in_stem[2:]
            if np.any(srhp_mask):
                srhp_strength = 0.25 * strength
                adj2_srhp = d2[srhp_mask] * ((11.2 - d2n[srhp_mask]) / d2n[srhp_mask])[:, None] * srhp_strength
                C[:-2][srhp_mask] -= adj2_srhp * 0.5; C[2:][srhp_mask] += adj2_srhp * 0.5
            ns_mask = ~srhp_mask
            if np.any(ns_mask):
                adj2_ns = d2[ns_mask] * ((11.2 - d2n[ns_mask]) / d2n[ns_mask])[:, None] * (0.10 * strength)
                C[:-2][ns_mask] -= adj2_ns * 0.5; C[2:][ns_mask] += adj2_ns * 0.5

            # ── Medium-Range Helical Periodicity (MRHP) Constraint ────────────
            if L >= 4:
                d_mrhp = C[3:] - C[:-3]; dist_mrhp = np.linalg.norm(d_mrhp, axis=1) + 1e-6
                mrhp_mask = in_stem[:-3] & in_stem[3:]
                if np.any(mrhp_mask):
                    mrhp_strength = 0.20 * strength
                    adj_mrhp = d_mrhp[mrhp_mask] * ((15.2 - dist_mrhp[mrhp_mask]) / dist_mrhp[mrhp_mask])[:, None] * mrhp_strength
                    C[:-3][mrhp_mask] -= adj_mrhp * 0.5; C[3:][mrhp_mask] += adj_mrhp * 0.5
            
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((13.5 - d3n) / d3n)[:, None] * (0.05 * strength)
                C[:-3] -= adj3; C[3:] += adj3
            if L >= 5:
                d4 = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((16.5 - d4n) / d4n)[:, None] * (0.03 * strength)
                C[:-4] -= adj4; C[4:] += adj4
            if L >= 5:
                d_lrhp = C[4:] - C[:-4]; dist_lrhp = np.linalg.norm(d_lrhp, axis=1) + 1e-6
                lrhp_mask = in_stem[:-4] & in_stem[4:]
                if np.any(lrhp_mask):
                    adj_lrhp = d_lrhp[lrhp_mask] * ((19.1 - dist_lrhp[lrhp_mask]) / dist_lrhp[lrhp_mask])[:, None] * (0.15 * strength)
                    C[:-4][lrhp_mask] -= adj_lrhp * 0.5; C[4:][lrhp_mask] += adj_lrhp * 0.5
    return X

# ─────────────── Model Inference ─────────────────────────────────────────────
def run_inference(test_csv: str, output_csv: str, code_dir: str, root_dir: str):
    sys.path.append(code_dir)
    from runner.inference import InferenceRunner
    df = pd.read_csv(test_csv)
    if not IS_KAGGLE and LOCAL_N_SAMPLES: df = df.head(LOCAL_N_SAMPLES)
    seg_map, stoich_map = build_segments_map(df)
    tmp_json = "input.json"; dump_dir = "dump"; os.makedirs(dump_dir, exist_ok=True)
    build_input_json(df, tmp_json)
    configs = build_configs(tmp_json, dump_dir, MODEL_NAME)
    runner = InferenceRunner(configs); results = runner.run() 
    all_rows = []
    for tid, data_list in results.items():
        if not data_list: continue
        data = data_list[0]
        try:
            atom_array = data.get("atom_array")
            if atom_array is None: continue
            p_mask = (atom_array.atom_name == "P") & atom_array.is_rna
            if p_mask.sum() == 0: p_mask = (atom_array.atom_name == "C1'") & atom_array.is_rna
            seq = df[df["target_id"] == tid]["sequence"].values[0]
            sample_coords = []
            for s_idx in range(min(len(data_list), N_SAMPLE)):
                s_data = data_list[s_idx]; s_aa = s_data.get("atom_array")
                s_p_mask = (s_aa.atom_name == "P") & s_aa.is_rna
                if s_p_mask.sum() == 0: s_p_mask = (s_aa.atom_name == "C1'") & s_aa.is_rna
                s_c = s_aa.coord[s_p_mask].astype(np.float64)
                conf = 0.5
                if hasattr(s_aa, "b_factor"): conf = np.mean(s_aa.b_factor[s_p_mask]) / 100.0
                s_c = adaptive_rna_constraints(s_c, tid, seg_map, confidence=conf)
                sample_coords.append(s_c)
            final_coords = np.stack(sample_coords)
            all_rows.extend(coords_to_rows(tid, seq, final_coords))
        except Exception as e:
            print(f"Error processing {tid}: {e}"); traceback.print_exc()
    sub_df = pd.DataFrame(all_rows); sub_df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")

# ─────────────── Main Execution ──────────────────────────────────────────────
def main():
    seed_everything(SEED)
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    if not ensure_required_files(root_dir):
        df = pd.read_csv(test_csv)
        if not IS_KAGGLE and LOCAL_N_SAMPLES: df = df.head(LOCAL_N_SAMPLES)
        dummy = []
        for _, r in df.iterrows():
            tid, seq = r["target_id"], r["sequence"]
            c = np.zeros((N_SAMPLE, len(seq), 3))
            dummy.extend(coords_to_rows(tid, seq, c))
        pd.DataFrame(dummy).to_csv(output_csv, index=False); return
    try:
        run_inference(test_csv, output_csv, code_dir, root_dir)
    except Exception as e:
        print(f"Inference failed: {e}"); traceback.print_exc()
        df = pd.read_csv(test_csv); dummy = []
        for _, r in df.iterrows():
            tid, seq = r["target_id"], r["sequence"]
            c = np.zeros((N_SAMPLE, len(seq), 3))
            dummy.extend(coords_to_rows(tid, seq, c))
        pd.DataFrame(dummy).to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
