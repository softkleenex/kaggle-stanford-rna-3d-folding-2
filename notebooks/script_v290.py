# !pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
# !pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# !pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl

import os
import sys
import subprocess
import traceback
from pathlib import Path

# ── Install dependencies offline ──────────────────────────────────────────────
# STRATEGY: Implement a 10.5 Å distance constraint between C1' atoms of base-paired residues (i, j) to stabilize the base-pair width and maintain the characteristic glycosidic orientation of A-form RNA.
# Implementation: Version 290 implements the 10.5 Å distance constraint between C1' atoms (estimated from Phosphorus positions) to stabilize the base-pair width and maintain the characteristic glycosidic orientation of A-form RNA.

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
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
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
    
    # Pre-detect stems for CHS, VBS, BPP, RPC, CSC, CSPP, BPC, VBPR, and BPPD
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── Base-Pair C1' Distance (BPC) Constraint ─────────────────────────
        # Implement a 10.5 Å distance constraint between C1'(i) and C1'(j) 
        # atoms for base-paired residues (i, j) to stabilize the Watson-Crick 
        # pair width and maintain the characteristic glycosidic orientation 
        # of A-form RNA. Since X stores P positions, we estimate C1' as a 
        # 2.1 Å offset from P along the local normal to the backbone.
        if stems:
            bpc_strength = 0.35 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
                        # Estimate local C1' for residue i
                        v1_i = X[i_s+1] - X[i_s]; v2_i = X[i_s-1] - X[i_s]
                        n_i = np.cross(v1_i, v2_i); dn_i = np.linalg.norm(n_i) + 1e-6
                        c1_i = X[i_s] + (n_i / dn_i) * 2.1
                        
                        # Estimate local C1' for residue j
                        v1_j = X[j_s+1] - X[j_s]; v2_j = X[j_s-1] - X[j_s]
                        n_j = np.cross(v1_j, v2_j); dn_j = np.linalg.norm(n_j) + 1e-6
                        c1_j = X[j_s] + (n_j / dn_j) * 2.1
                        
                        dv_bpc = c1_i - c1_j
                        dl_bpc = np.linalg.norm(dv_bpc) + 1e-6
                        # A-form C1'-C1' distance is typically around 10.5 Å
                        if 8.0 < dl_bpc < 13.0:
                            adj_bpc = dv_bpc * ((10.5 - dl_bpc) / dl_bpc) * bpc_strength
                            X[i_s] += adj_bpc * 0.5
                            X[j_s] -= adj_bpc * 0.5

        # ── Base-Pair Phosphorus Distance (BPPD) Constraint ──────────────────
        # Implement a 17.5 Å distance constraint between phosphorus atoms P(i) 
        # and P(j) for base-paired residues (i, j) to stabilize the helical 
        # diameter and maintain the characteristic duplex width of A-form RNA.
        if stems:
            bppd_strength = 0.38 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if i_s < len(X) and j_s < len(X):
                        dv_bppd = X[i_s] - X[j_s]
                        dl_bppd = np.linalg.norm(dv_bppd) + 1e-6
                        # A-form P(i)-P(j) distance is typically around 17.5 Å
                        if 12.0 < dl_bppd < 24.0:
                            adj_bppd = dv_bppd * ((17.5 - dl_bppd) / dl_bppd) * bppd_strength
                            X[i_s] += adj_bppd * 0.5
                            X[j_s] -= adj_bppd * 0.5

        # ── Inter-Strand Minor Groove Width (ISMGW) Constraint ───────────────
        # Implement a 16.8 Å distance constraint between phosphorus atoms P(i) 
        # and P(j+3) for base-paired residues (i, j) to stabilize the inter-strand 
        # minor groove width and maintain the characteristic wide, shallow minor 
        # groove of A-form RNA.
        if stems:
            ismgw_strength = 0.35 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    # Constraint between P(i) and P(j+3)
                    if i_s < len(X) and (j_s + 3) < len(X):
                        dv_ismgw1 = X[i_s] - X[j_s + 3]
                        dl_ismgw1 = np.linalg.norm(dv_ismgw1) + 1e-6
                        if 10.0 < dl_ismgw1 < 25.0:
                            adj_ismgw1 = dv_ismgw1 * ((16.8 - dl_ismgw1) / dl_ismgw1) * ismgw_strength
                            X[i_s] += adj_ismgw1 * 0.5
                            X[j_s + 3] -= adj_ismgw1 * 0.5
                            
                    # Symmetric constraint between P(j) and P(i+3)
                    if j_s < len(X) and (i_s + 3) < len(X):
                        dv_ismgw2 = X[j_s] - X[i_s + 3]
                        dl_ismgw2 = np.linalg.norm(dv_ismgw2) + 1e-6
                        if 10.0 < dl_ismgw2 < 25.0:
                            adj_ismgw2 = dv_ismgw2 * ((16.8 - dl_ismgw2) / dl_ismgw2) * ismgw_strength
                            X[j_s] += adj_ismgw2 * 0.5
                            X[i_s + 3] -= adj_ismgw2 * 0.5


        # ── C4' to P Distance (CPP) Constraint ──────────────────────────────
        # Implement a 5.8 Å distance constraint between the C4' and P atoms of 
        # the same nucleotide to enforce the rigid C3'-endo ribose pucker 
        # geometry essential for maintaining stable A-form RNA helical parameters.
        # Since X stores P-atom positions, we model this as an internal residue 
        # stabilization that influences the local curvature and P-P spacing.
        # We model this by enforcing a more rigid local P-P spacing of 5.85 Å
        # in helical regions which is strongly coupled to the C3'-endo pucker.
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
                # The 5.8 Å C4'-P constraint manifests as a ~5.85 Å P-P step in A-form.
                adj_cpp = d_cpp[cpp_mask] * ((5.85 - dist_cpp[cpp_mask]) / dist_cpp[cpp_mask])[:, None] * cpp_strength
                C[:-1][cpp_mask] -= adj_cpp * 0.5
                C[1:][cpp_mask] += adj_cpp * 0.5
            X[s:e] = C

        # ── Base-Pair Planarity (BPP) Constraint ─────────────────────────────
        # Enforce co-planar geometry between paired nucleobases to prevent
        # unphysical propeller-twist distortions in helical regions.
        if stems:
            bpp_strength = 0.15 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if i_s > 0 and i_s < len(X)-1 and j_s > 0 and j_s < len(X)-1:
                        # Plane normal estimation for base i
                        v1_i = X[i_s+1] - X[i_s]; v2_i = X[i_s-1] - X[i_s]
                        n_i = np.cross(v1_i, v2_i); dn_i = np.linalg.norm(n_i) + 1e-6
                        n_i /= dn_i
                        # Plane normal estimation for base j
                        v1_j = X[j_s+1] - X[j_s]; v2_j = X[j_s-1] - X[j_s]
                        n_j = np.cross(v1_j, v2_j); dn_j = np.linalg.norm(n_j) + 1e-6
                        n_j /= dn_j
                        # Enforce parallelism
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

        # ── Coaxial Stacking (CSC) Constraint ────────────────────────────────
        # Enforce collinearity of helical axes at multi-way junctions to prevent 
        # unphysical kinks in continuous A-form stacks.
        if len(stems) >= 2:
            csc_strength = 0.20 * strength
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    # Check junction proximity
                    ends_a = [stems[i][0], stems[i][-1]]
                    ends_b = [stems[j][0], stems[j][-1]]
                    for (ai, aj) in ends_a:
                        for (bi, bj) in ends_b:
                            # Distance between junctions
                            if abs(ai - bi) <= 3 or abs(ai - bj) <= 3 or abs(aj - bi) <= 3 or abs(aj - bj) <= 3:
                                axis_a = X[ai] - X[aj]; axis_b = X[bi] - X[bj]
                                d_a = np.linalg.norm(axis_a) + 1e-6
                                d_b = np.linalg.norm(axis_b) + 1e-6
                                u_a = axis_a / d_a; u_b = axis_b / d_b
                                cos_val = np.dot(u_a, u_b)
                                if abs(cos_val) < 0.95:
                                    # Force collinearity
                                    target_u = u_a if cos_val > 0 else -u_a
                                    adj_csc = (target_u - u_b) * csc_strength
                                    X[bi] += adj_csc * d_b * 0.5; X[bj] -= adj_csc * d_b * 0.5

        # ── Vertical Base-Pair Rise (VBPR) Constraint ───────────────────────
        # Implement a 2.8 Å vertical distance constraint between the centroids
        # of consecutive base pairs (i:j and i+1:j-1) to enforce the 
        # characteristic A-form helical rise and prevent unphysical stretching.
        if stems:
            vbpr_strength = 0.22 * strength
            for stem in stems:
                # stem is a list of pairs (i, j)
                for k in range(len(stem) - 1):
                    i1, j1 = stem[k]
                    i2, j2 = stem[k+1]
                    # i2 = i1 + 1, j2 = j1 - 1
                    
                    centroid1 = (X[i1] + X[j1]) * 0.5
                    centroid2 = (X[i2] + X[j2]) * 0.5
                    
                    dv_vbpr = centroid2 - centroid1
                    dl_vbpr = np.linalg.norm(dv_vbpr) + 1e-6
                    
                    if 1.5 < dl_vbpr < 5.0: # Only apply if reasonably close
                        adj_vbpr = dv_vbpr * ((2.8 - dl_vbpr) / dl_vbpr) * vbpr_strength
                        # Distribute correction to all four atoms
                        X[i1] -= adj_vbpr * 0.25
                        X[j1] -= adj_vbpr * 0.25
                        X[i2] += adj_vbpr * 0.25
                        X[j2] += adj_vbpr * 0.25

        # ── Cross-Strand P–P Distance (CSPP) Constraint ──────────────────────
        # Implement a 15.5 Å cross-strand P–P distance (minor groove) and
        # a 15.1 Å cross-strand P–P distance (minor groove reinforcement) to stabilize the
        # characteristic wide-shallow A-form architecture.
        # Version 266: Adds a 13.2 Å P(i)-P(j-5) distance constraint for base-paired residues (i, j).
        # Version 273: Updates P(i)-P(j+2) to 15.1 Å to stabilize the A-form minor groove width.
        # Version 274: Adds a 12.8 Å P(i)-P(j-1) distance constraint to reinforce the helical twist.
        # Version 279: Updates P(i)-P(j-1) to 15.5 Å for major groove diagonal stabilization.
        # Version 289: Adds a 10.8 Å P(i)-P(j-3) distance constraint to stabilize inter-strand major groove narrowing.
        if stems:
            cspp_strength = 0.25 * strength
            for stem in stems:
                # stem is list of pairs (i, j)
                for pair in stem:
                    i, j = pair
                    
                    # ── A-form Inter-Strand Major Groove Diagonal Constraint ────────────
                    # Implement a 15.5 Å distance constraint between phosphorus 
                    # atoms P(i) and P(j-1) for base-paired residues (i, j) 
                    # to stabilize the inter-strand major groove diagonal and 
                    # reinforce the characteristic A-form helical twist.
                    tj_diag = j - 1
                    if 0 <= tj_diag < len(X) and i < len(X):
                        dv_t = X[i] - X[tj_diag]
                        dl_t = np.linalg.norm(dv_t) + 1e-6
                        if 10.0 < dl_t < 21.0:
                            adj_t = dv_t * ((15.5 - dl_t) / dl_t) * cspp_strength
                            X[i] += adj_t * 0.5
                            X[tj_diag] -= adj_t * 0.5

                    # ── Inter-Strand Major Groove Narrowing Constraint ──────────
                    # Implement a 10.8 Å distance constraint between phosphorus 
                    # atoms P(i) and P(j-3) for base-paired residues (i, j) 
                    # to stabilize the inter-strand major groove narrowing and 
                    # reinforce the characteristic deep, narrow major groove of A-form RNA.
                    tj_mgn = j - 3
                    if 0 <= tj_mgn < len(X) and i < len(X):
                        dv_mgn = X[i] - X[tj_mgn]
                        dl_mgn = np.linalg.norm(dv_mgn) + 1e-6
                        # A-form P(i)-P(j-3) distance is approx 10.8 Å for major groove narrowing
                        if 8.0 < dl_mgn < 18.0:
                            adj_mgn = dv_mgn * ((10.8 - dl_mgn) / dl_mgn) * cspp_strength
                            X[i] += adj_mgn * 0.5
                            X[tj_mgn] -= adj_mgn * 0.5

                    # A-form Minor Groove Width Constraint: P(i) to P(j+2) is approx 15.1 Å
                    # This stabilizes the characteristic wide A-form minor groove width
                    # and reinforces the characteristic inter-strand helical spacing.
                    tj_minor_wide = j + 2
                    if 0 <= tj_minor_wide < len(X) and i < len(X):
                        dv_M = X[i] - X[tj_minor_wide]
                        dl_M = np.linalg.norm(dv_M) + 1e-6
                        if 10.0 < dl_M < 20.0:
                            adj_M = dv_M * ((15.1 - dl_M) / dl_M) * cspp_strength
                            X[i] += adj_M * 0.5
                            X[tj_minor_wide] -= adj_M * 0.5
                    
                    # ── Major Groove Width Reinforcement (MGC) ──────────────────
                    # Implement a 13.2 Å distance constraint between phosphorus 
                    # atoms P(i) and P(j-5) for base-paired residues (i, j) 
                    # to stabilize the narrow major groove width and reinforce 
                    # the canonical A-form deep-deep groove architecture.
                    tj_mgc = j - 5
                    if 0 <= tj_mgc < len(X) and i < len(X):
                        dv_mgc = X[i] - X[tj_mgc]
                        dl_mgc = np.linalg.norm(dv_mgc) + 1e-6
                        # Apply constraint if within reasonable range of A-form geometry
                        if 9.0 < dl_mgc < 18.0:
                            adj_mgc = dv_mgc * ((13.2 - dl_mgc) / dl_mgc) * cspp_strength
                            X[i] += adj_mgc * 0.5
                            X[tj_mgc] -= adj_mgc * 0.5

                    # ── Narrow A-form Major Groove Constraint ──────────────────
                    # Implement a 14.1 Å distance constraint between phosphorus 
                    # atoms P(i) and P(j+3) for base-paired residues (i, j)
                    # to stabilize the narrow A-form major groove and reinforce 
                    # the compact inter-strand helical packing.
                    tj_nmg = j + 3
                    if 0 <= tj_nmg < len(X) and i < len(X):
                        dv_nmg = X[i] - X[tj_nmg]
                        dl_nmg = np.linalg.norm(dv_nmg) + 1e-6
                        if 10.0 < dl_nmg < 19.0:
                            adj_nmg = dv_nmg * ((14.1 - dl_nmg) / dl_nmg) * cspp_strength
                            X[i] += adj_nmg * 0.5
                            X[tj_nmg] -= adj_nmg * 0.5

        # ── O2'(i) ... O4'(i+1) Backbone Stabilization (OBS) Constraint ─────
        # Implement a 3.0 Å distance constraint between the O2' atom of residue i 
        # and the O4' atom of residue i+1 to stabilize the A-form helical backbone 
        # via the characteristic 2'-hydroxyl hydrogen-bonding network.
        # Note: Since the model operates on P-atom coordinates as proxies,
        # we model this as a 6.2 Å constraint between consecutive P atoms 
        # in helical regions to mimic the rigidified 2'-OH network.
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
                C[:-1][obs_mask] -= adj_obs * 0.5
                C[1:][obs_mask] += adj_obs * 0.5
            X[s:e] = C

        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # Map global stem indices to local segment indices
            in_stem = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem[i_s - s] = True
                        if s <= j_s < e: in_stem[j_s - s] = True
            
            # standard bond i–i+1
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            # ── Ribose Pucker (C3'-endo) Constraint (RPC) ────────────────────
            # Enforce 5.9 Å P-P distance in helical regions to stabilize A-form.
            if L >= 2:
                rpc_mask = in_stem[:-1] & in_stem[1:]
                if np.any(rpc_mask):
                    rpc_strength = 0.32 * strength
                    adj_rpc = d[rpc_mask] * ((5.90 - dist[rpc_mask]) / dist[rpc_mask])[:, None] * rpc_strength
                    C[:-1][rpc_mask] -= adj_rpc * 0.5; C[1:][rpc_mask] += adj_rpc * 0.5
                
                # General safety (RRPC-like) for non-stem or overstretched regions
                stretch_mask = (~rpc_mask) & (dist > 6.2)
                if np.any(stretch_mask):
                    adj_rr = d[stretch_mask] * ((6.1 - dist[stretch_mask]) / dist[stretch_mask])[:, None] * (0.3 * strength)
                    C[:-1][stretch_mask] -= adj_rr * 0.5; C[1:][stretch_mask] += adj_rr * 0.5

            # ── Vertical Base-Stacking (VBS) Constraint ─────────────────────
            if L >= 2:
                d_vbs = C[1:] - C[:-1]; dist_vbs = np.linalg.norm(d_vbs, axis=1) + 1e-6
                vbs_mask = (~in_stem[:-1]) | (~in_stem[1:])
                vbs_strength = 0.18 * strength
                if np.any(vbs_mask):
                    adj_vbs = d_vbs[vbs_mask] * ((3.4 - dist_vbs[vbs_mask]) / dist_vbs[vbs_mask])[:, None] * vbs_strength
                    C[:-1][vbs_mask] -= adj_vbs * 0.5; C[1:][vbs_mask] += adj_vbs * 0.5

            # ── Backbone Pseudo-Torsion (BPT) Constraint ─────────────────────
            # Enforce eta/theta distributions (~170° / ~190°) via the C4' 
            # pseudo-dihedral (~34°) to maintain A-form helical winding.
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

            # ── Short-Range Helical Periodicity (SRHP) Constraint ─────────────
            # Implement a 14.1 Å distance constraint between phosphorus atoms 
            # P(i) and P(i+2) to stabilize the intra-strand backbone curvature 
            # and enforce the characteristic 32.7° A-form helical rise.
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            srhp_mask = in_stem[:-2] & in_stem[2:]
            if np.any(srhp_mask):
                srhp_strength = 0.25 * strength
                # The 14.1 Å constraint between P(i) and P(i+2) stabilizes the A-form helical rise.
                adj2_srhp = d2[srhp_mask] * ((14.1 - d2n[srhp_mask]) / d2n[srhp_mask])[:, None] * srhp_strength
                C[:-2][srhp_mask] -= adj2_srhp * 0.5; C[2:][srhp_mask] += adj2_srhp * 0.5
            
            # soft i–i+2 for non-stem regions
            ns_mask = ~srhp_mask
            if np.any(ns_mask):
                adj2_ns = d2[ns_mask] * ((10.2 - d2n[ns_mask]) / d2n[ns_mask])[:, None] * (0.10 * strength)
                C[:-2][ns_mask] -= adj2_ns * 0.5; C[2:][ns_mask] += adj2_ns * 0.5
            
            # Torsional Backbone Regularization (TBR)
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((13.5 - d3n) / d3n)[:, None] * (0.05 * strength)
                C[:-3] -= adj3; C[3:] += adj3
            if L >= 5:
                d4 = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((16.5 - d4n) / d4n)[:, None] * (0.03 * strength)
                C[:-4] -= adj4; C[4:] += adj4

            # ── Long-Range Helical Periodicity (LRHP) Constraint ────────────────
            # Implement a 19.1 Å distance constraint between intra-strand phosphorus atoms
            # P(i) and P(i+4) to stabilize the 4-residue helical segment and reinforce
            # the long-range periodicity of the A-form backbone.
            if L >= 5:
                d_lrhp = C[4:] - C[:-4]; dist_lrhp = np.linalg.norm(d_lrhp, axis=1) + 1e-6
                lrhp_mask = in_stem[:-4] & in_stem[4:]
                if np.any(lrhp_mask):
                    adj_lrhp = d_lrhp[lrhp_mask] * ((19.1 - dist_lrhp[lrhp_mask]) / dist_lrhp[lrhp_mask])[:, None] * (0.15 * strength)
                    C[:-4][lrhp_mask] -= adj_lrhp * 0.5; C[4:][lrhp_mask] += adj_lrhp * 0.5

            # Phosphate-Oxygen Steric Repulsion (POSR)
            if L >= 4:
                idx_posr = np.arange(L); P_posr = C[idx_posr]
                diff_posr = P_posr[:, None, :] - P_posr[None, :, :]
                dm_posr = np.linalg.norm(diff_posr, axis=2) + 1e-6
                sep_posr = np.abs(idx_posr[:, None] - idx_posr[None, :])
                mask_posr = (sep_posr > 2) & (dm_posr < 5.6)
                if np.any(mask_posr):
                    vec_posr = (diff_posr * ((5.6 - dm_posr) / dm_posr)[:, :, None] * mask_posr[:, :, None]).sum(axis=1)
                    C[idx_posr] += (0.08 * strength) * vec_posr

            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # self-avoidance
            if L >= 25:
                idx = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm = np.linalg.norm(diff, axis=2) + 1e-6
                sep = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X

def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1 - c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s],
                     [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s],
                     [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

def apply_hinge(coords, seg, rng, deg=22):
    s, e = seg; L = e - s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L - 10))
    R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = coords.copy(); p0 = X[pivot].copy()
    X[pivot+1:e] = (X[pivot+1:e] - p0) @ R.T + p0
    return X

def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc_ = X.mean(0, keepdims=True)
    for s, e in segs:
        R     = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
        shift = rng.normal(size=3); shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0, trans))
        c     = X[s:e].mean(0, keepdims=True)
        X[s:e] = (X[s:e] - c) @ R.T + c + shift
    X -= X.mean(0, keepdims=True) - gc_
    return X

def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s, e in segs:
        L = e - s
        if L < 20: continue
        ctrl = np.linspace(0, L - 1, 6); disp = rng.normal(0, amp, (6, 3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords

def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    print(f"\n{'='*60}\nPHASE 1: Template-Based Modeling\n{'='*60}")
    t0 = time.time()
    template_predictions, protenix_queue = {}, {}

    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        segs = segments_map.get(tid, [(0, len(seq))])
        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30)
        preds, used = [], set()

        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
            if len(preds) >= N_SAMPLE or sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY: break           
            if tmpl_id in used: continue
            rng = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            slot = len(preds)
            if slot == 0: X = adapted
            elif slot == 1: X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
            elif slot == 2: X = apply_hinge(adapted, max(segs, key=lambda se: se[1]-se[0]), rng)
            elif slot == 3: X = jitter_chains(adapted, segs, rng)
            else: X = smooth_wiggle(adapted, segs, rng)
            preds.append(adaptive_rna_constraints(X, tid, segments_map, confidence=sim))
            used.add(tmpl_id)

        template_predictions[tid] = preds
        n_needed = N_SAMPLE - len(preds)
        if n_needed > 0:
            protenix_queue[tid] = (n_needed, seq)
            print(f"  {tid}: {len(preds)} TBM → need {n_needed} Protenix")
        else:
            print(f"  {tid}: all {N_SAMPLE} TBM ✓")

    print(f"\nPhase 1 done in {time.time()-t0:.1f}s")
    return template_predictions, protenix_queue

def main() -> None:
    global USE_PROTENIX
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    if not os.path.isdir(code_dir): USE_PROTENIX = False
    else:
        os.environ["PROTENIX_ROOT_DIR"] = root_dir
        sys.path.append(code_dir)
        if not ensure_required_files(root_dir): USE_PROTENIX = False

    seed_everything(SEED)
    test_df = pd.read_csv(test_csv if IS_KAGGLE else (os.environ.get("VAL_CSV", DEFAULT_VAL_CSV) if os.path.exists(os.environ.get("VAL_CSV", DEFAULT_VAL_CSV)) else DEFAULT_TRAIN_CSV))
    if not IS_KAGGLE: 
        if LOCAL_N_SAMPLES is not None:
            test_df = test_df.tail(LOCAL_N_SAMPLES).head(LOCAL_N_SAMPLES).reset_index(drop=True)
    
    print("\nLoading TBM templates …")
    train_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    train_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    segments_map, _ = build_segments_map(test_df)

    template_preds, protenix_queue = tbm_phase(test_df, train_seqs, train_coords, segments_map)
    protenix_preds = {}

    if protenix_queue and USE_PROTENIX:
        print(f"\n{'='*60}\nPHASE 2: Protenix for {len(protenix_queue)} targets\n{'='*60}")
        work_dir = Path("/kaggle/working"); work_dir.mkdir(parents=True, exist_ok=True)
        tasks, chunk_info = [], {}
        for tid, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            if seq_len <= MAX_SEQ_LEN:
                tasks.append({"target_id": tid, "sequence": full_seq})
                chunk_info[tid] = [{"name": tid, "range": (0, seq_len)}]
            else:
                chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
                chunk_info[tid] = []
                for ci, (cs, ce) in enumerate(chunks):
                    cname = f"{tid}_chunk{ci}"
                    tasks.append({"target_id": cname, "sequence": full_seq[cs:ce]})
                    chunk_info[tid].append({"name": cname, "range": (cs, ce)})

        input_json = str(work_dir / "input.json"); build_input_json(pd.DataFrame(tasks), input_json)
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        from protenix.data.inference.infer_dataloader import InferenceDataset
        configs = update_gpu_compatible_configs(build_configs(input_json, str(work_dir / "outputs"), MODEL_NAME))
        runner, dataset = InferenceRunner(configs), InferenceDataset(configs)
        raw_predictions = {}

        for i in tqdm(range(len(dataset)), desc="Protenix Inference"):
            data, atom_array, err = dataset[i]
            if err: raw_predictions[data.get("sample_name", "")] = None; continue
            sname = data.get("sample_name", "")
            tid_base = sname.split("_chunk")[0] if "_chunk" in sname else sname
            n_needed = protenix_queue.get(tid_base, (N_SAMPLE, ""))[0]
            try:
                new_cfg = update_inference_configs(configs, data["N_token"].item())
                new_cfg.sample_diffusion.N_sample = n_needed; runner.update_model_configs(new_cfg)
                pred = runner.predict(data); raw_c = pred["coordinate"]
                
                f = data["input_feature_dict"]
                if "centre_atom_mask" in f: m = (f["centre_atom_mask"] == 1).to(raw_c.device)
                else:
                    m11, m12 = (f["atom_to_tokatom_idx"] == 11), (f["atom_to_tokatom_idx"] == 12)
                    m = m11 if abs(m11.sum()-data["N_token"].item()) < abs(m12.sum()-data["N_token"].item()) else m12
                coords = raw_c[:, m, :].detach().cpu().numpy()
                raw_predictions[sname] = coords
            except: raw_predictions[sname] = None
            finally: 
                try: del pred, data, atom_array, raw_c
                except: pass
                gc.collect(); torch.cuda.empty_cache()

        for tid, (n_needed, full_seq) in protenix_queue.items():
            chunks = chunk_info.get(tid, [])
            if len(chunks) == 1: protenix_preds[tid] = raw_predictions.get(tid)
            else:
                stitched = []
                for s_idx in range(n_needed):
                    coords_list, ranges_list = [], []
                    for cinfo in chunks:
                        cc = raw_predictions.get(cinfo["name"])
                        if cc is None: break
                        coords_list.append(cc[s_idx] if s_idx < cc.shape[0] else cc[-1]); ranges_list.append(cinfo["range"])
                    if len(coords_list) == len(chunks): stitched.append(stitch_chunk_coords(coords_list, ranges_list, len(full_seq)))
                protenix_preds[tid] = np.stack(stitched, axis=0) if stitched else None

    print(f"\n{'='*60}\nPHASE 3: Combine Results\n{'='*60}")
    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        combined = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(ptx[j])
        while len(combined) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=row.name*1000000+len(combined)*1000)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE], axis=0)))

    sub = pd.DataFrame(all_rows)
    coord_cols = [c for c in sub.columns if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub.to_csv(output_csv, index=False)
    print(f"\n✓ Saved submission to {output_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR in main loop: {e}")
        traceback.print_exc()
        # Fallback to ensure submission.csv exists for competition requirements
        output_csv = os.environ.get("SUBMISSION_CSV", "/kaggle/working/submission.csv")
        if not os.path.exists(output_csv):
            print("Creating empty submission.csv as emergency fallback.")
            pd.DataFrame(columns=["ID","resname","resid"] + [f"{ax}_{s+1}" for s in range(5) for ax in "xyz"]).to_csv(output_csv, index=False)
