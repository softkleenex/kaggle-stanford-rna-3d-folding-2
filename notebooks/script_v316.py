import os
import sys
import subprocess
import traceback
import gc
import json
import time
from pathlib import Path

# ── Install dependencies offline ──────────────────────────────────────────────
# Strategy: Implement a 15.2 Å distance constraint between C5' atoms for 
# base-paired residues (i, j) to stabilize the backbone orientation and 
# reinforce the characteristic A-form helical diameter.
# Implementation: In our P-atom proxy model, the 15.2 Å C5'-C5' distance 
# corresponds to a ~17.5 Å P-P cross-strand distance in A-form helices.

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
    else:
        print(f"Warning: Wheel not found {w}, skipping installation.")

import pandas as pd
import numpy as np
import torch
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
def find_input_file(pattern):
    for p in Path('/kaggle/input').rglob(pattern):
        return str(p)
    return None

DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = find_input_file('test_sequences.csv') or f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = find_input_file('train_sequences.csv') or f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = find_input_file('train_labels.csv') or f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = find_input_file('validation_sequences.csv') or f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = find_input_file('validation_labels.csv') or f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

# Dynamic resolution for Protenix code
DEFAULT_CODE_DIR = next(Path('/kaggle/input').rglob('Protenix-v1'), None)
if DEFAULT_CODE_DIR:
    DEFAULT_CODE_DIR = str(DEFAULT_CODE_DIR)
else:
    DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"

DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

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
    full = adaptive_rna_constraints(full, "stitched", {}, confidence=0.5, passes=2)

    # ── Chain-Continuity Elastic (CCE) Constraint ───────────────────────────
    for i in range(seq_len - 1):
        diff = full[i+1] - full[i]
        dist = np.linalg.norm(diff) + 1e-12
        if dist > 1.61:
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
    return al

_aligner = _make_aligner()

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    pairs = []
    dist_threshold = 12.5
    for i in range(L):
        for j in range(i + 4, L):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < dist_threshold:
                pairs.append((i, j))
    
    stems = []
    if not pairs: return stems
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
        if len(stem) >= 3: stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    stems    = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── C5'-C5' Distance Constraint (C5C5) ──────────────────────────────
        # Implement a 15.2 Å distance constraint between C5' atoms for 
        # base-paired residues (i, j) to stabilize the backbone orientation.
        # In our P-atom proxy model, the 15.2 Å C5'-C5' distance 
        # corresponds to a ~17.5 Å P-P cross-strand distance in A-form.
        if stems:
            c5c5_strength = 0.30 * strength
            for stem in stems:
                for i_p, j_p in stem:
                    # In A-form, P(i) to P(j) is roughly 17.5-18.0 Å
                    dv_c5 = X[i_p] - X[j_p]
                    dl_c5 = np.linalg.norm(dv_c5) + 1e-6
                    if 14.0 < dl_c5 < 21.0:
                        adj_c5 = dv_c5 * ((17.5 - dl_c5) / dl_c5) * c5c5_strength
                        X[i_p] += adj_c5 * 0.5
                        X[j_p] -= adj_c5 * 0.5

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
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
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
                    for (ai, aj) in [stem_a[0], stem_a[-1]]:
                        for (bi, bj) in [stem_b[0], stem_b[-1]]:
                            if min(abs(ai - bi), abs(ai - bj), abs(aj - bi), abs(aj - bj)) <= 5: 
                                center_a = (X[ai] + X[aj]) * 0.5; center_b = (X[bi] + X[bj]) * 0.5
                                d_centers = np.linalg.norm(center_a - center_b) + 1e-6
                                if d_centers < 15.0:
                                    adj_c = (center_a - center_b) * ((10.0 - d_centers) / d_centers) * chs_strength
                                    X[ai] += adj_c; X[aj] += adj_c; X[bi] -= adj_c; X[bj] -= adj_c

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
                for i_p, j_p in stem:
                    tj_minor = j_p - 3
                    if 0 <= tj_minor < len(X):
                        dv_m = X[i_p] - X[tj_minor]; dl_m = np.linalg.norm(dv_m) + 1e-6
                        if 11.0 < dl_m < 21.0:
                            adj_m = dv_m * ((15.5 - dl_m) / dl_m) * cspp_strength
                            X[i_p] += adj_m * 0.5; X[tj_minor] -= adj_m * 0.5

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
            
            # ── Ribose Pucker (C3'-endo) Constraint (RPC) ────────────────────
            rpc_mask = in_stem[:-1] & in_stem[1:]
            if np.any(rpc_mask):
                rpc_strength = 0.32 * strength
                adj_rpc = d[rpc_mask] * ((5.90 - dist[rpc_mask]) / dist[rpc_mask])[:, None] * rpc_strength
                C[:-1][rpc_mask] -= adj_rpc * 0.5; C[1:][rpc_mask] += adj_rpc * 0.5

            # ── Backbone Pseudo-Torsion (BPT) Constraint ─────────────────────
            if L >= 4:
                bpt_strength = 0.14 * strength
                p1, p2, p3, p4 = C[:-3], C[1:-2], C[2:-1], C[3:]
                v12, v23, v34 = p2 - p1, p3 - p2, p4 - p3
                n1 = np.cross(v12, v23); n2 = np.cross(v23, v34)
                n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-9)
                n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9)
                phi = np.arccos(np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0))
                phi = np.where(np.sum(v12 * n2, axis=1) < 0, -phi, phi)
                axis = v23 / (np.linalg.norm(v23, axis=1, keepdims=True) + 1e-9)
                force = np.cross(n2, axis) * (0.61 - phi)[:, None] * bpt_strength
                C[3:] += force * 0.5; C[:-3] -= force * 0.5

            # self-avoidance
            if L >= 25:
                idx = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm = np.linalg.norm(diff, axis=2) + 1e-6
                mask = (np.abs(idx[:, None] - idx[None, :]) > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X

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

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict or abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        norm_s = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
        pct_id = 100 * identical / len(query_seq)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6; coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords

def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    seed_everything(SEED)
    test_df = pd.read_csv(test_csv)
    if not IS_KAGGLE and LOCAL_N_SAMPLES: test_df = test_df.head(LOCAL_N_SAMPLES)
    
    print("\nLoading TBM templates …")
    train_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    train_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    segments_map = {r["target_id"]: [(0, len(r["sequence"]))] for _, r in test_df.iterrows()}

    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences_detailed(seq, train_seqs, train_coords, top_n=30)
        preds = []
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id) in enumerate(similar):
            if len(preds) >= N_SAMPLE or pct_id < MIN_PERCENT_IDENTITY: break
            X = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            preds.append(adaptive_rna_constraints(X, tid, segments_map, confidence=sim))
        
        while len(preds) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=row.name*1000+len(preds))
            preds.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        
        all_rows.extend(coords_to_rows(tid, seq, np.stack(preds[:N_SAMPLE], axis=0)))

    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    print(f"\n✓ Saved submission to {output_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        pd.DataFrame(columns=["ID","resname","resid"] + [f"{ax}_{s+1}" for s in range(5) for ax in "xyz"]).to_csv(DEFAULT_OUTPUT, index=False)
