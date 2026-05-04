import os
import sys
import subprocess
import traceback

# ── Install dependencies offline ──────────────────────────────────────────────
# Strategy: "Implement a 11.4 Å distance constraint between C1' atoms of 
# residues i and j-1 for base-paired residues (i, j) to stabilize the 
# cross-strand diagonal and reinforce the characteristic 32.7° A-form helical twist."
# Implementation: Version 315 adds the Cross-Strand Diagonal C1' (CSDC) 
# constraint to the adaptive_rna_constraints function.

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
from pathlib import Path

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
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
# Use dynamic path resolution as per rules
def find_dataset_path(pattern):
    for p in Path('/kaggle/input').rglob(pattern):
        return str(p.parent)
    return "/kaggle/input/stanford-rna-3d-folding-2"

DATA_BASE              = find_dataset_path('test_sequences.csv')
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

# Find Protenix code
DEFAULT_CODE_DIR = next(iter(Path('/kaggle/input').rglob('protenix-v1-adjusted')), None)
if DEFAULT_CODE_DIR:
    # Adjust to find the actual code root based on typical structure
    potential_roots = list(DEFAULT_CODE_DIR.rglob('protenix'))
    if potential_roots:
        DEFAULT_CODE_DIR = str(potential_roots[0].parent)
    else:
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
        (next(iter(Path(root_dir).rglob(f"{MODEL_NAME}.pt")), Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt"), "checkpoint"),
        (next(iter(Path(root_dir).rglob("components.cif")), Path(root_dir) / "common" / "components.cif"), "CCD file"),
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
    centroid_P = P.mean(axis=0); centroid_Q = Q.mean(axis=0)
    Pc = P - centroid_P; Qc = Q - centroid_Q
    H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T; t = centroid_Q - R @ centroid_P
    return R, t

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len: return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype); out[:coords.shape[0]] = coords
        return out

    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]
        ov_start = cur_start; ov_end = min(prev_end, cur_end); ov_len = ov_end - ov_start
        if ov_len < 3: aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
        cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]
        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3: aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)

    full = np.zeros((seq_len, 3), dtype=np.float64); weights = np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        actual_end = min(s + coords.shape[0], seq_len); used_len = actual_end - s
        w = np.ones(used_len, dtype=np.float64)
        if i > 0:
            ov_len = min(chunk_ranges[i - 1][1], e) - s
            if ov_len > 0: w[:ov_len] = np.linspace(0.0, 1.0, ov_len)
        if i < len(chunk_ranges) - 1:
            ramp_len = actual_end - chunk_ranges[i + 1][0]
            if ramp_len > 0: w[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)
        full[s:actual_end] += coords[:used_len] * w[:, None]; weights[s:actual_end] += w

    mask = weights > 0; full[mask] /= weights[mask, None]
    full = adaptive_rna_constraints(full, "stitched", {}, confidence=0.5, passes=2)
    for i in range(seq_len - 1):
        diff = full[i+1] - full[i]; dist = np.linalg.norm(diff) + 1e-12
        if dist > 1.61: adj = diff * ((1.61 - dist) / dist) * 0.5; full[i] -= adj; full[i+1] += adj
    return full

def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner(); al.mode = "global"; al.match_score = 2; al.mismatch_score = -1.5
    al.open_gap_score = -8; al.extend_gap_score = -0.4; al.query_left_open_gap_score = -8
    al.query_left_extend_gap_score = -0.4; al.query_right_open_gap_score = -8
    al.query_right_extend_gap_score = -0.4; al.target_left_open_gap_score = -8
    al.target_left_extend_gap_score = -0.4; al.target_right_open_gap_score = -8
    al.target_right_extend_gap_score = -0.4
    return al

_aligner = _make_aligner()

def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or str(stoich).strip() == "": return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";") for ch, cnt in [part.split(":")]]

def parse_fasta(fasta_content: str) -> dict:
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if cur is not None: out[cur] = "".join(parts)
            cur = line[1:].split()[0]; parts = []
        else: parts.append(line.replace(" ", ""))
    if cur is not None: out[cur] = "".join(parts)
    return out

def get_chain_segments(row) -> list:
    seq, stoich, all_sq = row["sequence"], row.get("stoichiometry", ""), row.get("all_sequences", "")
    if (pd.isna(stoich) or pd.isna(all_sq) or str(stoich).strip() == "" or str(all_sq).strip() == ""): return [(0, len(seq))]
    try:
        chain_dict, order = parse_fasta(all_sq), parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None: return [(0, len(seq))]
            for _ in range(cnt): segs.append((pos, pos + len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except Exception: return [(0, len(seq))]

def build_segments_map(df: pd.DataFrame) -> tuple:
    seg_map, stoich_map = {}, {}
    for _, r in df.iterrows():
        tid = r["target_id"]; seg_map[tid] = get_chain_segments(r)
        raw_s = r.get("stoichiometry", ""); stoich_map[tid] = "" if pd.isna(raw_s) else str(raw_s)
    return seg_map, stoich_map

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}; prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segs, t_segs = alignment.aligned; aq, at, qi, ti = [], [], 0, 0
    for (qs, qe), (ts, te) in zip(q_segs, t_segs):
        while qi < qs: aq.append(query_seq[qi]); at.append("-"); qi += 1
        while ti < ts: aq.append("-"); at.append(template_seq[ti]); ti += 1
        for qp, tp in zip(range(qs, qe), range(ts, te)): aq.append(query_seq[qp]); at.append(template_seq[tp])
        qi, ti = qe, te
    while qi < len(query_seq): aq.append(query_seq[qi]); at.append("-"); qi += 1
    while ti < len(template_seq): aq.append("-"); at.append(template_seq[ti]); ti += 1
    return "".join(aq), "".join(at)

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict or abs(len(tseq)-len(query_seq))/max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        norm_s = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
        pct_id = 100 * identical / len(query_seq); aq, at = _build_aligned_strings(query_seq, tseq, aln)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln = next(iter(_aligner.align(query_seq, template_seq))); new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs): new_coords[qs:qe] = chunk
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i+1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0: w = (i-pv)/(nv-pv); new_coords[i] = (1-w)*new_coords[pv] + w*new_coords[nv]
            elif pv >= 0: new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0: new_coords[i] = new_coords[nv] + [3, 0, 0]
            else: new_coords[i] = [i*3, 0, 0]
    return np.nan_to_num(new_coords)

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    pairs, dist_threshold = [], 12.5
    for i in range(L):
        for j in range(i + 4, L):
            if np.linalg.norm(coords[i] - coords[j]) < dist_threshold: pairs.append((i, j))
    if not pairs: return []
    pairs.sort(); stems, used = [], [False] * len(pairs)
    for i in range(len(pairs)):
        if used[i]: continue
        stem, used[i], curr_i, curr_j = [pairs[i]], True, *pairs[i]
        for k in range(i+1, len(pairs)):
            if not used[k] and pairs[k][0] == curr_i+1 and pairs[k][1] == curr_j-1:
                stem.append(pairs[k]); used[k] = True; curr_i, curr_j = pairs[k]
        if len(stem) >= 3: stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X = coords.copy(); segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02); stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── Cross-Strand Diagonal C1' (CSDC) Constraint ─────────────────────
        # Implement a 11.4 Å distance constraint between C1' atoms of residues 
        # i and j-1 for base-paired residues (i, j) to stabilize the cross-strand 
        # diagonal and reinforce the characteristic 32.7° A-form helical twist.
        # Since X stores P-atom positions as proxies, we model this as a 
        # 11.4 Å constraint between P(i) and P(j-1) for paired (i, j).
        if stems:
            csdc_strength = 0.26 * strength
            for stem in stems:
                for i, j in stem:
                    # Constraint between residue i and residue j-1
                    idx_j_minus_1 = j - 1
                    if 0 <= idx_j_minus_1 < len(X):
                        dv_csdc = X[i] - X[idx_j_minus_1]
                        dl_csdc = np.linalg.norm(dv_csdc) + 1e-6
                        if 8.5 < dl_csdc < 14.5:
                            adj_csdc = dv_csdc * ((11.4 - dl_csdc) / dl_csdc) * csdc_strength
                            X[i] += adj_csdc * 0.5
                            X[idx_j_minus_1] -= adj_csdc * 0.5

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
                C[:-1][cpp_mask] -= adj_cpp * 0.5; C[1:][cpp_mask] += adj_cpp * 0.5
            X[s:e] = C

        if stems:
            bpp_strength = 0.15 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
                        v1_i, v2_i = X[i_s+1] - X[i_s], X[i_s-1] - X[i_s]; n_i = np.cross(v1_i, v2_i); n_i /= (np.linalg.norm(n_i)+1e-6)
                        v1_j, v2_j = X[j_s+1] - X[j_s], X[j_s-1] - X[j_s]; n_j = np.cross(v1_j, v2_j); n_j /= (np.linalg.norm(n_j)+1e-6)
                        dot = np.dot(n_i, n_j)
                        if abs(dot) < 0.98:
                            target_n = n_i if dot > 0 else -n_i; adj_bpp = (target_n - n_j) * bpp_strength
                            X[j_s+1] += adj_bpp; X[j_s-1] += adj_bpp; X[i_s+1] -= adj_bpp; X[i_s-1] -= adj_bpp

        if len(stems) >= 2:
            chs_strength = 0.12 * strength
            for i in range(len(stems)):
                for j in range(i+1, len(stems)):
                    for (ai, aj) in [stems[i][0], stems[i][-1]]:
                        for (bi, bj) in [stems[j][0], stems[j][-1]]:
                            if min(abs(ai-bi), abs(ai-bj), abs(aj-bi), abs(aj-bj)) <= 5:
                                vec_a, vec_b = X[ai]-X[aj], X[bi]-X[bj]
                                d_a, d_b = np.linalg.norm(vec_a)+1e-6, np.linalg.norm(vec_b)+1e-6
                                center_a, center_b = (X[ai]+X[aj])*0.5, (X[bi]+X[bj])*0.5
                                d_centers = np.linalg.norm(center_a - center_b) + 1e-6
                                if d_centers < 15.0:
                                    adj_c = (center_a - center_b) * ((10.0-d_centers)/d_centers) * chs_strength
                                    X[ai]+=adj_c; X[aj]+=adj_c; X[bi]-=adj_c; X[bj]-=adj_c
                                if np.dot(vec_a, vec_b)/(d_a*d_b) < 0.95:
                                    u = (vec_a/d_a+vec_b/d_b)*0.5; u /= np.linalg.norm(u)+1e-6
                                    X[ai]+= (u*d_a*0.5+center_a-X[ai])*chs_strength; X[aj]+= (-u*d_a*0.5+center_a-X[aj])*chs_strength
                                    X[bi]+= (u*d_b*0.5+center_b-X[bi])*chs_strength; X[bj]+= (-u*d_b*0.5+center_b-X[bj])*chs_strength

        if len(stems) >= 2:
            csc_strength = 0.20 * strength
            for i in range(len(stems)):
                for j in range(i+1, len(stems)):
                    for (ai, aj) in [stems[i][0], stems[i][-1]]:
                        for (bi, bj) in [stems[j][0], stems[j][-1]]:
                            if min(abs(ai-bi), abs(ai-bj), abs(aj-bi), abs(aj-bj)) <= 3:
                                axis_a, axis_b = X[ai]-X[aj], X[bi]-X[bj]
                                d_b, u_a = np.linalg.norm(axis_b)+1e-6, (X[ai]-X[aj])/(np.linalg.norm(X[ai]-X[aj])+1e-6)
                                cos_v = np.dot(u_a, axis_b/d_b)
                                if abs(cos_v) < 0.95:
                                    target_u = u_a if cos_v > 0 else -u_a
                                    adj = (target_u - axis_b/d_b)*csc_strength
                                    X[bi]+=adj*d_b*0.5; X[bj]-=adj*d_b*0.5

        if stems:
            vbpr_strength = 0.22 * strength
            for stem in stems:
                for k in range(len(stem)-1):
                    i1, j1 = stem[k]; i2, j2 = stem[k+1]; c1, c2 = (X[i1]+X[j1])*0.5, (X[i2]+X[j2])*0.5
                    dv = c2-c1; dl = np.linalg.norm(dv)+1e-6
                    if 1.5 < dl < 5.0:
                        adj = dv*((2.8-dl)/dl)*vbpr_strength
                        X[i1]-=adj*0.25; X[j1]-=adj*0.25; X[i2]+=adj*0.25; X[j2]+=adj*0.25

        if stems:
            cspp_strength = 0.25 * strength
            for stem in stems:
                for i, j in stem:
                    for offset, target_d in [(-3, 15.5), (2, 10.2)]:
                        tj = j + offset
                        if 0 <= tj < len(X):
                            dv = X[i]-X[tj]; dl = np.linalg.norm(dv)+1e-6
                            if (offset == -3 and 11.0 < dl < 21.0) or (offset == 2 and 7.5 < dl < 14.5):
                                adj = dv*((target_d-dl)/dl)*cspp_strength
                                X[i]+=adj*0.5; X[tj]-=adj*0.5

        obs_strength = 0.28 * strength
        for s, e in segments:
            C = X[s:e]; L = e-s
            if L < 2: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            obs_mask = in_stem_seg[:-1] & in_stem_seg[1:]
            if np.any(obs_mask):
                d = C[1:]-C[:-1]; dl = np.linalg.norm(d, axis=1)+1e-6
                adj = d[obs_mask]*((6.2-dl[obs_mask])/dl[obs_mask])[:, None]*obs_strength
                C[:-1][obs_mask]-=adj*0.5; C[1:][obs_mask]+=adj*0.5
            X[s:e] = C

        for s, e in segments:
            C, L = X[s:e], e-s
            if L < 3: continue
            in_stem = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem[i_s-s] = True
            
            d = C[1:]-C[:-1]; dist = np.linalg.norm(d, axis=1)+1e-6
            adj = d*((5.95-dist)/dist)[:, None]*(0.22*strength); C[:-1]-=adj; C[1:]+=adj
            
            if L >= 2:
                rpc_mask = in_stem[:-1] & in_stem[1:]; rpc_strength = 0.32*strength
                if np.any(rpc_mask):
                    adj = d[rpc_mask]*((5.90-dist[rpc_mask])/dist[rpc_mask])[:, None]*rpc_strength
                    C[:-1][rpc_mask]-=adj*0.5; C[1:][rpc_mask]+=adj*0.5
                stretch = (~rpc_mask) & (dist > 6.2)
                if np.any(stretch):
                    adj = d[stretch]*((6.1-dist[stretch])/dist[stretch])[:, None]*(0.3*strength)
                    C[:-1][stretch]-=adj*0.5; C[1:][stretch]+=adj*0.5

            if L >= 2:
                vbs_mask = (~in_stem[:-1]) | (~in_stem[1:]); vbs_strength = 0.18*strength
                if np.any(vbs_mask):
                    adj = d[vbs_mask]*((3.4-dist[vbs_mask])/dist[vbs_mask])[:, None]*vbs_strength
                    C[:-1][vbs_mask]-=adj*0.5; C[1:][vbs_mask]+=adj*0.5

            if L >= 4:
                bpt_strength = 0.14*strength; p1, p2, p3, p4 = C[:-3], C[1:-2], C[2:-1], C[3:]
                v12, v23, v34 = p2-p1, p3-p2, p4-p3; n1, n2 = np.cross(v12, v23), np.cross(v23, v34)
                n1 /= (np.linalg.norm(n1, axis=1, keepdims=True)+1e-9); n2 /= (np.linalg.norm(n2, axis=1, keepdims=True)+1e-9)
                phi = np.arccos(np.clip(np.sum(n1*n2, axis=1), -1.0, 1.0))
                phi = np.where(np.sum(v12*n2, axis=1) < 0, -phi, phi)
                axis = v23 / (np.linalg.norm(v23, axis=1, keepdims=True)+1e-9)
                force = np.cross(n2, axis) * (0.61 - phi)[:, None] * bpt_strength
                C[3:] += force * 0.5; C[:-3] -= force * 0.5

            d2 = C[2:]-C[:-2]; d2n = np.linalg.norm(d2, axis=1)+1e-6
            adj2 = d2*((10.2-d2n)/d2n)[:, None]*(0.10*strength); C[:-2]-=adj2; C[2:]+=adj2
            if L >= 4:
                d3 = C[3:]-C[:-3]; d3n = np.linalg.norm(d3, axis=1)+1e-6
                adj3 = d3*((13.5-d3n)/d3n)[:, None]*(0.05*strength); C[:-3]-=adj3; C[3:]+=adj3
            if L >= 5:
                d4 = C[4:]-C[:-4]; d4n = np.linalg.norm(d4, axis=1)+1e-6
                adj4 = d4*((16.5-d4n)/d4n)[:, None]*(0.03*strength); C[:-4]-=adj4; C[4:]+=adj4
            if L >= 4:
                diff_posr = C[:, None, :] - C[None, :, :]; dm = np.linalg.norm(diff_posr, axis=2)+1e-6
                mask = (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) > 2) & (dm < 5.6)
                if np.any(mask): C += (0.08*strength) * (diff_posr * ((5.6-dm)/dm)[:, :, None] * mask[:, :, None]).sum(axis=1)

            C[1:-1] += (0.06*strength) * (0.5*(C[:-2]+C[2:])-C[1:-1])
            if L >= 25:
                idx = np.linspace(0, L-1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P = C[idx]; diff = P[:, None, :] - P[None, :, :]; dm = np.linalg.norm(diff, axis=2)+1e-6
                mask = (np.abs(idx[:, None] - idx[None, :]) > 2) & (dm < 3.2)
                if np.any(mask): C[idx] += (0.015*strength) * (diff * ((3.2-dm)/dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
            X[s:e] = C
    return X

def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)+1e-12; x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1-c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s], [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s], [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

def apply_hinge(coords, seg, rng, deg=22):
    s, e = seg; L = e-s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L-10)); R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = coords.copy(); p0 = X[pivot].copy(); X[pivot+1:e] = (X[pivot+1:e]-p0)@R.T+p0
    return X

def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc_ = X.mean(0, keepdims=True)
    for s, e in segs:
        R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
        shift = rng.normal(size=3); shift = shift/(np.linalg.norm(shift)+1e-12)*float(rng.uniform(0, trans))
        c = X[s:e].mean(0, keepdims=True); X[s:e] = (X[s:e]-c)@R.T+c+shift
    X -= X.mean(0, keepdims=True) - gc_
    return X

def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s, e in segs:
        L = e-s
        if L < 20: continue
        ctrl, disp = np.linspace(0, L-1, 6), rng.normal(0, amp, (6, 3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n): ang = i*0.6; coords[i] = [10.0*np.cos(ang), 10.0*np.sin(ang), i*2.5]
    return coords

def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    print(f"\n{'='*60}\nPHASE 1: Template-Based Modeling\n{'='*60}"); t0 = time.time()
    template_predictions, protenix_queue = {}, {}
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]; segs = segments_map.get(tid, [(0, len(seq))])
        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30)
        preds, used = [], set()
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
            if len(preds) >= N_SAMPLE or sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY: break
            if tmpl_id in used: continue
            rng = np.random.default_rng((row.name*10000000000+i*10007)%(2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            if len(preds) == 0: X = adapted
            elif len(preds) == 1: X = adapted + rng.normal(0, max(0.01, (0.40-sim)*0.06), adapted.shape)
            elif len(preds) == 2: X = apply_hinge(adapted, max(segs, key=lambda se: se[1]-se[0]), rng)
            elif len(preds) == 3: X = jitter_chains(adapted, segs, rng)
            else: X = smooth_wiggle(adapted, segs, rng)
            preds.append(adaptive_rna_constraints(X, tid, segments_map, confidence=sim)); used.add(tmpl_id)
        template_predictions[tid] = preds; n_needed = N_SAMPLE - len(preds)
        if n_needed > 0: protenix_queue[tid] = (n_needed, seq); print(f"  {tid}: {len(preds)} TBM → need {n_needed} Protenix")
        else: print(f"  {tid}: all {N_SAMPLE} TBM ✓")
    print(f"\nPhase 1 done in {time.time()-t0:.1f}s"); return template_predictions, protenix_queue

def main() -> None:
    global USE_PROTENIX
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    if not os.path.isdir(code_dir): USE_PROTENIX = False
    else:
        os.environ["PROTENIX_ROOT_DIR"] = root_dir; sys.path.append(code_dir)
        if not ensure_required_files(root_dir): USE_PROTENIX = False
    seed_everything(SEED); test_df = pd.read_csv(test_csv if IS_KAGGLE else (os.environ.get("VAL_CSV", DEFAULT_VAL_CSV) if os.path.exists(os.environ.get("VAL_CSV", DEFAULT_VAL_CSV)) else DEFAULT_TRAIN_CSV))
    if not IS_KAGGLE and LOCAL_N_SAMPLES: test_df = test_df.tail(LOCAL_N_SAMPLES).reset_index(drop=True)
    print("\nLoading TBM templates …")
    train_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    train_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    segments_map, _ = build_segments_map(test_df)
    template_preds, protenix_queue = tbm_phase(test_df, train_seqs, train_coords, segments_map)
    protenix_preds = {}

    if protenix_queue and USE_PROTENIX:
        print(f"\n{'='*60}\nPHASE 2: Protenix for {len(protenix_queue)} targets\n{'='*60}"); work_dir = Path("/kaggle/working"); tasks, chunk_info = [], {}
        for tid, (n_needed, full_seq) in protenix_queue.items():
            if len(full_seq) <= MAX_SEQ_LEN: tasks.append({"target_id": tid, "sequence": full_seq}); chunk_info[tid] = [{"name": tid, "range": (0, len(full_seq))}]
            else:
                chunks = split_into_chunks(len(full_seq), MAX_SEQ_LEN, CHUNK_OVERLAP); chunk_info[tid] = []
                for ci, (cs, ce) in enumerate(chunks): cname = f"{tid}_chunk{ci}"; tasks.append({"target_id": cname, "sequence": full_seq[cs:ce]}); chunk_info[tid].append({"name": cname, "range": (cs, ce)})
        input_json = str(work_dir / "input.json"); build_input_json(pd.DataFrame(tasks), input_json)
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        from protenix.data.inference.infer_dataloader import InferenceDataset
        configs = update_gpu_compatible_configs(build_configs(input_json, str(work_dir / "outputs"), MODEL_NAME))
        runner, dataset, raw_predictions = InferenceRunner(configs), InferenceDataset(configs), {}
        for i in tqdm(range(len(dataset)), desc="Protenix Inference"):
            data, atom_array, err = dataset[i]
            if err: raw_predictions[data.get("sample_name", "")] = None; continue
            sname = data.get("sample_name", ""); tid_base = sname.split("_chunk")[0] if "_chunk" in sname else sname
            try:
                new_cfg = update_inference_configs(configs, data["N_token"].item()); new_cfg.sample_diffusion.N_sample = protenix_queue.get(tid_base, (N_SAMPLE, ""))[0]; runner.update_model_configs(new_cfg)
                pred = runner.predict(data); raw_c = pred["coordinate"]
                m = (data["input_feature_dict"]["centre_atom_mask"] == 1).to(raw_c.device) if "centre_atom_mask" in data["input_feature_dict"] else (data["input_feature_dict"]["atom_to_tokatom_idx"] == 11 if abs(data["input_feature_dict"]["atom_to_tokatom_idx"] == 11).sum()-data["N_token"].item() < abs(data["input_feature_dict"]["atom_to_tokatom_idx"] == 12).sum()-data["N_token"].item() else (data["input_feature_dict"]["atom_to_tokatom_idx"] == 12)).to(raw_c.device)
                raw_predictions[sname] = raw_c[:, m, :].detach().cpu().numpy()
            except: raw_predictions[sname] = None
            finally: gc.collect(); torch.cuda.empty_cache()
        for tid, (n_needed, full_seq) in protenix_queue.items():
            chunks = chunk_info.get(tid, [])
            if len(chunks) == 1: protenix_preds[tid] = raw_predictions.get(tid)
            else:
                stitched = []
                for s_idx in range(n_needed):
                    coords_list, ranges_list = [], []
                    for cinfo in chunks:
                        cc = raw_predictions.get(cinfo["name"])
                        if cc is not None: coords_list.append(cc[s_idx] if s_idx < cc.shape[0] else cc[-1]); ranges_list.append(cinfo["range"])
                    if len(coords_list) == len(chunks): stitched.append(stitch_chunk_coords(coords_list, ranges_list, len(full_seq)))
                protenix_preds[tid] = np.stack(stitched, axis=0) if stitched else None

    print(f"\n{'='*60}\nPHASE 3: Combine Results\n{'='*60}"); all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]; combined = list(template_preds.get(tid, [])); ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(ptx[j])
        while len(combined) < N_SAMPLE: combined.append(adaptive_rna_constraints(generate_rna_structure(seq, seed=row.name*1000000+len(combined)*1000), tid, segments_map, confidence=0.2))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE], axis=0)))
    sub = pd.DataFrame(all_rows); coord_cols = [c for c in sub.columns if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999); sub.to_csv(output_csv, index=False); print(f"\n✓ Saved submission to {output_csv}")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        traceback.print_exc()
        if not os.path.exists(DEFAULT_OUTPUT): pd.DataFrame(columns=["ID","resname","resid"] + [f"{ax}_{s+1}" for s in range(5) for ax in "xyz"]).to_csv(DEFAULT_OUTPUT, index=False)
