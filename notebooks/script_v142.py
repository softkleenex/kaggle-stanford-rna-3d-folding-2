import os
import sys
import pandas as pd

# Fix Jupyter magic commands to valid Python
os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
# On Kaggle competition rerun, KAGGLE_IS_COMPETITION_RERUN is set to a truthy value.
# When running locally we do NOT exit — instead we cap the test set to a small
# number of samples so the notebook finishes quickly and compute local validation RMSD.

IS_KAGGLE = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
LOCAL_N_SAMPLES = 2

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

from pathlib import Path

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm


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
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "128"))

# TBM quality thresholds — sequences below these get routed to Protenix
MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

# Set False to skip Protenix and use de-novo fallback instead
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


def ensure_required_files(root_dir: str) -> None:
    for p, name in [
        (Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt",          "checkpoint"),
        (Path(root_dir) / "common" / "components.cif",                "CCD file"),
        (Path(root_dir) / "common" / "components.cif.rdkit_mol.pkl",  "CCD cache"),
    ]:
        if not p.exists():
            print(f"Warning: Missing {name}: {p}")


# ─────────────── Protenix Input / Config Helpers ─────────────────────────────
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
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
        
    return (f["atom_to_tokatom_idx"] == 11).bool()


def get_feature_c1_mask(data: dict) -> torch.Tensor:
    f = data["input_feature_dict"]
    if "centre_atom_mask" in f:
        return f["centre_atom_mask"].long() == 1
    return f["atom_to_tokatom_idx"].long() == 12


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


def stitch_chunk_coords(chunk_coords_list: list,
                        chunk_ranges: list,
                        seq_len: int) -> np.ndarray:
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

    return full


# ─────────────── TBM Core Functions ──────────────────────────────────────────
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
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";")
            for ch, cnt in [part.split(":")]]


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
    if (pd.isna(stoich) or pd.isna(all_sq)
            or str(stoich).strip() == "" or str(all_sq).strip() == ""):
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


def get_secondary_structure(seq: str) -> list:
    """Finds contiguous Watson-Crick/Wobble stems (greedy) for coplanarity constraints."""
    n = len(seq)
    pairs = []
    used = [False] * n
    for stem_len in range(16, 2, -1):
        for i in range(n - stem_len):
            for j in range(n - 1, i + stem_len + 3, -1):
                match = True
                for k in range(stem_len):
                    if used[i+k] or used[j-k]:
                        match = False; break
                    p = seq[i+k] + seq[j-k]
                    if p not in ("AU", "UA", "CG", "GC", "GU", "UG"):
                        match = False; break
                if match:
                    for k in range(stem_len):
                        pairs.append((i+k, j-k))
                        used[i+k] = used[j-k] = True
    return pairs


def adaptive_rna_constraints(coords, target_id, sequence, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    
    pairs = get_secondary_structure(sequence)
    res_to_seg = {}
    for s, e in segments:
        for idx in range(s, e):
            res_to_seg[idx] = (s, e)

    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # self-avoidance & NEW Electrostatic Phosphate Repulsion
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                
                # Standard steric self-avoidance (< 3.2 Å)
                mask_steric = (sep > 2) & (dm < 3.2)
                if np.any(mask_steric):
                    vec_steric = (diff * ((3.2 - dm) / dm)[:, :, None] * mask_steric[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec_steric
                    
                # NEW: Electrostatic Phosphate Repulsion (1/r^2 force for 3.2 <= r < 8.0, separated by > 3)
                # Mimics RNA backbone electrostatics to prevent unphysical loop compression.
                mask_elec = (sep > 3) & (dm >= 3.2) & (dm < 8.0)
                if np.any(mask_elec):
                    force_mag = 1.0 / (dm * dm)
                    vec_elec = (diff / dm[:, :, None] * force_mag[:, :, None] * mask_elec[:, :, None]).sum(axis=1)
                    C[idx] += (0.05 * strength) * vec_elec
                    
            # Base-Stacking Parallelism constraint (Normal vector alignment < 15°)
            if L >= 4:
                v1 = C[1:-2] - C[:-3]; v2 = C[2:-1] - C[1:-2]; v3 = C[3:] - C[2:-1]
                n1 = np.cross(v1, v2); n1_norm = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-6; n1 /= n1_norm
                n2 = np.cross(v2, v3); n2_norm = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-6; n2 /= n2_norm
                dot = np.clip((n1 * n2).sum(axis=1), -1.0, 1.0); angle = np.arccos(dot)
                mask_stack = (angle > 0.2618) & (angle < 1.047)
                if np.any(mask_stack):
                    force = (n2 - n1) * (0.08 * strength)
                    C[1:-2][mask_stack] -= force[mask_stack]; C[2:-1][mask_stack] += force[mask_stack]

            X[s:e] = C
            
        # Base-Pair Coplanarity & Helical Twist Constraints
        for i, j in pairs:
            if i in res_to_seg and j in res_to_seg:
                si, ei = res_to_seg[i]; sj, ej = res_to_seg[j]
                if si < i < ei - 1 and sj < j < ej - 1:
                    vi1 = X[i] - X[i-1]; vi2 = X[i+1] - X[i]
                    ni = np.cross(vi1, vi2); ni_m = np.linalg.norm(ni) + 1e-6; ni /= ni_m
                    vj1 = X[j] - X[j-1]; vj2 = X[j+1] - X[j]
                    nj = np.cross(vj1, vj2); nj_m = np.linalg.norm(nj) + 1e-6; nj /= nj_m
                    dot = np.clip(np.dot(ni, -nj), -1.0, 1.0); angle = np.arccos(dot)
                    if angle > 0.1745 and angle < 1.047: 
                        force = (-nj - ni) * (0.12 * strength); X[i] -= force; X[j] += force

                # NEW: Helical Twist Constraint (32.7° ± 3°)
                # Targets canonical A-form periodicity by aligning successive base-pair normals.
                if i+1 < len(X) and j-1 >= 0 and (i+1, j-1) in pairs:
                    if si < i < ei - 1 and sj < j < ej - 1:
                        # Vectors for base-pair (i, j)
                        v_bp1 = X[j] - X[i]; v_bp1 /= np.linalg.norm(v_bp1) + 1e-6
                        # Vectors for next base-pair (i+1, j-1)
                        v_bp2 = X[j-1] - X[i+1]; v_bp2 /= np.linalg.norm(v_bp2) + 1e-6
                        
                        # Current twist angle in local projection
                        # We use the dot product of base-pair vectors to estimate rotation.
                        dot_twist = np.clip(np.dot(v_bp1, v_bp2), -1.0, 1.0)
                        curr_twist = np.arccos(dot_twist)
                        
                        # Ideal A-form twist = 32.7 degrees = 0.5707 radians.
                        ideal_twist = 0.5707
                        # Tolerance = 3 degrees = 0.0524 radians.
                        if abs(curr_twist - ideal_twist) > 0.0524:
                            # Restoration force: push/pull to restore ideal twist
                            # This is a proxy for the angular constraint.
                            diff = (curr_twist - ideal_twist) * (0.15 * strength)
                            # Apply small correction to base-pair vectors
                            # We rotate v_bp2 slightly around the helical axis (estimated as ni)
                            # For simplicity, we use a linear approximation here.
                            X[i+1] += diff * v_bp1; X[j-1] -= diff * v_bp1
                            
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
    """Idealized A-form RNA helix — last-resort de-novo fallback."""
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords


# ─────────────── TBM Phase ───────────────────────────────────────────────────
def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    print(f"\n{'='*60}")
    print(f"PHASE 1: Template-Based Modeling")
    print(f"  MIN_SIMILARITY = {MIN_SIMILARITY}  |  MIN_PCT_IDENTITY = {MIN_PERCENT_IDENTITY}")
    print(f"{'='*60}")
    t0 = time.time()

    template_predictions: dict = {}
    protenix_queue:       dict = {}

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]
        segs = segments_map.get(tid, [(0, len(seq))])

        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30)
        preds   = []
        used    = set()

        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
            if len(preds) >= N_SAMPLE:
                break
            if sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY:
                break           
            if tmpl_id in used:
                continue

            rng     = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)

            slot = len(preds)
            if slot == 0:
                X = adapted
            elif slot == 1:
                X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
            elif slot == 2:
                longest = max(segs, key=lambda se: se[1] - se[0])
                X = apply_hinge(adapted, longest, rng)
            elif slot == 3:
                X = jitter_chains(adapted, segs, rng)
            else:
                X = smooth_wiggle(adapted, segs, rng)

            refined = adaptive_rna_constraints(X, tid, seq, segments_map, confidence=sim)
            preds.append(refined)
            used.add(tmpl_id)

        template_predictions[tid] = preds
        n_needed = N_SAMPLE - len(preds)
        if n_needed > 0:
            protenix_queue[tid] = (n_needed, seq)
            print(f"  {tid} ({len(seq)} nt): {len(preds)} TBM → need {n_needed} from Protenix")
        else:
            print(f"  {tid} ({len(seq)} nt): all {N_SAMPLE} from TBM ✓")

    elapsed = time.time() - t0
    n_full  = len(test_df) - len(protenix_queue)
    print(f"\nPhase 1 done in {elapsed:.1f}s")
    print(f"  Fully covered by TBM : {n_full}")
    print(f"  Need Protenix        : {len(protenix_queue)}")
    return template_predictions, protenix_queue


# ─────────────── Main ────────────────────────────────────────────────────────
def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()

    if not os.path.isdir(code_dir):
        print(f"Warning: Missing PROTENIX_CODE_DIR: {code_dir}. "
              "Set PROTENIX_CODE_DIR to the repo path if using Protenix.")

    os.environ["PROTENIX_ROOT_DIR"] = root_dir
    if os.path.isdir(code_dir):
        sys.path.append(code_dir)
    ensure_required_files(root_dir)
    seed_everything(SEED)

    # ── Load test and validation data ───────────────────────────────────────
    print("\nLoading datasets ...")
    if not IS_KAGGLE:
        print(f"Running in LOCAL VALIDATION mode. Extracting {LOCAL_N_SAMPLES} targets from train set.")
        test_df_full = pd.read_csv(DEFAULT_TRAIN_CSV)
        test_df = test_df_full.head(LOCAL_N_SAMPLES).reset_index(drop=True)
        # Ensure we don't leak the identical sequences into TBM by removing them from train pool
        train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV).iloc[LOCAL_N_SAMPLES:].reset_index(drop=True)
        val_true_labels = process_labels(pd.read_csv(DEFAULT_TRAIN_LBLS))
    else:
        test_df_full = pd.read_csv(test_csv)
        test_df = test_df_full.reset_index(drop=True)
        train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV)
        val_true_labels = None

    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Test targets : {len(test_df)}")
    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        test_df, combined_seqs, train_coords, segments_map
    )

    # ─── PHASE 2: Protenix (only for targets that need extra predictions) ──
    protenix_preds: dict = {}

    if protenix_queue and USE_PROTENIX and os.path.isdir(code_dir):
        print(f"\n{'='*60}")
        print(f"PHASE 2: Protenix for {len(protenix_queue)} targets")
        print(f"{'='*60}")

        work_dir = Path("/kaggle/working")
        work_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        chunk_info = {}
        
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            if seq_len <= MAX_SEQ_LEN:
                tasks.append({"target_id": target_id, "sequence": full_seq})
                chunk_info[target_id] = [{"name": target_id, "range": (0, seq_len)}]
                print(f"  {target_id} ({seq_len} nt): single pass queued")
            else:
                chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
                print(f"  {target_id} ({seq_len} nt): {len(chunks)} chunks queued ")
                
                chunk_info[target_id] = []
                for ci, (cs, ce) in enumerate(chunks):
                    chunk_name = f"{target_id}_chunk{ci}"
                    sub_seq = full_seq[cs:ce]
                    tasks.append({"target_id": chunk_name, "sequence": sub_seq})
                    chunk_info[target_id].append({"name": chunk_name, "range": (cs, ce)})

        tasks_df = pd.DataFrame(tasks)
        input_json_path = str(work_dir / "protenix_queue_input.json")
        build_input_json(tasks_df, input_json_path)

        try:
            from protenix.data.inference.infer_dataloader import InferenceDataset
            from runner.inference import (InferenceRunner,
                                          update_gpu_compatible_configs,
                                          update_inference_configs)

            configs = build_configs(input_json_path, str(work_dir / "outputs"), MODEL_NAME)
            configs = update_gpu_compatible_configs(configs)
            runner  = InferenceRunner(configs)
            dataset = InferenceDataset(configs)

            raw_predictions = {}

            def _extract_c1_coords(prediction, feat, chunk_seq_len, raw_coords):
                if "centre_atom_mask" in feat:
                    mask = (feat["centre_atom_mask"] == 1).to(raw_coords.device)
                elif "atom_to_tokatom_idx" in feat:
                    m11 = (feat["atom_to_tokatom_idx"] == 11).to(raw_coords.device)
                    m12 = (feat["atom_to_tokatom_idx"] == 12).to(raw_coords.device)
                    c11, c12 = m11.sum(), m12.sum()
                    mask = m11 if abs(c11 - chunk_seq_len) < abs(c12 - chunk_seq_len) else m12
                else:
                    mask = torch.zeros(raw_coords.shape[1], dtype=torch.bool, device=raw_coords.device)
                
                coords = raw_coords[:, mask, :].detach().cpu().numpy()
                
                if coords.shape[1] > 1:
                    diffs = np.linalg.norm(coords[0, 1:] - coords[0, :-1], axis=-1)
                    if np.all(diffs < 1e-4):
                        print(f"    WARNING: Collapsed coordinates detected")
                        return None
                
                if coords.shape[1] != chunk_seq_len:
                    if coords.shape[1] == 1 and chunk_seq_len > 1:
                        return None
                    padded = np.zeros((coords.shape[0], chunk_seq_len, 3), dtype=np.float32)
                    ml = min(coords.shape[1], chunk_seq_len)
                    padded[:, :ml, :] = coords[:, :ml, :]
                    coords = padded
                return coords

            for i in tqdm(range(len(dataset)), desc="Protenix Inference"):
                data, atom_array, err = dataset[i]
                sample_name = data.get("sample_name", f"sample_{i}")
                
                if err:
                    print(f"  {sample_name} data error: {err}")
                    raw_predictions[sample_name] = None
                    del data, atom_array, err
                    gc.collect(); torch.cuda.empty_cache(); gc.collect()
                    continue
                
                target_id = sample_name.split("_chunk")[0] if "_chunk" in sample_name else sample_name
                n_needed = protenix_queue.get(target_id, (N_SAMPLE, ""))[0]
                sub_seq_len = data["N_token"].item()
                
                try:
                    new_cfg = update_inference_configs(configs, sub_seq_len)
                    new_cfg.sample_diffusion.N_sample = n_needed
                    runner.update_model_configs(new_cfg)
                    
                    pred = runner.predict(data)
                    raw_coords = pred["coordinate"]
                    
                    coords = _extract_c1_coords(pred, data["input_feature_dict"], 
                                                sub_seq_len, raw_coords)
                    raw_predictions[sample_name] = coords
                except Exception as exc:
                    print(f"  {sample_name} inference failed: {exc}")
                    raw_predictions[sample_name] = None
                finally:
                    try: del pred, data, atom_array, raw_coords
                    except: pass
                    gc.collect(); torch.cuda.empty_cache(); gc.collect()

            for target_id, (n_needed, full_seq) in protenix_queue.items():
                seq_len = len(full_seq)
                chunks = chunk_info.get(target_id, [])
                
                if not chunks:
                    continue

                if len(chunks) == 1:
                    coords = raw_predictions.get(target_id)
                    protenix_preds[target_id] = coords
                    if coords is not None:
                        print(f"  {target_id}: {coords.shape[0]} predictions generated")
                    else:
                        print(f"  {target_id}: FAILED")
                else:
                    chunk_results_per_sample = {s: [] for s in range(n_needed)}
                    all_ok = True
                    
                    for ci, cinfo in enumerate(chunks):
                        cname = cinfo["name"]
                        crange = cinfo["range"]
                        ccoords = raw_predictions.get(cname)
                        
                        if ccoords is None:
                            all_ok = False
                            break
                        
                        for s_idx in range(n_needed):
                            if s_idx < ccoords.shape[0]:
                                chunk_results_per_sample[s_idx].append((ccoords[s_idx], crange))
                            else:
                                chunk_results_per_sample[s_idx].append((ccoords[-1], crange))
                    
                    if not all_ok:
                        print(f"  {target_id}: chunked inference incomplete, using fallback")
                        protenix_preds[target_id] = None
                        continue
                    
                    stitched_samples = []
                    for s_idx in range(n_needed):
                        items = chunk_results_per_sample[s_idx]
                        coords_list = [c for c, _ in items]
                        ranges_list = [r for _, r in items]
                        full_coords = stitch_chunk_coords(coords_list, ranges_list, seq_len)
                        stitched_samples.append(full_coords)
                    
                    result = np.stack(stitched_samples, axis=0)
                    protenix_preds[target_id] = result
                    print(f"  {target_id}: {result.shape[0]} stitched predictions generated")

        except Exception as e:
            print(f"Protenix loading/inference failed: {e}")

    elif protenix_queue and not USE_PROTENIX:
        print(f"\nPHASE 2 skipped (USE_PROTENIX=False). "
              f"De-novo fallback will cover {len(protenix_queue)} targets.")

    # ─── PHASE 3: Combine everything ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*60}")

    all_rows = []

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        combined: list = list(template_preds.get(tid, []))

        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])

        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, seq, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    # ── Save ───────────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]
    coord_cols = [c for c in cols if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub[cols].to_csv(output_csv, index=False)

    print(f"\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")

    # ── Evaluate Local Validation RMSD ─────────────────────────────────────
    if not IS_KAGGLE and val_true_labels is not None:
        rmsd_list = []
        pred_coords = process_labels(sub)
        for tid in test_df['target_id']:
            if tid in pred_coords and tid in val_true_labels:
                P = pred_coords[tid]
                Q = val_true_labels[tid]
                if len(P) == len(Q):
                    R, t = kabsch_align(P, Q)
                    P_aligned = (P @ R.T) + t
                    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1))))
                    rmsd_list.append(rmsd)
                    print(f"  [Validation] {tid} RMSD: {rmsd:.5f}")
        
        if rmsd_list:
            final_rmsd = np.mean(rmsd_list)
            print(f"\nLOCAL_VALIDATION_RMSD: {final_rmsd:.5f}")
        else:
            print("\nLOCAL_VALIDATION_RMSD: 99.99999")


if __name__ == "__main__":
    main()
