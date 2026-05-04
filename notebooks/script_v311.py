import os
import sys
import subprocess
import traceback
from pathlib import Path

# ── Install dependencies offline ──────────────────────────────────────────────
# Strategy: "Implement a 15.5 Å distance constraint between phosphorus atoms P(i) 
# and P(j-1) for base-paired residues (i, j) to stabilize the diagonal cross-strand 
# geometry and reinforce the characteristic 32.7° A-form helical twist."
# Implementation: Version 311 integrates the 15.5 Å P(i)-P(j-1) diagonal constraint
# alongside the 16.8 Å P(i)-P(j+1) constraint for base-paired residues.

wheels = [
    "biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
    "biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
]

for w_name in wheels:
    found = list(Path('/kaggle/input').rglob(w_name))
    if found:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(found[0])])
            print(f"Successfully installed {w_name} from {found[0]}")
        except Exception as e:
            print(f"Failed to install {w_name}: {e}")
            traceback.print_exc()
    else:
        print(f"Warning: Wheel not found {w_name}, skipping installation.")

import pandas as pd
import numpy as np
import torch
import gc
import json
import time
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))

# How many test samples to use when running locally
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets "
          f"will be processed to save time.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

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
def resolve_data_paths():
    # Dynamic path resolution for Kaggle competition datasets
    data_dir = None
    for p in Path('/kaggle/input').iterdir():
        if p.is_dir() and (p / 'test_sequences.csv').exists():
            data_dir = p
            break
    
    if data_dir is None:
        data_dir = Path("/kaggle/input/stanford-rna-3d-folding-2")
    
    test_csv   = os.environ.get("TEST_CSV",           str(data_dir / "test_sequences.csv"))
    train_csv  = str(data_dir / "train_sequences.csv")
    train_lbls = str(data_dir / "train_labels.csv")
    val_csv    = str(data_dir / "validation_sequences.csv")
    val_lbls   = str(data_dir / "validation_labels.csv")
    output_csv = os.environ.get("SUBMISSION_CSV",     "/kaggle/working/submission.csv")
    
    # Protenix Code Resolution
    code_dir = os.environ.get("PROTENIX_CODE_DIR", "")
    if not code_dir:
        found_code = list(Path('/kaggle/input').rglob('Protenix-v1'))
        if found_code:
            # Pick the one that looks most like the Protenix root (contains runner/ checkpoint/ etc.)
            for fc in found_code:
                if (fc / "runner").exists() and (fc / "checkpoint").exists():
                    code_dir = str(fc)
                    break
    
    if not code_dir:
        code_dir = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
        
    root_dir = os.environ.get("PROTENIX_ROOT_DIR", code_dir)
    
    return test_csv, train_csv, train_lbls, val_csv, val_lbls, output_csv, code_dir, root_dir

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

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int, target_id: str, segments_map: dict) -> np.ndarray:
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
    full = adaptive_rna_constraints(full, target_id, segments_map, confidence=0.5, passes=2)

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
    parts = str(stoich).split(";")
    res = []
    for p in parts:
        if ":" in p:
            ch, cnt = p.split(":")
            res.append((ch.strip(), int(cnt)))
    return res

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
        try:
            alns = _aligner.align(query_seq, tseq)
            aln = next(iter(alns))
            norm_s    = aln.score / (2 * min(len(query_seq), len(tseq)))
            identical = sum(
                1 for (qs, qe), (ts, te) in zip(*aln.aligned)
                for qp, tp in zip(range(qs, qe), range(ts, te))
                if query_seq[qp] == tseq[tp]
            )
            pct_id = 100 * identical / len(query_seq)
            aq, at = _build_aligned_strings(query_seq, tseq, aln)
            results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
        except StopIteration:
            continue
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    try:
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
    except:
        return generate_rna_structure(query_seq)

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
        if len(stem) >= 3:
            stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── Asymmetric P-P Distance (APDC) Constraint ────────────────────────
        # Strategy: "Implement a 15.5 Å distance constraint between phosphorus atoms 
        # P(i) and P(j-1) for base-paired residues (i, j) to stabilize the diagonal 
        # cross-strand geometry and reinforce the characteristic 32.7° A-form helical twist."
        if stems:
            apdc_strength = 0.23 * strength
            for stem in stems:
                for pair in stem:
                    i, j = pair
                    # P(i) to P(j-1) is approx 15.5 Å (Requested Strategy)
                    tj_minus = j - 1
                    if 0 <= tj_minus < len(X) and i < len(X):
                        dv_minus = X[i] - X[tj_minus]; dl_minus = np.linalg.norm(dv_minus) + 1e-6
                        if 11.0 < dl_minus < 20.0:
                            adj_minus = dv_minus * ((15.5 - dl_minus) / dl_minus) * apdc_strength
                            X[i] += adj_minus * 0.5; X[tj_minus] -= adj_minus * 0.5
                    # P(i) to P(j+1) is approx 16.8 Å (Stability reinforcement)
                    tj_plus = j + 1
                    if 0 <= tj_plus < len(X) and i < len(X):
                        dv_plus = X[i] - X[tj_plus]; dl_plus = np.linalg.norm(dv_plus) + 1e-6
                        if 12.0 < dl_plus < 21.0:
                            adj_plus = dv_plus * ((16.8 - dl_plus) / dl_plus) * apdc_strength
                            X[i] += adj_plus * 0.5; X[tj_plus] -= adj_plus * 0.5

        # ── Cross-Strand P–P Distance (CSPP) Constraint ──────────────────────
        if stems:
            cspp_strength = 0.25 * strength
            for stem in stems:
                for pair in stem:
                    i, j = pair
                    tj_minor = j - 3
                    if 0 <= tj_minor < len(X) and i < len(X):
                        dv_m = X[i] - X[tj_minor]; dl_m = np.linalg.norm(dv_m) + 1e-6
                        if 11.0 < dl_m < 21.0:
                            adj_m = dv_m * ((15.5 - dl_m) / dl_m) * cspp_strength
                            X[i] += adj_m * 0.5; X[tj_minor] -= adj_m * 0.5
                    tj_major = j + 2
                    if 0 <= tj_major < len(X) and i < len(X):
                        dv_M = X[i] - X[tj_major]; dl_M = np.linalg.norm(dv_M) + 1e-6
                        if 7.5 < dl_M < 14.5:
                            adj_M = dv_M * ((10.2 - dl_M) / dl_M) * cspp_strength
                            X[i] += adj_M * 0.5; X[tj_major] -= adj_M * 0.5

        # Other standard constraints (CPP, BPP, CHS, CSC, VBPR, OBS, etc.)
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 2: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            
            # CPP
            cpp_mask = in_stem_seg[:-1] & in_stem_seg[1:]
            if np.any(cpp_mask):
                d_cpp = C[1:] - C[:-1]; dist_cpp = np.linalg.norm(d_cpp, axis=1) + 1e-6
                adj_cpp = d_cpp[cpp_mask] * ((5.85 - dist_cpp[cpp_mask]) / dist_cpp[cpp_mask])[:, None] * (0.35 * strength)
                C[:-1][cpp_mask] -= adj_cpp * 0.5; C[1:][cpp_mask] += adj_cpp * 0.5
            X[s:e] = C

        if stems:
            # VBPR
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

        # Generic local connectivity
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
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
            print(f"  {tid}: {len(preds)} TBM \u2192 need {n_needed} Protenix")
        else:
            print(f"  {tid}: all {N_SAMPLE} TBM \u2713")

    print(f"\nPhase 1 done in {time.time()-t0:.1f}s")
    return template_predictions, protenix_queue

def main() -> None:
    global USE_PROTENIX
    test_csv, train_csv, train_lbls, val_csv, val_lbls, output_csv, code_dir, root_dir = resolve_data_paths()
    
    if not os.path.isdir(code_dir): 
        print(f"Protenix code dir not found at {code_dir}, disabling Protenix.")
        USE_PROTENIX = False
    else:
        os.environ["PROTENIX_ROOT_DIR"] = root_dir
        sys.path.append(code_dir)
        if not ensure_required_files(root_dir): 
            print("Required Protenix files missing, disabling Protenix.")
            USE_PROTENIX = False

    seed_everything(SEED)
    test_df = pd.read_csv(test_csv)
    if not IS_KAGGLE: 
        if LOCAL_N_SAMPLES is not None:
            test_df = test_df.tail(LOCAL_N_SAMPLES).head(LOCAL_N_SAMPLES).reset_index(drop=True)
    
    print("\nLoading TBM templates \u2026")
    tr_seqs = pd.read_csv(train_csv); vl_seqs = pd.read_csv(val_csv)
    train_seqs = pd.concat([tr_seqs, vl_seqs], ignore_index=True)
    
    tr_lbls = pd.read_csv(train_lbls); vl_lbls = pd.read_csv(val_lbls)
    train_coords = process_labels(pd.concat([tr_lbls, vl_lbls], ignore_index=True))
    
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
        try:
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
                        if len(coords_list) == len(chunks): stitched.append(stitch_chunk_coords(coords_list, ranges_list, len(full_seq), tid, segments_map))
                    protenix_preds[tid] = np.stack(stitched, axis=0) if stitched else None
        except Exception as e:
            print(f"Protenix Phase failed: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}\nPHASE 3: Combine Results\n{'='*60}")
    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        combined = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(adaptive_rna_constraints(ptx[j], tid, segments_map, confidence=0.7))
        while len(combined) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=row.name*1000000+len(combined)*1000)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE], axis=0)))

    sub = pd.DataFrame(all_rows)
    coord_cols = [c for c in sub.columns if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub.to_csv(output_csv, index=False)
    print(f"\n\u2713 Saved submission to {output_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR in main loop: {e}")
        traceback.print_exc()
        output_csv = "/kaggle/working/submission.csv"
        if not os.path.exists(output_csv):
            print("Creating empty submission.csv as emergency fallback.")
            pd.DataFrame(columns=["ID","resname","resid"] + [f"{ax}_{s+1}" for s in range(5) for ax in "xyz"]).to_csv(output_csv, index=False)
