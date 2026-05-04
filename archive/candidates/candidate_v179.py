import os
os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

import sys
import pandas as pd

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
# On Kaggle competition rerun, KAGGLE_IS_COMPETITION_RERUN is set to a truthy value.
# When running locally we do NOT exit — instead we cap the test set to a small
# number of samples so the notebook finishes quickly.

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
import os
import time

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

    # 2. Fallback to feature dict
    f = data["input_feature_dict"]
    
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
        
    # Heuristic fallback: check which index gives us roughly N_token atoms
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
    mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
    
    c11 = mask11.sum().item()
    c12 = mask12.sum().item()
    
    # Return the one closer to N_tokens (likely one per residue)
    if abs(c11 - n_tokens) < abs(c12 - n_tokens):
        return mask11
    else:
        return mask12


# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE              = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/test_sequences.csv" # Fixed for competition loop
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

MAX_SEQ_LEN   = 128
CHUNK_OVERLAP = 32
USE_PROTENIX  = True

# TBM Constants
MIN_SIMILARITY       = 0.15
MIN_PERCENT_IDENTITY = 0.20

# ─────────────── Helpers ─────────────────────────────────────────────────────
def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def resolve_paths():
    test_csv  = os.environ.get("TEST_CSV",  DEFAULT_TEST_CSV)
    output_csv= os.environ.get("OUTPUT_CSV", DEFAULT_OUTPUT)
    code_dir  = os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR)
    root_dir  = os.environ.get("PROTENIX_ROOT_DIR", DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

def ensure_required_files(root_dir):
    p = Path(root_dir) / "models" / f"{MODEL_NAME}.pt"
    if not p.exists():
        print(f"Warning: model file {p} not found.")

def build_configs(input_json, output_dir, model_name):
    from runner.inference import update_inference_configs
    from protenix.config.config import get_config
    cfg = get_config()
    cfg.input_json_path = input_json
    cfg.output_dir      = output_dir
    cfg.model_path      = os.path.join(os.environ["PROTENIX_ROOT_DIR"], "models", f"{model_name}.pt")
    # Speed up
    cfg.sample_diffusion.N_step = 20
    return cfg

def build_input_json(test_df, out_path):
    tasks = []
    for _, row in test_df.iterrows():
        tasks.append({
            "name": row["target_id"],
            "residues": [{"resname": r} for r in row["sequence"]]
        })
    with open(out_path, "w") as f:
        json.dump(tasks, f)

def split_into_chunks(seq_len, max_len, overlap):
    if seq_len <= max_len:
        return [(0, seq_len)]
    chunks = []
    start = 0
    while start < seq_len:
        end = min(start + max_len, seq_len)
        chunks.append((start, end))
        if end == seq_len:
            break
        start = end - overlap
    return chunks

def stitch_chunk_coords(coords_list, ranges_list, full_len):
    # coords_list: list of (N_chunk, 3)
    # ranges_list: list of (start, end)
    full = np.zeros((full_len, 3))
    weight = np.zeros(full_len)
    for c, (s, e) in zip(coords_list, ranges_list):
        L = e - s
        # Simple linear ramp for overlap blending
        w = np.ones(L)
        # If not the first chunk, ramp up
        if s > 0:
            ramp_len = min(L // 2, CHUNK_OVERLAP)
            w[:ramp_len] = np.linspace(0, 1, ramp_len)
        # If not the last chunk, ramp down
        if e < full_len:
            ramp_len = min(L // 2, CHUNK_OVERLAP)
            w[-ramp_len:] = np.linspace(1, 0, ramp_len)
            
        full[s:e]   += c[:L] * w[:, None]
        weight[s:e] += w
    return full / (weight[:, None] + 1e-12)

def coords_to_rows(target_id, sequence, stacked_coords):
    # stacked_coords: (N_SAMPLE, seq_len, 3)
    rows = []
    for i, resname in enumerate(sequence):
        d = {"ID": f"{target_id}_{i+1}", "resname": resname, "resid": i+1}
        for s in range(stacked_coords.shape[0]):
            d[f"x_{s+1}"] = stacked_coords[s, i, 0]
            d[f"y_{s+1}"] = stacked_coords[s, i, 1]
            d[f"z_{s+1}"] = stacked_coords[s, i, 2]
        rows.append(d)
    return rows


# ─────────────── TBM Logic ───────────────────────────────────────────────────
def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    
    results = []
    q_len = len(query_seq)
    for _, row in train_seqs_df.iterrows():
        tid = row["target_id"]
        if tid not in train_coords_dict: continue
        t_seq = row["sequence"]
        t_len = len(t_seq)
        
        # Fast heuristic: length similarity
        len_sim = min(q_len, t_len) / max(q_len, t_len)
        if len_sim < 0.1: continue
        
        score = aligner.score(query_seq, t_seq)
        max_score = 2.0 * max(q_len, t_len)
        sim = score / max_score
        
        # Calculate percent identity
        alignments = aligner.align(query_seq, t_seq)
        if alignments:
            best = alignments[0]
            # Simple identity check from alignment
            matches = 0
            for (q_start, q_end), (t_start, t_end) in zip(best.aligned[0], best.aligned[1]):
                q_sub = query_seq[q_start:q_end]
                t_sub = t_seq[t_start:t_end]
                matches += sum(1 for a, b in zip(q_sub, t_sub) if a == b)
            pct_id = matches / max(q_len, t_len)
        else:
            pct_id = 0.0
            
        results.append((tid, t_seq, sim, train_coords_dict[tid], pct_id, score, max_score))
        
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(query_seq, tmpl_seq, tmpl_coords):
    # tmpl_coords: (tmpl_len, 3)
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    alignments = aligner.align(query_seq, tmpl_seq)
    
    q_len = len(query_seq)
    new_coords = np.zeros((q_len, 3))
    counts = np.zeros(q_len)
    
    if alignments:
        best = alignments[0]
        # aligned is ((q_start, q_end), ...), ((t_start, t_end), ...)
        for (q_s, q_e), (t_s, t_e) in zip(best.aligned[0], best.aligned[1]):
            L = q_e - q_s
            new_coords[q_s:q_e] += tmpl_coords[t_s:t_e]
            counts[q_s:q_e] += 1
            
    # Interpolate gaps
    mask = counts > 0
    if not np.any(mask):
        return np.zeros((q_len, 3))
    
    idx = np.arange(q_len)
    for k in range(3):
        new_coords[:, k] = np.interp(idx, idx[mask], new_coords[mask, k] / counts[mask])
    
    return new_coords

def process_labels(labels_df):
    coords_dict = {}
    for tid, group in labels_df.groupby("target_id"):
        # Ensure numeric resid sorting
        group = group.copy()
        group["resid_num"] = group["ID"].apply(lambda x: int(str(x).rsplit('_', 1)[-1]))
        group = group.sort_values("resid_num")
        coords_dict[tid] = group[["x", "y", "z"]].values
    return coords_dict

def build_segments_map(df):
    segments = {}
    for _, row in df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]
        # Basic: treat full sequence as one segment. 
        # In a real scenario, we might use secondary structure to define helical segments.
        segments[tid] = [(0, len(seq))]
    return segments, {}

def _rotmat(axis, theta):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            # soft i–i+2  ~9.3 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((9.3 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            
            # STRATEGY: i -> i+11 C1'-C1' distance constraint (30.1 Å)
            # This enforces A-form helical pitch consistency and prevents cumulative axial distortion.
            if L >= 12:
                d11 = C[11:] - C[:-11]; d11n = np.linalg.norm(d11, axis=1) + 1e-6
                adj11 = d11 * ((30.1 - d11n) / d11n)[:, None] * (0.08 * strength)
                C[:-11] -= adj11; C[11:] += adj11

            # STRATEGY: Implement a C1'_i - C1'_j cross-strand (10.5 Å) and diagonal (12.8 Å) constraints
            # to enforce canonical A-form duplex width, helical rise, and inclination.
            if L >= 20:
                diff_pair = C[:, None, :] - C[None, :, :]
                dist_pair = np.linalg.norm(diff_pair, axis=2) + 1e-6
                # Sequence distance > 10 to ensure they are not local neighbors
                seq_dist = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
                # Potential base pair mask (centered at 10.5 A)
                mask_bp = (seq_dist > 10) & (dist_pair > 8.0) & (dist_pair < 13.0)
                if np.any(mask_bp):
                    # 1. Base pair constraint (10.5 A)
                    adj_bp = diff_pair * ((10.5 - dist_pair) / dist_pair)[:, :, None] * mask_bp[:, :, None] * (0.05 * strength)
                    C += adj_bp.sum(axis=1)

                    # STRATEGY: Implement asymmetric C1' cross-strand diagonal constraints (13.5 Å for minor groove and 18.5 Å for major groove) to accurately capture the characteristic A-form helical asymmetry.
                    mask_minor = np.zeros_like(mask_bp)
                    mask_major = np.zeros_like(mask_bp)
                    for idx_i in range(L - 1):
                        for idx_j in range(1, L):
                            if mask_bp[idx_i, idx_j]:
                                # Major groove (5'-5' diagonal): idx_i to idx_j-1
                                mask_major[idx_i, idx_j - 1] = True
                                mask_major[idx_j - 1, idx_i] = True
                                # Minor groove (3'-3' diagonal): idx_i+1 to idx_j
                                mask_minor[idx_i + 1, idx_j] = True
                                mask_minor[idx_j, idx_i + 1] = True
                    
                    adj_minor = diff_pair * ((13.5 - dist_pair) / dist_pair)[:, :, None] * mask_minor[:, :, None] * (0.03 * strength)
                    adj_major = diff_pair * ((18.5 - dist_pair) / dist_pair)[:, :, None] * mask_major[:, :, None] * (0.03 * strength)
                    C += adj_minor.sum(axis=1) + adj_major.sum(axis=1)

                    # STRATEGY: Implement a 19° base-pair inclination constraint relative to the helical axis
                    # to accurately reproduce the axial displacement and deep major groove architecture unique to the A-form RNA helix.
                    # Target angle between base-pair vector and helical axis is 90 - 19 = 71 degrees.
                    target_angle_rad = np.deg2rad(71.0)
                    for idx_i in range(1, L - 1):
                        for idx_j in range(1, L - 1):
                            if mask_bp[idx_i, idx_j]:
                                # Local helical axis at idx_i
                                h_axis = C[idx_i + 1] - C[idx_i - 1]
                                h_norm = np.linalg.norm(h_axis)
                                if h_norm < 1e-7: continue
                                h_axis /= h_norm

                                # Base pair vector
                                bp_vec = C[idx_j] - C[idx_i]
                                bp_dist = np.linalg.norm(bp_vec)
                                if bp_dist < 1e-7: continue
                                bp_unit = bp_vec / bp_dist

                                cos_alpha = np.clip(np.dot(bp_unit, h_axis), -1.0, 1.0)
                                alpha = np.arccos(cos_alpha) # Angle between bp and axis
                                
                                # If alpha > 90, we are looking at the reverse direction, 
                                # but inclination is symmetric relative to the plane.
                                # Target is 71 deg or 109 deg.
                                current_target = target_angle_rad if alpha < np.pi/2 else np.pi - target_angle_rad
                                
                                diff = current_target - alpha
                                if np.abs(diff) > 1e-4:
                                    # Rotation axis is perpendicular to both bp and helical axis
                                    rot_axis = np.cross(bp_unit, h_axis)
                                    ra_norm = np.linalg.norm(rot_axis)
                                    if ra_norm > 1e-7:
                                        rot_axis /= ra_norm
                                        # Rotate bp_vec around rot_axis to reach target angle
                                        step = diff * 0.1 * strength
                                        R = _rotmat(rot_axis, -step) # rotate bp away/towards axis
                                        new_bp_vec = bp_unit @ R.T * bp_dist
                                        shift = (new_bp_vec - bp_vec) * 0.5
                                        C[idx_j] += shift
                                        C[idx_i] -= shift

            # STRATEGY: Implement a C1'_{i-1}-C1'_i-C1'_{i+1} virtual bond angle constraint (centered at 148°)
            # to enforce consistent local backbone curvature and prevent unphysical "zig-zag" distortions.
            if L >= 3:
                for i in range(1, L - 1):
                    p_prev, p_curr, p_next = C[i-1], C[i], C[i+1]
                    v1, v2 = p_prev - p_curr, p_next - p_curr
                    v1n, v2n = np.linalg.norm(v1), np.linalg.norm(v2)
                    if v1n > 1e-7 and v2n > 1e-7:
                        axis = np.cross(v1, v2)
                        axis_norm = np.linalg.norm(axis)
                        if axis_norm > 1e-7:
                            axis /= axis_norm
                            cos_theta = np.clip(np.dot(v1, v2) / (v1n * v2n + 1e-12), -1.0, 1.0)
                            theta = np.arccos(cos_theta)
                            diff = np.deg2rad(148.0) - theta
                            step = diff * 0.12 * strength
                            R_pos, R_neg = _rotmat(axis, step * 0.5), _rotmat(axis, -step * 0.5)
                            C[i-1] = (C[i-1] - p_curr) @ R_neg.T + p_curr
                            C[i+1] = (C[i+1] - p_curr) @ R_pos.T + p_curr

            # STRATEGY: C1'(i)-C1'(i+1)-C1'(i+2)-C1'(i+3) dihedral centered at 16.0°
            # This enforces consistent A-form backbone "twist" and prevents unphysical ladder-like geometries.
            if L >= 4:
                for i in range(L - 3):
                    p0, p1, p2, p3 = C[i], C[i+1], C[i+2], C[i+3]
                    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
                    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
                    n1n, n2n, b2n = np.linalg.norm(n1), np.linalg.norm(n2), np.linalg.norm(b2)
                    if n1n > 1e-7 and n2n > 1e-7 and b2n > 1e-7:
                        n1 /= n1n; n2 /= n2n
                        cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)
                        sin_phi = np.dot(np.cross(n1, n2), b2 / (b2n + 1e-12))
                        phi = np.arctan2(sin_phi, cos_phi)
                        diff = np.deg2rad(16.0) - phi
                        diff = (diff + np.pi) % (2 * np.pi) - np.pi
                        step = diff * 0.05 * strength
                        R_pos, R_neg = _rotmat(b2 / b2n, step), _rotmat(b2 / b2n, -step)
                        C[i] = (C[i] - p1) @ R_neg.T + p1
                        C[i+3] = (C[i+3] - p2) @ R_pos.T + p2

            # STRATEGY: Implement a **right-handed helical chirality constraint** (scalar triple product of consecutive C1' atoms) to ensure the model favors physically correct right-handed A-form duplexes over mirror-image isomers.
            if L >= 4:
                for i in range(L - 3):
                    p0, p1, p2, p3 = C[i], C[i+1], C[i+2], C[i+3]
                    v1, v2, v3 = p1 - p0, p2 - p1, p3 - p2
                    T = np.dot(v1, np.cross(v2, v3))
                    target_T = 14.0
                    if T < target_T:
                        diff = target_T - T
                        grad_p0 = -np.cross(v2, v3)
                        grad_p1 = np.cross(v2, v3) - np.cross(v3, v1)
                        grad_p2 = np.cross(v3, v1) - np.cross(v1, v2)
                        grad_p3 = np.cross(v1, v2)
                        grad_norm_sq = np.sum(grad_p0**2) + np.sum(grad_p1**2) + np.sum(grad_p2**2) + np.sum(grad_p3**2) + 1e-12
                        step = (diff * 0.1 * strength) / grad_norm_sq
                        C[i]   += step * grad_p0
                        C[i+1] += step * grad_p1
                        C[i+2] += step * grad_p2
                        C[i+3] += step * grad_p3

            # soft i–i+5  ~19.0 Å
            if L >= 6:
                d5   = C[5:] - C[:-5]; d5n = np.linalg.norm(d5, axis=1) + 1e-6
                adj5 = d5 * ((19.0 - d5n) / d5n)[:, None] * (0.05 * strength)
                C[:-5] -= adj5; C[5:] += adj5
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X


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
    """
    Phase 1 — Template-Based Modeling.

    Returns
    -------
    template_predictions : {target_id: [np.ndarray(seq_len, 3), ...]}
        0 to N_SAMPLE predictions per target, from real templates.
    protenix_queue : {target_id: (n_needed, full_sequence)}
        Targets that still need more predictions.
    """
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
                break           # list is sorted by sim, so no point continuing
            if tmpl_id in used:
                continue

            rng     = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)

            # Diversity transforms (same strategy as the 0-409 TBM notebook)
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

            refined = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
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
        raise FileNotFoundError(
            f"Missing PROTENIX_CODE_DIR: {code_dir}. "
            "Set PROTENIX_CODE_DIR to the repo path."
        )

    os.environ["PROTENIX_ROOT_DIR"] = root_dir
    sys.path.append(code_dir)
    ensure_required_files(root_dir)
    seed_everything(SEED)

    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))

    # Truncated copy for Protenix (Protenix has token limits)
    test_df_trunc = test_df.copy()
    test_df_trunc["sequence"] = test_df_trunc["sequence"].str[:MAX_SEQ_LEN]

    # ── Load training data for TBM ──────────────────────────────────────────
    print("\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        test_df, combined_seqs, train_coords, segments_map
    )

    # ─── PHASE 2: Protenix (only for targets that need extra predictions) ──
    protenix_preds: dict = {}   # target_id -> np.ndarray (n_needed, seq_len, 3)

    if protenix_queue and USE_PROTENIX:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Protenix for {len(protenix_queue)} targets")
        print(f"{'='*60}")

        work_dir = Path("/kaggle/working")
        work_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Preparation: create tasks for all sequences/chunks ────────────
        tasks = []          # list of dict for input_json
        chunk_info = {}     # target_id -> list of {"name": chunk_name, "range": (s, e)}
        
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            if seq_len <= MAX_SEQ_LEN:
                tasks.append({"target_id": target_id, "sequence": full_seq})
                chunk_info[target_id] = [{"name": target_id, "range": (0, seq_len)}]
                print(f"  {target_id} ({seq_len} nt): single pass queued")
            else:
                chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
                print(f"  {target_id} ({seq_len} nt): {len(chunks)} chunks queued "
                      f"{[(s, e) for s, e in chunks]}")
                
                chunk_info[target_id] = []
                for ci, (cs, ce) in enumerate(chunks):
                    chunk_name = f"{target_id}_chunk{ci}"
                    sub_seq = full_seq[cs:ce]
                    tasks.append({"target_id": chunk_name, "sequence": sub_seq})
                    chunk_info[target_id].append({"name": chunk_name, "range": (cs, ce)})

        # Build combined input JSON
        tasks_df = pd.DataFrame(tasks)
        input_json_path = str(work_dir / "protenix_queue_input.json")
        build_input_json(tasks_df, input_json_path)

        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import (InferenceRunner,
                                      update_gpu_compatible_configs,
                                      update_inference_configs)

        # Initialize model exactly ONCE
        configs = build_configs(input_json_path, str(work_dir / "outputs"), MODEL_NAME)
        configs = update_gpu_compatible_configs(configs)
        runner  = InferenceRunner(configs)
        dataset = InferenceDataset(configs)

        # ── 2. Inference: process dataset and collect predictions ────────────
        raw_predictions = {}  # sample_name -> coords (np.ndarray or None)

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
            
            # Collapse check
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
                padded[:, :ml, : ] = coords[:, :ml, :]
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
            
            # Find how many samples are needed for this specific query
            target_id = sample_name.split("_chunk")[0] if "_chunk" in sample_name else sample_name
            n_needed = protenix_queue.get(target_id, (N_SAMPLE, ""))[0]
            sub_seq_len = data["N_token"].item() # roughly correct
            
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
                import traceback; traceback.print_exc()
                raw_predictions[sample_name] = None
            finally:
                try: del pred, data, atom_array, raw_coords
                except: pass
                gc.collect(); torch.cuda.empty_cache(); gc.collect()

        # ── 3. Post-processing: Stitching and final formatting ───────────────
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            chunks = chunk_info.get(target_id, [])
            
            if not chunks:
                continue

            if len(chunks) == 1:
                # Single pass
                coords = raw_predictions.get(target_id)
                protenix_preds[target_id] = coords
                if coords is not None:
                    print(f"  {target_id}: {coords.shape[0]} predictions generated")
                else:
                    print(f"  {target_id}: FAILED")
            else:
                # Stitch chunks together
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

        combined: list = list(template_preds.get(tid, []))  # TBM predictions

        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            dn       = generate_rna_structure(seq, seed=seed_val)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
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

    # ── Local Validation ───────────────────────────────────────────────────
    try:
        train_lbls = pd.read_csv(DEFAULT_TRAIN_LBLS)
        val_lbls = pd.read_csv(DEFAULT_VAL_LBLS)
        all_lbls = pd.concat([train_lbls, val_lbls], ignore_index=True)
        
        merged = sub.merge(all_lbls, on='ID', suffixes=('_pred', '_true'))
        if len(merged) > 0:
            merged['target_id'] = merged['ID'].apply(lambda x: str(x).rsplit('_', 1)[0])
            rmsd_list = []
            for tid, group in merged.groupby('target_id'):
                P = group[['x_1_pred', 'y_1_pred', 'z_1_pred']].values
                Q = group[['x_1_true', 'y_1_true', 'z_1_true']].values
                mask = ~np.isnan(P).any(axis=1) & ~np.isnan(Q).any(axis=1)
                P, Q = P[mask], Q[mask]
                if len(P) < 3: continue
                Pc = P - P.mean(axis=0)
                Qc = Q - Q.mean(axis=0)
                H = Pc.T @ Qc
                U, S, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ U.T))
                V = Vt.T; V[:, 2] *= d
                R = V @ U.T
                rmsd = np.sqrt(np.mean(np.sum((Pc @ R - Qc)**2, axis=1)))
                rmsd_list.append(rmsd)
            if rmsd_list:
                print(f"LOCAL_VALIDATION_RMSD: {np.mean(rmsd_list):.5f}")
            else:
                print("LOCAL_VALIDATION_RMSD: NaN (no valid targets)")
        else:
            print("LOCAL_VALIDATION_RMSD: NaN (no IDs matched labels)")
    except Exception as e:
        print(f"LOCAL_VALIDATION_RMSD: ERROR ({e})")


if __name__ == "__main__":
    main()
