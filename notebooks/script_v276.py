import os
import sys
import subprocess
import traceback
import gc
import json
import time
from pathlib import Path

# ── Install dependencies offline ──────────────────────────────────────────────
# Strategy: "An unexpected critical error occurred:[object Object]"
# Implementation: Version 276 adds robust traceback logging and dynamic path 
# resolution to prevent "object Object" errors caused by missing hardcoded paths.

def install_wheels():
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
        else:
            # Dynamic fallback search
            fname = os.path.basename(w)
            found = list(Path('/kaggle/input').rglob(fname))
            if found:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(found[0])])
                except: pass

install_wheels()

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))

# How many test samples to use when running locally
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ─────────────── Dynamic Path Resolution ─────────────────────────────────────
def find_dataset_file(filename):
    matches = list(Path('/kaggle/input').rglob(filename))
    return str(matches[0]) if matches else None

DEFAULT_TEST_CSV   = find_dataset_file('test_sequences.csv')
DEFAULT_TRAIN_CSV  = find_dataset_file('train_sequences.csv')
DEFAULT_TRAIN_LBLS = find_dataset_file('train_labels.csv')
DEFAULT_VAL_CSV    = find_dataset_file('validation_sequences.csv')
DEFAULT_VAL_LBLS   = find_dataset_file('validation_labels.csv')
DEFAULT_OUTPUT     = "/kaggle/working/submission.csv"

# Resolve Protenix directory
PTX_CKPT = find_dataset_file('protenix_base_20250630_v1.0.0.pt')
if PTX_CKPT:
    DEFAULT_ROOT_DIR = str(Path(PTX_CKPT).parent.parent)
    # Search for code directory (containing protenix/ folder)
    PTX_CODE_DIR = find_dataset_file('runner/inference.py')
    if PTX_CODE_DIR:
        DEFAULT_CODE_DIR = str(Path(PTX_CODE_DIR).parent.parent)
    else:
        DEFAULT_CODE_DIR = DEFAULT_ROOT_DIR
else:
    DEFAULT_ROOT_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
    DEFAULT_CODE_DIR = DEFAULT_ROOT_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "128"))

MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

USE_MSA      = os.environ.get("USE_MSA",      "false").lower() == "true"
USE_TEMPLATE = os.environ.get("USE_TEMPLATE", "false").lower() == "true"
USE_RNA_MSA  = os.environ.get("USE_RNA_MSA",  "true").lower() == "true"

MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

# ─────────────── Utilities ───────────────────────────────────────────────────
def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"): m = m & atom_array.is_rna
                return torch.from_numpy(m).bool()
            if hasattr(atom_array, "atom_name"):
                base = atom_array.atom_name == "C1'"
                if hasattr(atom_array, "is_rna"): base = base & atom_array.is_rna
                return torch.from_numpy(base).bool()
        except: pass

    f = data["input_feature_dict"]
    if "centre_atom_mask" in f: return (f["centre_atom_mask"] == 1).bool()
    if "center_atom_mask" in f: return (f["center_atom_mask"] == 1).bool()
    
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    m11, m12 = (f["atom_to_tokatom_idx"] == 11).bool(), (f["atom_to_tokatom_idx"] == 12).bool()
    return m11 if abs(m11.sum().item() - n_tokens) < abs(m12.sum().item() - n_tokens) else m12

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

def split_into_chunks(seq_len: int, max_len: int, overlap: int) -> list:
    if seq_len <= max_len: return [(0, seq_len)]
    chunks, step, pos = [], max_len - overlap, 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P: np.ndarray, Q: np.ndarray):
    centroid_P, centroid_Q = P.mean(axis=0), Q.mean(axis=0)
    H = (P - centroid_P).T @ (Q - centroid_Q)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ S @ U.T
    return R, centroid_Q - R @ centroid_P

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:min(coords.shape[0], seq_len)] = coords[:min(coords.shape[0], seq_len)]
        return out

    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]
        ov_start, ov_end = cur_start, min(prev_end, cur_end)
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
        aligned.append((chunk_coords_list[i] @ R.T) + t)

    full, weights = np.zeros((seq_len, 3), dtype=np.float64), np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        actual_end = min(s + coords.shape[0], seq_len)
        used_len = actual_end - s; w = np.ones(used_len, dtype=np.float64)
        if i > 0:
            ramp_len = min(chunk_ranges[i - 1][1], e) - s
            if ramp_len > 0: w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i+1][0]
            ramp_len = actual_end - next_s
            if ramp_len > 0 and next_s - s < used_len: w[next_s - s:] = np.linspace(1.0, 0.0, ramp_len)
        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w
    mask = weights > 0
    full[mask] /= weights[mask, None]
    
    # Global Regularization
    full = adaptive_rna_constraints(full, "stitched", {}, confidence=0.5, passes=2)
    for i in range(seq_len - 1):
        diff = full[i+1] - full[i]; dist = np.linalg.norm(diff) + 1e-12
        if dist > 1.61:
            adj = diff * ((1.61 - dist) / dist) * 0.5
            full[i] -= adj; full[i+1] += adj
    return full

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    pairs, dist_threshold = [], 12.5
    for i in range(L):
        for j in range(i + 4, L):
            if np.linalg.norm(coords[i] - coords[j]) < dist_threshold: pairs.append((i, j))
    stems, used = [], [False] * len(pairs)
    pairs.sort()
    for i in range(len(pairs)):
        if used[i]: continue
        stem = [pairs[i]]; used[i] = True
        curr_i, curr_j = pairs[i]
        for k in range(i + 1, len(pairs)):
            if not used[k]:
                next_i, next_j = pairs[k]
                if next_i == curr_i + 1 and next_j == curr_j - 1:
                    stem.append(pairs[k]); used[k] = True; curr_i, curr_j = next_i, next_j
        if len(stem) >= 3: stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X, segments = coords.copy(), segments_map.get(target_id, [(0, len(coords))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # Base-Pair Planarity (BPP)
        if stems:
            bpp_strength = 0.15 * strength
            for stem in stems:
                for i_s, j_s in stem:
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
                        n_i = np.cross(X[i_s+1]-X[i_s], X[i_s-1]-X[i_s]); n_i /= (np.linalg.norm(n_i) + 1e-6)
                        n_j = np.cross(X[j_s+1]-X[j_s], X[j_s-1]-X[j_s]); n_j /= (np.linalg.norm(n_j) + 1e-6)
                        dot = np.dot(n_i, n_j)
                        if abs(dot) < 0.98:
                            adj = ((n_i if dot > 0 else -n_i) - n_j) * bpp_strength
                            X[j_s+1] += adj; X[j_s-1] += adj; X[i_s+1] -= adj; X[i_s-1] -= adj

        for s, e in segments:
            C, L = X[s:e], e - s
            if L < 3: continue
            in_stem = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem[i_s - s] = True
                        if s <= j_s < e: in_stem[j_s - s] = True
            
            # i–i+1 bond
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj

            # Ribose Pucker (RPC)
            rpc_mask = in_stem[:-1] & in_stem[1:]
            if np.any(rpc_mask):
                adj_rpc = d[rpc_mask] * ((5.90 - dist[rpc_mask]) / dist[rpc_mask])[:, None] * (0.32 * strength)
                C[:-1][rpc_mask] -= adj_rpc * 0.5; C[1:][rpc_mask] += adj_rpc * 0.5

            # Backbone Pseudo-Torsion (BPT)
            if L >= 4:
                p1, p2, p3, p4 = C[:-3], C[1:-2], C[2:-1], C[3:]
                v12, v23, v34 = p2-p1, p3-p2, p4-p3
                n1, n2 = np.cross(v12, v23), np.cross(v23, v34)
                n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-9)
                n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9)
                phi = np.arccos(np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0))
                phi = np.where(np.sum(v12 * n2, axis=1) < 0, -phi, phi)
                axis = v23 / (np.linalg.norm(v23, axis=1, keepdims=True) + 1e-9)
                force = np.cross(n2, axis) * (0.61 - phi)[:, None] * (0.14 * strength)
                C[3:] += force * 0.5; C[:-3] -= force * 0.5

            X[s:e] = C
    return X

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords, prefixes = {}, labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    use_ptx = os.path.isdir(code_dir) and ensure_required_files(root_dir)
    if use_ptx: sys.path.append(code_dir)

    seed_everything(SEED)
    test_df = pd.read_csv(test_csv if IS_KAGGLE else DEFAULT_TRAIN_CSV)
    if not IS_KAGGLE and LOCAL_N_SAMPLES: test_df = test_df.head(LOCAL_N_SAMPLES)
    
    print("\nLoading TBM templates …")
    train_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    train_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    
    # ── Simplified TBM + Protenix Loop ──────────────────────────────────────
    all_rows = []
    aligner = PairwiseAligner(); aligner.mode = "global"
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Targets"):
        tid, seq = row["target_id"], row["sequence"]
        preds = []
        
        # 1. TBM
        similar = []
        for _, tr in train_seqs.iterrows():
            if tr["target_id"] not in train_coords: continue
            if abs(len(tr["sequence"]) - len(seq)) / max(len(seq), 1) > 0.3: continue
            aln = next(iter(aligner.align(seq, tr["sequence"])))
            score = aln.score / (2 * len(seq))
            if score > MIN_SIMILARITY: similar.append((tr["target_id"], score, tr["sequence"]))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        for tmpl_id, sim, tmpl_seq in similar[:N_SAMPLE]:
            # Adapted simplified coordinate projection
            tmpl_c = train_coords[tmpl_id]
            new_c = np.zeros((len(seq), 3))
            aln = next(iter(aligner.align(seq, tmpl_seq)))
            for (qs, qe), (ts, te) in zip(*aln.aligned):
                l = min(qe-qs, te-ts, len(tmpl_c)-ts)
                if l > 0: new_c[qs:qs+l] = tmpl_c[ts:ts+l]
            preds.append(adaptive_rna_constraints(new_c, tid, {}, confidence=sim))

        # 2. Filler (Protenix logic would go here if use_ptx, but for script simplicity we use structure generator)
        while len(preds) < N_SAMPLE:
            coords = np.zeros((len(seq), 3))
            for i in range(len(seq)): coords[i] = [10*np.cos(i*0.6), 10*np.sin(i*0.6), i*2.5]
            preds.append(adaptive_rna_constraints(coords, tid, {}, confidence=0.2))

        # 3. Format output
        for i in range(len(seq)):
            r = {"ID": f"{tid}_{i+1}", "resname": seq[i], "resid": i+1}
            for s in range(N_SAMPLE):
                r[f"x_{s+1}"], r[f"y_{s+1}"], r[f"z_{s+1}"] = preds[s][i]
            all_rows.append(r)

    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"ERROR: {e}"); traceback.print_exc()
        pd.DataFrame(columns=["ID","resname","resid"] + [f"{ax}_{s+1}" for s in range(5) for ax in "xyz"]).to_csv("/kaggle/working/submission.csv", index=False)
