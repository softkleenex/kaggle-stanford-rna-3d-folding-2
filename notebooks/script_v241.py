import subprocess
import sys
import os

def install_wheels():
    wheels = [
        "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
    ]
    for wheel in wheels:
        if os.path.exists(wheel):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
            except Exception as e:
                print(f"Failed to install {wheel}: {e}")

install_wheels()


import os
import sys
import pandas as pd

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True 

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
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                return torch.as_tensor(m)
        except: pass

    if "input_feature_dict" in data:
        f = data["input_feature_dict"]
        if "centre_atom_mask" in f:
            return (f["centre_atom_mask"] == 1)
        
        if "atom_to_tokatom_idx" in f:
            m11 = (f["atom_to_tokatom_idx"] == 11)
            m12 = (f["atom_to_tokatom_idx"] == 12)
            n_tokens = data.get("N_token", torch.tensor(0)).item()
            if abs(m11.sum().item() - n_tokens) < abs(m12.sum().item() - n_tokens):
                return m11
            return m12
    return None

def resolve_paths():
    if IS_KAGGLE:
        DATA_BASE = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
        if not os.path.exists(DATA_BASE):
            DATA_BASE = "/kaggle/input/stanford-rna-3d-folding-2"
        CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
        ROOT_DIR = CODE_DIR
        OUTPUT_CSV = "/kaggle/working/submission.csv"
        TEST_CSV = f"{DATA_BASE}/test_sequences.csv"
    else:
        DATA_BASE = "./data"
        CODE_DIR = "./protenix"
        ROOT_DIR = "."
        OUTPUT_CSV = "./submission.csv"
        TEST_CSV = f"{DATA_BASE}/test_sequences.csv"
    return TEST_CSV, OUTPUT_CSV, CODE_DIR, ROOT_DIR

DEFAULT_TRAIN_CSV  = "/kaggle/input/stanford-rna-3d-folding-2/train_sequences.csv"
DEFAULT_TRAIN_LBLS = "/kaggle/input/stanford-rna-3d-folding-2/train_labels.csv"
DEFAULT_VAL_CSV    = "/kaggle/input/stanford-rna-3d-folding-2/validation_sequences.csv"
DEFAULT_VAL_LBLS   = "/kaggle/input/stanford-rna-3d-folding-2/validation_labels.csv"

# Hyperparameters
MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 40
SEED          = 42
MAX_SEQ_LEN   = 512
CHUNK_OVERLAP = 64
MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0
USE_PROTENIX         = True
USE_MSA              = "false"
USE_TEMPLATE         = "false"
USE_RNA_MSA          = "false"

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def stitch_chunk_coords(coords_list, ranges_list, full_len):
    full = np.zeros((full_len, 3))
    counts = np.zeros(full_len)
    for c, (s, e) in zip(coords_list, ranges_list):
        full[s:e] += c
        counts[s:e] += 1
    full /= np.where(counts[:, None] > 0, counts[:, None], 1)
    
    for i in range(full_len - 1):
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
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            d = np.linalg.norm(coords[i] - coords[j])
            if 10.0 < d < 22.0:
                pairs.append((i, j))
    
    stems = []
    used = set()
    for i, j in pairs:
        if (i, j) in used: continue
        stem = [(i, j)]
        used.add((i, j))
        for k in range(1, 12):
            ni, nj = i + k, j - k
            if ni < L and nj >= 0 and ni < nj and (ni, nj) in pairs:
                stem.append((ni, nj)); used.add((ni, nj))
            else: break
        if len(stem) >= 3: stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    stems = detect_simple_stems(X)
    
    for _ in range(passes):
        # ── Consecutive Nucleobase Stacking (CNBS) Constraint ────────────────
        cnbs_strength = 0.35 * strength
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 2: continue
            d_cnbs = C[1:] - C[:-1]
            dist_cnbs = np.linalg.norm(d_cnbs, axis=1) + 1e-12
            adj_cnbs = d_cnbs * ((5.9 - dist_cnbs) / dist_cnbs)[:, None] * cnbs_strength
            C[:-1] -= adj_cnbs * 0.5
            C[1:] += adj_cnbs * 0.5
            X[s:e] = C

        # ── Base-Pair Phosphorus Distance (BPPD) Constraint ──────────────────
        if stems:
            bppd_strength = 0.28 * strength
            for stem in stems:
                for i_bp, j_bp in stem:
                    if i_bp < len(X) and j_bp < len(X):
                        dv_bp = X[j_bp] - X[i_bp]; dl_bp = np.linalg.norm(dv_bp) + 1e-6
                        adj_bp = dv_bp * ((17.5 - dl_bp) / dl_bp) * bppd_strength
                        X[i_bp] -= adj_bp * 0.5; X[j_bp] += adj_bp * 0.5

        # ── STRATEGY: 12.0 Å Major Groove Distance Constraint (P(i)-P(j-2)) ──
        # Implement a 12.0 Å distance constraint between phosphorus atoms P(i) 
        # and P(j-2) for base-paired residues (i, j) to enforce the 
        # characteristic narrow major groove of A-form RNA and resolve the 
        # structural collapse caused by the misapplied v240 constraint.
        if stems:
            amg_strength = 0.25 * strength
            for stem in stems:
                for i_bp, j_bp in stem:
                    # Constraint for P(i) and P(j-2)
                    p1, p2 = i_bp, j_bp - 2
                    if p1 < len(X) and p2 >= 0 and p2 < len(X):
                        if any(s <= p2 and j_bp < e for s, e in segments):
                            dv = X[p2] - X[p1]; dl = np.linalg.norm(dv) + 1e-6
                            adj = dv * ((12.0 - dl) / dl) * amg_strength
                            X[p1] -= adj * 0.5; X[p2] += adj * 0.5
                    
                    # Constraint for P(j) and P(i-2)
                    p1, p2 = j_bp, i_bp - 2
                    if p1 < len(X) and p2 >= 0 and p2 < len(X):
                        if any(s <= p2 and i_bp < e for s, e in segments):
                            dv = X[p2] - X[p1]; dl = np.linalg.norm(dv) + 1e-6
                            adj = dv * ((12.0 - dl) / dl) * amg_strength
                            X[p1] -= adj * 0.5; X[p2] += adj * 0.5

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

        # ── Backbone Smoothing & PPD2/PPD3 ──────────────────────────────────
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 4: continue
            in_stem_seg = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_seg[i_s - s] = True
                        if s <= j_s < e: in_stem_seg[j_s - s] = True
            
            # PPD2: i to i+2
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            ppd2_mask = in_stem_seg[:-2] & in_stem_seg[2:]
            if np.any(ppd2_mask):
                adj2 = d2[ppd2_mask] * ((11.7 - d2n[ppd2_mask]) / d2n[ppd2_mask])[:, None] * (0.26 * strength)
                C[:-2][ppd2_mask] -= adj2 * 0.5; C[2:][ppd2_mask] += adj2 * 0.5
            
            # PPD3: i to i+3
            d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
            ppd3_mask = in_stem_seg[:-3] & in_stem_seg[3:]
            if np.any(ppd3_mask):
                adj3 = d3[ppd3_mask] * ((14.8 - d3n[ppd3_mask]) / d3n[ppd3_mask])[:, None] * (0.26 * strength)
                C[:-3][ppd3_mask] -= adj3 * 0.5; C[3:][ppd3_mask] += adj3 * 0.5
            
            X[s:e] = C

            # General backbone adjacency
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.9 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj * 0.5; C[1:] += adj * 0.5
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

def coords_to_rows(tid, seq, coords):
    rows = []
    for s in range(coords.shape[0]):
        for r in range(coords.shape[1]):
            rows.append({
                "target_id": tid,
                "resid": r + 1,
                "x_1": coords[s, r, 0],
                "y_1": coords[s, r, 1],
                "z_1": coords[s, r, 2],
                "sample_id": s
            })
    return rows

def main() -> None:
    seed_everything(SEED)
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    
    test_df = pd.read_csv(test_csv)
    if not IS_KAGGLE and LOCAL_N_SAMPLES:
        test_df = test_df.head(LOCAL_N_SAMPLES)

    print("\nLoading TBM templates \u2026")
    train_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    train_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    segments_map, _ = build_segments_map(test_df)

    template_preds, protenix_queue = tbm_phase(test_df, train_seqs, train_coords, segments_map)
    
    protenix_preds = {}
    for tid, (n, seq) in protenix_queue.items():
        coords_list = []
        for i in range(n):
            coords_list.append(generate_rna_structure(seq, seed=i))
        protenix_preds[tid] = np.stack(coords_list)

    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        combined = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(adaptive_rna_constraints(ptx[j], tid, segments_map, confidence=0.3))
        
        while len(combined) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=len(combined))
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.1))
            
        all_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE])))

    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    print(f"\n\u2713 Saved submission to {output_csv}")

if __name__ == "__main__":
    main()
