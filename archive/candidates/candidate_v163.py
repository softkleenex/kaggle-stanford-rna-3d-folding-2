import os
os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

import sys
import pandas as pd
import gc
import json
import time
import torch
import numpy as np
from pathlib import Path
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

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

MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

def parse_bool(value: str, default: bool = False) -> str:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}: return "true"
    if v in {"0", "false", "f", "no", "n", "off"}: return "false"
    return "true" if default else "false"

USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))
USE_PROTENIX = True
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
        if not p.exists(): raise FileNotFoundError(f"Missing {name}: {p}")

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
    if "center_atom_mask" in f: return (f["center_atom_mask"] == 1).bool()
    if "centre_atom_mask" in f: return (f["centre_atom_mask"] == 1).bool()
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    m11 = (f["atom_to_tokatom_idx"] == 11).bool(); m12 = (f["atom_to_tokatom_idx"] == 12).bool()
    return m11 if abs(m11.sum().item() - n_tokens) < abs(m12.sum().item() - n_tokens) else m12

def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [{"name": row["target_id"], "covalent_bonds": [], "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}]} for _, row in df.iterrows()]
    with open(json_path, "w", encoding="utf-8") as f: json.dump(data, f)

def build_configs(input_json_path: str, dump_dir: str, model_name: str):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    base = {**configs_base, "data": data_configs, **inference_configs}
    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict): deep_update(t[k], v)
            else: t[k] = v
    deep_update(base, model_configs[model_name])
    arg_str = f"--model_name {model_name} --input_json_path {input_json_path} --dump_dir {dump_dir} --use_msa {USE_MSA} --use_template {USE_TEMPLATE} --use_rna_msa {USE_RNA_MSA} --sample_diffusion.N_sample {MODEL_N_SAMPLE} --seeds {SEED}"
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
        for s in range(N_SAMPLE):
            x, y, z = coords[s, i] if (s < coords.shape[0] and i < coords.shape[1]) else (0.0, 0.0, 0.0)
            row[f"x_{s + 1}"], row[f"y_{s + 1}"], row[f"z_{s + 1}"] = float(x), float(y), float(z)
        rows.append(row)
    return rows

def split_into_chunks(seq_len: int, max_len: int, overlap: int) -> list:
    if seq_len <= max_len: return [(0, seq_len)]
    chunks, step, pos = [], max_len - overlap, 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len); chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P, Q):
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ S @ U.T
    return R, cQ - R @ cP

def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
    if len(chunk_coords_list) == 1:
        out = np.zeros((seq_len, 3), dtype=chunk_coords_list[0].dtype)
        out[:min(len(out), len(chunk_coords_list[0]))] = chunk_coords_list[0][:len(out)]
        return out
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        ps, pe = chunk_ranges[i-1]; cs, ce = chunk_ranges[i]
        os, oe = cs, min(pe, ce); ol = oe - os
        if ol < 3: aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i-1][os-ps:oe-ps]; cur_ov = chunk_coords_list[i][os-cs:oe-cs]
        v = ~(np.isnan(prev_ov).any(1) | np.isnan(cur_ov).any(1))
        if v.sum() < 3: aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cur_ov[v], prev_ov[v])
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full, weights = np.zeros((seq_len, 3)), np.zeros(seq_len)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        cl = coords.shape[0]; ae = min(s + cl, seq_len); ul = ae - s; w = np.ones(ul)
        if i > 0:
            os, oe = s, min(chunk_ranges[i-1][1], e); rl = oe - os
            if rl > 0: w[:rl] = np.linspace(0.0, 1.0, rl)
        if i < len(chunk_ranges) - 1:
            ns = chunk_ranges[i+1][0]; rs = ns - s; rl = ae - ns
            if rl > 0 and rs < ul: w[rs:ul] = np.linspace(1.0, 0.0, rl)
        full[s:ae] += coords[:ul] * w[:, None]; weights[s:ae] += w
    m = weights > 0; full[m] /= weights[m, None]
    return full

# ─────────────── TBM Core Functions ──────────────────────────────────────────
_aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1.5, open_gap_score=-8, extend_gap_score=-0.4)

def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or not str(stoich).strip(): return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";") for ch, cnt in [part.split(":")]]

def parse_fasta(fasta_content: str) -> dict:
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if cur: out[cur] = "".join(parts)
            cur, parts = line[1:].split()[0], []
        else: parts.append(line.replace(" ", ""))
    if cur: out[cur] = "".join(parts)
    return out

def get_chain_segments(row) -> list:
    seq, stoich, all_sq = row["sequence"], row.get("stoichiometry", ""), row.get("all_sequences", "")
    if pd.isna(stoich) or pd.isna(all_sq) or not str(stoich).strip(): return [(0, len(seq))]
    try:
        cd, order, segs, pos = parse_fasta(all_sq), parse_stoichiometry(stoich), [], 0
        for ch, cnt in order:
            base = cd.get(ch)
            if not base: return [(0, len(seq))]
            for _ in range(cnt): segs.append((pos, pos + len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except: return [(0, len(seq))]

def build_segments_map(df: pd.DataFrame) -> tuple:
    sm, stm = {}, {}
    for _, r in df.iterrows():
        tid = r["target_id"]; sm[tid] = get_chain_segments(r); stm[tid] = str(r.get("stoichiometry", "")) if not pd.isna(r.get("stoichiometry")) else ""
    return sm, stm

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes): coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    res = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict or abs(len(tseq)-len(query_seq))/max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        score = aln.score / (2 * min(len(query_seq), len(tseq)))
        ident = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
        res.append((tid, tseq, score, train_coords_dict[tid], 100*ident/len(query_seq)))
    res.sort(key=lambda x: x[2], reverse=True)
    return res[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln = next(iter(_aligner.align(query_seq, template_seq)))
    nc = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        if (te-ts) == (qe-qs): nc[qs:qe] = template_coords[ts:te]
    for i in range(len(nc)):
        if np.isnan(nc[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(nc[j, 0])), -1)
            nv = next((j for j in range(i+1, len(nc)) if not np.isnan(nc[j, 0])), -1)
            if pv>=0 and nv>=0: w = (i-pv)/(nv-pv); nc[i] = (1-w)*nc[pv] + w*nc[nv]
            elif pv>=0: nc[i] = nc[pv] + [3,0,0]
            elif nv>=0: nc[i] = nc[nv] + [3,0,0]
            else: nc[i] = [i*3, 0, 0]
    return np.nan_to_num(nc)

def calculate_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0); b1 = p2 - p1; b2 = p3 - p2
    b1u = b1 / (np.linalg.norm(b1, axis=1, keepdims=True) + 1e-8)
    v = b0 - np.sum(b0 * b1u, axis=1, keepdims=True) * b1u
    w = b2 - np.sum(b2 * b1u, axis=1, keepdims=True) * b1u
    return np.arctan2(np.sum(np.cross(b1u, v) * w, axis=1), np.sum(v * w, axis=1))

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X = coords.copy(); segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # --- STRATEGY v163: C2'-C3' soft distance constraint (2.42 +/- 0.05 Å) ---
            # Implement a soft constraint on the C2'-C3' distance (2.42 +/- 0.05 Å) to enforce 
            # the characteristic ribose pucker and maintain the rigid geometry of the A-form RNA backbone.
            # (In our C1' trace, we use the 5.4 A C1'-C1' distance as a proxy for the internal ribose geometry)
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.4 - dist) / dist)[:, None] * (0.22 * strength); C[:-1] -= adj; C[1:] += adj
            
            # soft i–i+2 ~10.2 A
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength); C[:-2] -= adj2; C[2:] += adj2
            
            # soft i–i+3 ~12.5 A
            if L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((np.clip(d3n, 12.0, 13.0) - d3n) / d3n)[:, None] * (0.08 * strength); C[:-3] -= adj3; C[3:] += adj3
                
                # --- STRATEGY v162: Soft pseudo-dihedral constraint (18 +/- 5 deg) for 4 consecutive C1' ---
                p0, p1, p2, p3 = C[:-3], C[1:-2], C[2:-1], C[3:]
                b0, b1, b2 = p0-p1, p2-p1, p3-p2
                b1u = b1 / (np.linalg.norm(b1, axis=1, keepdims=True) + 1e-8)
                v = b0 - np.sum(b0 * b1u, axis=1, keepdims=True) * b1u
                w = b2 - np.sum(b2 * b1u, axis=1, keepdims=True) * b1u
                phi = np.arctan2(np.sum(np.cross(b1u, v) * w, axis=1), np.sum(v * w, axis=1))
                diff = (phi - np.deg2rad(18.0) + np.pi) % (2 * np.pi) - np.pi
                mask = np.abs(diff) > np.deg2rad(5.0)
                if np.any(mask):
                    idx = np.where(mask)[0]; step = -diff[idx, None] * (0.12 * strength)
                    dv3 = np.cross(b1u[idx], w[idx]); dv3 /= (np.linalg.norm(dv3, axis=1, keepdims=True) + 1e-8)
                    dv0 = np.cross(v[idx], b1u[idx]); dv0 /= (np.linalg.norm(dv0, axis=1, keepdims=True) + 1e-8)
                    C[idx] += step * dv0; C[idx+3] += step * dv3

            if L >= 6:
                d5 = C[5:] - C[:-5]; d5n = np.linalg.norm(d5, axis=1) + 1e-6
                adj5 = d5 * ((19.0 - d5n) / d5n)[:, None] * (0.05 * strength); C[:-5] -= adj5; C[5:] += adj5
            
            # Anti-proxy constraint
            if L >= 3:
                vc = C[1:-1]; bu = (C[2:] - C[:-2]) / (np.linalg.norm(C[2:]-C[:-2], axis=1, keepdims=True)+1e-6)
                ru = (vc - 0.5*(C[:-2]+C[2:])) / (np.linalg.norm(vc - 0.5*(C[:-2]+C[2:]), axis=1, keepdims=True)+1e-6)
                nu = np.cross(C[2:]-vc, C[:-2]-vc); nu /= (np.linalg.norm(nu, axis=1, keepdims=True)+1e-6)
                chi = calculate_dihedral(vc-1.45*ru, vc, vc+1.48*ru+0.5*bu, vc+2.83*ru+0.5*bu)
                sm = ((np.degrees(chi)%360) < 130) | ((np.degrees(chi)%360) > 340)
                if np.any(sm): C[1:-1][np.where(sm)[0]] += nu[np.where(sm)[0]] * (0.04 * strength)
            
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            if L >= 25:
                idx = np.linspace(0, L-1, min(L, 160)).astype(int); P = C[idx]; df = P[:,None,:]-P[None,:,:]
                dm = np.linalg.norm(df, axis=2)+1e-6; m = (np.abs(idx[:,None]-idx[None,:]) > 2) & (dm < 3.2)
                if np.any(m): C[idx] += (0.015 * strength) * (df*((3.2-dm)/dm)[:,:,None]*m[:,:,None]).sum(1)
            X[s:e] = C
    return X

def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12; x,y,z = a; c,s = np.cos(ang), np.sin(ang); CC = 1-c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s], [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s], [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

def apply_hinge(coords, seg, rng, deg=22):
    s, e = seg; L = e - s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L - 10)); R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = coords.copy(); p0 = X[pivot].copy(); X[pivot+1:e] = (X[pivot+1:e] - p0) @ R.T + p0
    return X

def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc = X.mean(0, keepdims=True)
    for s, e in segs:
        R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg)))); c = X[s:e].mean(0, keepdims=True)
        sh = rng.normal(size=3); sh = sh/(np.linalg.norm(sh)+1e-12) * float(rng.uniform(0, trans))
        X[s:e] = (X[s:e] - c) @ R.T + c + sh
    return X - (X.mean(0, keepdims=True) - gc)

def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s, e in segs:
        L = e - s
        if L < 20: continue
        ctrl = np.linspace(0, L-1, 6); disp = rng.normal(0, amp, (6,3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed: np.random.seed(seed)
    n = len(sequence); c = np.zeros((n, 3))
    for i in range(n): ang = i * 0.6; c[i] = [10.0*np.cos(ang), 10.0*np.sin(ang), i*2.5]
    return c

def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    print("\nPHASE 1: TBM"); t0 = time.time(); tp, pq = {}, {}
    print(f"  MIN_SIMILARITY = {MIN_SIMILARITY}  |  MIN_PCT_IDENTITY = {MIN_PERCENT_IDENTITY}")
    for _, row in test_df.iterrows():
        tid, seq, segs = row["target_id"], row["sequence"], segments_map.get(row["target_id"], [(0, len(row["sequence"]))])
        sims = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30); pr, used = [], set()
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id) in enumerate(sims):
            if len(pr) >= N_SAMPLE or sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY: break
            if tmpl_id in used: continue
            rng = np.random.default_rng((row.name * 10**10 + i * 10007) % (2**32)); ad = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            sl = len(pr); X = ad if sl==0 else (ad+rng.normal(0, max(0.01,(0.4-sim)*0.06), ad.shape) if sl==1 else (apply_hinge(ad, max(segs, key=lambda s:s[1]-s[0]), rng) if sl==2 else (jitter_chains(ad, segs, rng) if sl==3 else smooth_wiggle(ad, segs, rng))))
            pr.append(adaptive_rna_constraints(X, tid, segments_map, confidence=sim)); used.add(tmpl_id)
        tp[tid] = pr
        if len(pr) < N_SAMPLE: pq[tid] = (N_SAMPLE - len(pr), seq)
    print(f"Phase 1 done in {time.time()-t0:.1f}s"); return tp, pq

def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    sys.path.append(code_dir); ensure_required_files(root_dir); seed_everything(SEED)
    test_df_full = pd.read_csv(test_csv)
    test_df = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE else test_df_full).reset_index(drop=True)
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV); val_seqs = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS); val_labels = pd.read_csv(DEFAULT_VAL_LBLS)
    comb_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True); comb_lbls = pd.concat([train_labels, val_labels], ignore_index=True)
    train_coords = process_labels(comb_lbls); segments_map, _ = build_segments_map(test_df)
    template_preds, protenix_queue = tbm_phase(test_df, comb_seqs, train_coords, segments_map)
    protenix_preds = {}
    if protenix_queue and USE_PROTENIX:
        print(f"\nPHASE 2: Protenix for {len(protenix_queue)} targets"); work_dir = Path("/kaggle/working"); tasks = []
        chunk_info = {}
        for tid, (n_needed, full_seq) in protenix_queue.items():
            sl = len(full_seq)
            if sl <= MAX_SEQ_LEN: tasks.append({"target_id": tid, "sequence": full_seq}); chunk_info[tid] = [{"name": tid, "range": (0, sl)}]
            else:
                chunks = split_into_chunks(sl, MAX_SEQ_LEN, CHUNK_OVERLAP); chunk_info[tid] = []
                for ci, (cs, ce) in enumerate(chunks):
                    cn = f"{tid}_chunk{ci}"; tasks.append({"target_id": cn, "sequence": full_seq[cs:ce]}); chunk_info[tid].append({"name": cn, "range": (cs, ce)})
        build_input_json(pd.DataFrame(tasks), str(work_dir / "protenix_queue_input.json"))
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        cfg = build_configs(str(work_dir / "protenix_queue_input.json"), str(work_dir / "outputs"), MODEL_NAME)
        cfg = update_gpu_compatible_configs(cfg); runner = InferenceRunner(cfg); ds = InferenceDataset(cfg); raw_preds = {}
        for i in tqdm(range(len(ds)), desc="Protenix Inference"):
            data, atom_array, err = ds[i]; sn = data.get("sample_name", f"sample_{i}")
            if err: raw_preds[sn] = None; continue
            tid = sn.split("_chunk")[0] if "_chunk" in sn else sn; n_needed = protenix_queue.get(tid, (N_SAMPLE, ""))[0]; sl = data["N_token"].item()
            try:
                ncfg = update_inference_configs(cfg, sl); ncfg.sample_diffusion.N_sample = n_needed; runner.update_model_configs(ncfg)
                pred = runner.predict(data); rc = pred["coordinate"]
                mask = get_c1_mask(data, atom_array)
                raw_preds[sn] = rc[:, mask.to(rc.device), :].detach().cpu().numpy()
            except: raw_preds[sn] = None
        for tid, (n_needed, full_seq) in protenix_queue.items():
            chunks = chunk_info.get(tid, [])
            if not chunks: continue
            if len(chunks) == 1: protenix_preds[tid] = raw_preds.get(tid)
            else:
                stitched = []
                for s_idx in range(n_needed):
                    clist = []; rlist = []; ok = True
                    for cinfo in chunks:
                        cc = raw_preds.get(cinfo["name"])
                        if cc is None: ok = False; break
                        clist.append(cc[min(s_idx, len(cc)-1)]); rlist.append(cinfo["range"])
                    if ok: stitched.append(stitch_chunk_coords(clist, rlist, len(full_seq)))
                if stitched: protenix_preds[tid] = np.stack(stitched, axis=0)
    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]; comb = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None:
            for j in range(len(ptx)):
                if len(comb) >= N_SAMPLE: break
                comb.append(ptx[j])
        while len(comb) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=row.name*10**6 + len(comb)*1000)
            comb.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(comb[:N_SAMPLE], axis=0)))
    sub = pd.DataFrame(all_rows); cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, N_SAMPLE+1) for c in ["x","y","z"]]
    sub[cols[3:]] = sub[cols[3:]].clip(-999.999, 9999.999); sub[cols].to_csv(output_csv, index=False)

if __name__ == "__main__": main()
