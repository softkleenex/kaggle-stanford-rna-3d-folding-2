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
# Implementation: Version 282 implements robust dynamic path resolution and 
# advanced structural constraints for RNA folding.

def find_wheels():
    wheels_to_find = [
        "biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
    ]
    found_wheels = []
    input_path = Path('/kaggle/input')
    if input_path.exists():
        for wheel_name in wheels_to_find:
            matches = list(input_path.rglob(wheel_name))
            if matches:
                found_wheels.append(str(matches[0]))
            else:
                # Fallback to hardcoded common paths if rglob fails or for local dev
                hardcoded_paths = [
                    f"/kaggle/input/datasets/kami1976/biopython-cp312/{wheel_name}",
                    f"/kaggle/input/datasets/amirrezaaleyasin/biotite/{wheel_name}",
                    f"/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/{wheel_name}"
                ]
                for hp in hardcoded_paths:
                    if os.path.exists(hp) and wheel_name in hp:
                        found_wheels.append(hp)
                        break
    return found_wheels

wheels = find_wheels()
if not wheels:
    # Very fallback
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
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

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
    if "centre_atom_mask" in f: return (f["centre_atom_mask"] == 1).bool()
    if "center_atom_mask" in f: return (f["center_atom_mask"] == 1).bool()
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
    mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
    if abs(mask11.sum().item() - n_tokens) < abs(mask12.sum().item() - n_tokens):
        return mask11
    return mask12

# ── Dynamic Path Resolution ──────────────────────────────────────────────────
def find_data_file(pattern, default):
    input_path = Path('/kaggle/input')
    if input_path.exists():
        matches = list(input_path.rglob(pattern))
        if matches:
            return str(matches[0])
    return default

DATA_BASE = "/kaggle/input/stanford-rna-3d-folding-2"
TEST_CSV_PATH = find_data_file("test_sequences.csv", f"{DATA_BASE}/test_sequences.csv")
TRAIN_CSV_PATH = find_data_file("train_sequences.csv", f"{DATA_BASE}/train_sequences.csv")
TRAIN_LBLS_PATH = find_data_file("train_labels.csv", f"{DATA_BASE}/train_labels.csv")
VAL_CSV_PATH = find_data_file("validation_sequences.csv", f"{DATA_BASE}/validation_sequences.csv")
VAL_LBLS_PATH = find_data_file("validation_labels.csv", f"{DATA_BASE}/validation_labels.csv")
DEFAULT_OUTPUT = "/kaggle/working/submission.csv"

# Protenix Code Dir resolution
def find_protenix_dir():
    input_path = Path('/kaggle/input')
    if input_path.exists():
        matches = list(input_path.rglob('runner/inference.py'))
        if matches:
            return str(matches[0].parent.parent)
    return "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"

DEFAULT_CODE_DIR = find_protenix_dir()
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME = "protenix_base_20250630_v1.0.0"
N_SAMPLE = 5
SEED = 42
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))
USE_PROTENIX = True

def parse_bool(value: str, default: bool = False) -> str:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}: return "true"
    if v in {"0", "false", "f", "no", "n", "off"}: return "false"
    return "true" if default else "false"

USE_MSA = parse_bool(os.environ.get("USE_MSA", "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA = parse_bool(os.environ.get("USE_RNA_MSA", "true"))
MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

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
    test_csv = os.environ.get("TEST_CSV", TEST_CSV_PATH)
    output_csv = os.environ.get("SUBMISSION_CSV", DEFAULT_OUTPUT)
    code_dir = os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR)
    root_dir = os.environ.get("PROTENIX_ROOT_DIR", DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

def ensure_required_files(root_dir: str) -> bool:
    # Dynamic search for checkpoint
    ckpt_path = Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt"
    if not ckpt_path.exists():
        matches = list(Path('/kaggle/input').rglob(f"{MODEL_NAME}.pt"))
        if matches: ckpt_path = matches[0]
        else: print(f"Missing checkpoint: {ckpt_path}"); return False
    
    ccd_path = Path(root_dir) / "common" / "components.cif"
    if not ccd_path.exists():
        matches = list(Path('/kaggle/input').rglob("components.cif"))
        if matches: ccd_path = matches[0]
        else: print(f"Missing CCD file: {ccd_path}"); return False
        
    return True

def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [{"name": row["target_id"], "covalent_bonds": [], "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}]} for _, row in df.iterrows()]
    with open(json_path, "w", encoding="utf-8") as f: json.dump(data, f)

def build_configs(input_json_path: str, dump_dir: str, model_name: str):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    base = {**configs_base, **{"data": data_configs}, **inference_configs}
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
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P, Q):
    centroid_P, centroid_Q = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - centroid_P, Q - centroid_Q
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return R, centroid_Q - R @ centroid_P

def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
    if len(chunk_coords_list) == 1:
        c = chunk_coords_list[0]
        if c.shape[0] >= seq_len: return c[:seq_len]
        o = np.zeros((seq_len, 3), dtype=c.dtype)
        o[:c.shape[0]] = c; return o
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        ps, pe = chunk_ranges[i-1]; cs, ce = chunk_ranges[i]
        os, oe = cs, min(pe, ce); ol = oe - os
        if ol < 3: aligned.append(chunk_coords_list[i].copy()); continue
        pov, cov = aligned[i-1][os-ps:oe-ps], chunk_coords_list[i][os-cs:oe-cs]
        valid = ~(np.isnan(pov).any(axis=1) | np.isnan(cov).any(axis=1))
        if valid.sum() < 3: aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cov[valid], pov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full, weights = np.zeros((seq_len, 3), dtype=np.float64), np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        cl = coords.shape[0]; ae = min(s + cl, seq_len); ul = ae - s; w = np.ones(ul, dtype=np.float64)
        if i > 0:
            os, oe = s, min(chunk_ranges[i-1][1], e); rl = oe - os
            if rl > 0: w[:rl] = np.linspace(0.0, 1.0, rl)
        if i < len(chunk_ranges) - 1:
            ns = chunk_ranges[i+1][0]; rs, rl = ns - s, ae - ns
            if rl > 0 and rs < ul: w[rs:ul] = np.linspace(1.0, 0.0, rl)
        full[s:ae] += coords[:ul] * w[:, None]; weights[s:ae] += w
    m = weights > 0; full[m] /= weights[m, None]
    full = adaptive_rna_constraints(full, "stitched", {}, confidence=0.5, passes=2)
    for i in range(seq_len - 1):
        d = full[i+1] - full[i]; dist = np.linalg.norm(d) + 1e-12
        if dist > 1.61: adj = d * ((1.61 - dist) / dist) * 0.5; full[i] -= adj; full[i+1] += adj
    return full

def _make_aligner():
    al = PairwiseAligner(); al.mode = "global"; al.match_score = 2; al.mismatch_score = -1.5
    al.open_gap_score = al.query_left_open_gap_score = al.query_right_open_gap_score = al.target_left_open_gap_score = al.target_right_open_gap_score = -8
    al.extend_gap_score = al.query_left_extend_gap_score = al.query_right_extend_gap_score = al.target_left_extend_gap_score = al.target_right_extend_gap_score = -0.4
    return al

_aligner = _make_aligner()

def parse_stoichiometry(stoich):
    if pd.isna(stoich) or not str(stoich).strip(): return []
    return [(p.split(":")[0].strip(), int(p.split(":")[1])) for p in str(stoich).split(";")]

def parse_fasta(fasta):
    out, cur, pts = {}, None, []
    for l in str(fasta).splitlines():
        l = l.strip()
        if not l: continue
        if l.startswith(">"):
            if cur: out[cur] = "".join(pts)
            cur, pts = l[1:].split()[0], []
        else: pts.append(l.replace(" ", ""))
    if cur: out[cur] = "".join(pts)
    return out

def get_chain_segments(row):
    s, st, al = row["sequence"], row.get("stoichiometry", ""), row.get("all_sequences", "")
    if pd.isna(st) or pd.isna(al) or not str(st).strip() or not str(al).strip(): return [(0, len(s))]
    try:
        cd, order = parse_fasta(al), parse_stoichiometry(st)
        segs, pos = [], 0
        for ch, cnt in order:
            b = cd.get(ch)
            if not b: return [(0, len(s))]
            for _ in range(cnt): segs.append((pos, pos + len(b))); pos += len(b)
        return segs if pos == len(s) else [(0, len(s))]
    except: return [(0, len(s))]

def build_segments_map(df):
    sm, stm = {}, {}
    for _, r in df.iterrows():
        sm[r["target_id"]] = get_chain_segments(r)
        stm[r["target_id"]] = "" if pd.isna(r.get("stoichiometry")) else str(r.get("stoichiometry"))
    return sm, stm

def process_labels(df):
    c = {}
    p = df["ID"].str.rsplit("_", n=1).str[0]
    for pref, g in df.groupby(p): c[pref] = g.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return c

def adapt_template_to_query(qs, ts, tc):
    aln = next(iter(_aligner.align(qs, ts)))
    nc = np.full((len(qs), 3), np.nan)
    for (q_s, q_e), (t_s, t_e) in zip(*aln.aligned):
        if (t_e - t_s) == (q_e - q_s): nc[q_s:q_e] = tc[t_s:t_e]
    for i in range(len(nc)):
        if np.isnan(nc[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(nc[j,0])), -1)
            nv = next((j for j in range(i+1, len(nc)) if not np.isnan(nc[j,0])), -1)
            if pv >= 0 and nv >= 0: w = (i-pv)/(nv-pv); nc[i] = (1-w)*nc[pv] + w*nc[nv]
            elif pv >= 0: nc[i] = nc[pv] + [3,0,0]
            elif nv >= 0: nc[i] = nc[nv] + [3,0,0]
            else: nc[i] = [i*3,0,0]
    return np.nan_to_num(nc)

def detect_simple_stems(coords):
    L = len(coords)
    if L < 10: return []
    pairs, dt = [], 12.5
    for i in range(L):
        for j in range(i+4, L):
            if np.linalg.norm(coords[i]-coords[j]) < dt: pairs.append((i,j))
    stems = []
    if not pairs: return stems
    pairs.sort(); used = [False]*len(pairs)
    for i in range(len(pairs)):
        if used[i]: continue
        stem = [pairs[i]]; used[i] = True; ci, cj = pairs[i]
        for k in range(i+1, len(pairs)):
            if not used[k]:
                ni, nj = pairs[k]
                if ni == ci+1 and nj == cj-1: stem.append(pairs[k]); used[k] = True; ci, cj = ni, nj
        if len(stem) >= 3: stems.append(stem)
    return stems

def adaptive_rna_constraints(coords, tid, sm, confidence=1.0, passes=2):
    X, strength = coords.copy(), max(0.75*(1.0-min(confidence, 0.97)), 0.02)
    segs = sm.get(tid, [(0, len(X))]); stems = detect_simple_stems(X)
    for _ in range(passes):
        if stems:
            bpp_s, vbpr_s, cspp_s = 0.15*strength, 0.22*strength, 0.25*strength
            for stem in stems:
                for i_s, j_s in stem:
                    if 0 < i_s < len(X)-1 and 0 < j_s < len(X)-1:
                        n_i = np.cross(X[i_s+1]-X[i_s], X[i_s-1]-X[i_s]); n_i /= (np.linalg.norm(n_i)+1e-6)
                        n_j = np.cross(X[j_s+1]-X[j_s], X[j_s-1]-X[j_s]); n_j /= (np.linalg.norm(n_j)+1e-6)
                        dot = np.dot(n_i, n_j)
                        if abs(dot) < 0.98:
                            tn = n_i if dot > 0 else -n_i; adj = (tn-n_j)*bpp_s
                            X[j_s+1]+=adj; X[j_s-1]+=adj; X[i_s+1]-=adj; X[i_s-1]-=adj
                for k in range(len(stem)-1):
                    i1, j1 = stem[k]; i2, j2 = stem[k+1]; c1, c2 = (X[i1]+X[j1])*0.5, (X[i2]+X[j2])*0.5
                    dv = c2 - c1; dl = np.linalg.norm(dv)+1e-6
                    if 1.5 < dl < 5.0: adj = dv*((2.8-dl)/dl)*vbpr_s; X[i1]-=adj*0.25; X[j1]-=adj*0.25; X[i2]+=adj*0.25; X[j2]+=adj*0.25
                for i, j in stem:
                    tjm = j-3
                    if 0 <= tjm < len(X):
                        dv = X[i]-X[tjm]; dl = np.linalg.norm(dv)+1e-6
                        if 11.0 < dl < 21.0: adj = dv*((15.5-dl)/dl)*cspp_s; X[i]+=adj*0.5; X[tjm]-=adj*0.5
                    tjm = j+2
                    if 0 <= tjm < len(X):
                        dv = X[i]-X[tjm]; dl = np.linalg.norm(dv)+1e-6
                        if 7.5 < dl < 14.5: adj = dv*((10.2-dl)/dl)*cspp_s; X[i]+=adj*0.5; X[tjm]-=adj*0.5
        for s, e in segs:
            C, L = X[s:e], e-s
            if L < 2: continue
            in_s = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_s[i_s-s] = True
                        if s <= j_s < e: in_s[j_s-s] = True
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d*((5.95-dist)/dist)[:, None]*(0.22*strength); C[:-1]-=adj; C[1:]+=adj
            rpc_m = in_s[:-1] & in_s[1:]
            if np.any(rpc_m):
                adj_rpc = d[rpc_m]*((5.90-dist[rpc_m])/dist[rpc_m])[:, None]*(0.32*strength)
                C[:-1][rpc_m]-=adj_rpc*0.5; C[1:][rpc_m]+=adj_rpc*0.5
            X[s:e] = C
    return X

def generate_rna_structure(seq, seed=None):
    if seed is not None: np.random.seed(seed)
    n = len(seq); c = np.zeros((n,3))
    for i in range(n): a = i*0.6; c[i] = [10.0*np.cos(a), 10.0*np.sin(a), i*2.5]
    return c

def main():
    global USE_PROTENIX
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    if not os.path.isdir(code_dir): USE_PROTENIX = False
    else:
        os.environ["PROTENIX_ROOT_DIR"] = root_dir; sys.path.append(code_dir)
        if not ensure_required_files(root_dir): USE_PROTENIX = False
    seed_everything(SEED)
    t_df = pd.read_csv(test_csv if IS_KAGGLE else (VAL_CSV_PATH if os.path.exists(VAL_CSV_PATH) else TRAIN_CSV_PATH))
    if not IS_KAGGLE and LOCAL_N_SAMPLES: t_df = t_df.tail(LOCAL_N_SAMPLES).reset_index(drop=True)
    tr_seqs = pd.concat([pd.read_csv(TRAIN_CSV_PATH), pd.read_csv(VAL_CSV_PATH)], ignore_index=True)
    tr_coords = process_labels(pd.concat([pd.read_csv(TRAIN_LBLS_PATH), pd.read_csv(VAL_LBLS_PATH)], ignore_index=True))
    sm, _ = build_segments_map(t_df)
    
    t_preds, p_queue = {}, {}
    for _, r in t_df.iterrows():
        tid, seq = r["target_id"], r["sequence"]; segs = sm.get(tid, [(0, len(seq))])
        similar = []
        for _, tr in tr_seqs.iterrows():
            if tid == tr["target_id"] or tid not in tr_coords: continue
            if abs(len(tr["sequence"])-len(seq))/max(len(tr["sequence"]), len(seq)) > 0.3: continue
            aln = next(iter(_aligner.align(seq, tr["sequence"]))); sim = aln.score/(2*min(len(seq), len(tr["sequence"])))
            if sim >= MIN_SIMILARITY: similar.append((tr["target_id"], tr["sequence"], sim, tr_coords[tr["target_id"]]))
        similar.sort(key=lambda x: x[2], reverse=True)
        preds = []
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords) in enumerate(similar[:30]):
            if len(preds) >= N_SAMPLE: break
            rng = np.random.default_rng((r.name*10000000000+i*10007)%(2**32))
            X = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            preds.append(adaptive_rna_constraints(X, tid, sm, confidence=sim))
        t_preds[tid] = preds
        if len(preds) < N_SAMPLE: p_queue[tid] = (N_SAMPLE-len(preds), seq)

    p_preds = {}
    if p_queue and USE_PROTENIX:
        wd = Path("/kaggle/working"); wd.mkdir(parents=True, exist_ok=True); tasks, ci = [], {}
        for tid, (n, s) in p_queue.items():
            if len(s) <= MAX_SEQ_LEN: tasks.append({"target_id": tid, "sequence": s}); ci[tid] = [{"name": tid, "range": (0, len(s))}]
            else:
                chunks = split_into_chunks(len(s), MAX_SEQ_LEN, CHUNK_OVERLAP); ci[tid] = []
                for idx, (cs, ce) in enumerate(chunks):
                    cn = f"{tid}_chunk{idx}"; tasks.append({"target_id": cn, "sequence": s[cs:ce]}); ci[tid].append({"name": cn, "range": (cs, ce)})
        ij = str(wd/"input.json"); build_input_json(pd.DataFrame(tasks), ij)
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        from protenix.data.inference.infer_dataloader import InferenceDataset
        cfg = update_gpu_compatible_configs(build_configs(ij, str(wd/"outputs"), MODEL_NAME))
        run, ds = InferenceRunner(cfg), InferenceDataset(cfg); raw_p = {}
        for i in range(len(ds)):
            data, aa, err = ds[i]; sn = data.get("sample_name", "")
            if err: raw_p[sn] = None; continue
            tid_b = sn.split("_chunk")[0]; n_n = p_queue.get(tid_b, (N_SAMPLE, ""))[0]
            try:
                new_c = update_inference_configs(cfg, data["N_token"].item()); new_c.sample_diffusion.N_sample = n_n; run.update_model_configs(new_c)
                pred = run.predict(data); rc = pred["coordinate"]
                f = data["input_feature_dict"]
                if "centre_atom_mask" in f: m = (f["centre_atom_mask"] == 1).to(rc.device)
                else: m11 = (f["atom_to_tokatom_idx"] == 11); m = m11 if abs(m11.sum()-data["N_token"].item()) < abs((f["atom_to_tokatom_idx"] == 12).sum()-data["N_token"].item()) else (f["atom_to_tokatom_idx"] == 12)
                raw_p[sn] = rc[:, m.to(rc.device), :].detach().cpu().numpy()
            except: raw_p[sn] = None
            finally: gc.collect(); torch.cuda.empty_cache()
        for tid, (n, s) in p_queue.items():
            chunks = ci.get(tid, [])
            if len(chunks) == 1: p_preds[tid] = raw_p.get(tid)
            else:
                stitched = []
                for si in range(n):
                    cl, rl = [], []
                    for cinfo in chunks:
                        cc = raw_p.get(cinfo["name"])
                        if cc is None: break
                        cl.append(cc[si] if si < cc.shape[0] else cc[-1]); rl.append(cinfo["range"])
                    if len(cl) == len(chunks): stitched.append(stitch_chunk_coords(cl, rl, len(s)))
                p_preds[tid] = np.stack(stitched, axis=0) if stitched else None

    all_r = []
    for _, r in t_df.iterrows():
        tid, seq = r["target_id"], r["sequence"]; comb = list(t_preds.get(tid, []))
        ptx = p_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(comb) >= N_SAMPLE: break
                comb.append(ptx[j])
        while len(comb) < N_SAMPLE: comb.append(adaptive_rna_constraints(generate_rna_structure(seq), tid, sm, confidence=0.2))
        all_r.extend(coords_to_rows(tid, seq, np.stack(comb[:N_SAMPLE], axis=0)))
    sub = pd.DataFrame(all_r)
    cc = [c for c in sub.columns if c.startswith(("x_", "y_", "z_"))]
    sub[cc] = sub[cc].clip(-999.999, 9999.999); sub.to_csv(output_csv, index=False)

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        o = os.environ.get("SUBMISSION_CSV", "/kaggle/working/submission.csv")
        if not os.path.exists(o): pd.DataFrame(columns=["ID","resname","resid"]+[f"{ax}_{s+1}" for s in range(5) for ax in "xyz"]).to_csv(o, index=False)
