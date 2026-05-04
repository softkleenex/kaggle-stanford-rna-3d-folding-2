import os
import sys

# Convert Jupyter magic !pip to os.system for .py compatibility
# Replicated from Top 1 Solution (v27) logic with Kabsch alignment and chunk stitching.
# Strategy: Replicating 0.448 scoring baseline and fixing the automated planning [object Object] error.

os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True #os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") == "Batch"
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ─────────────── Paths & Constants ───────────────────────────────────────────
def find_path(name, is_dir=False):
    for p in Path("/kaggle/input").rglob(name):
        if is_dir and p.is_dir(): return str(p)
        if not is_dir and p.is_file(): return str(p)
    return None

test_csv_path = find_path("test_sequences.csv")
DATA_BASE = os.path.dirname(test_csv_path) if test_csv_path else "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV   = find_path("test_sequences.csv") or f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV  = find_path("train_sequences.csv") or f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS = find_path("train_labels.csv") or f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV    = find_path("validation_sequences.csv") or f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS   = find_path("validation_labels.csv") or f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT     = "/kaggle/working/submission.csv"

protenix_base = find_path("protenix-v1-adjusted", is_dir=True)
if protenix_base:
    protenix_dirs = list(Path(protenix_base).rglob("Protenix-v1"))
    DEFAULT_CODE_DIR = str(protenix_dirs[-1]) if protenix_dirs else protenix_base
else:
    DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = 512
CHUNK_OVERLAP = 128
USE_PROTENIX  = True

# TBM Constants
MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0

# ─────────────── General Utilities ───────────────────────────────────────────
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
        
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
    mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
    
    if abs(mask11.sum().item() - n_tokens) < abs(mask12.sum().item() - n_tokens):
        return mask11
    else:
        return mask12

def kabsch_align(P: np.ndarray, Q: np.ndarray):
    """Compute optimal rotation R and translation t so that R @ P + t ≈ Q."""
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = cQ - R @ cP
    return R, t

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:min(len(coords), seq_len)] = coords[:min(len(coords), seq_len)]
        return out

    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        p_s, p_e = chunk_ranges[i-1]
        c_s, c_e = chunk_ranges[i]
        ov_s, ov_e = c_s, min(p_e, c_e)
        ov_len = ov_e - ov_s
        if ov_len < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue
        prev_ov = aligned[i-1][ov_s - p_s : ov_e - p_s]
        cur_ov  = chunk_coords_list[i][ov_s - c_s : ov_e - c_s]
        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)

    full = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        actual_end = min(s + coords.shape[0], seq_len)
        used_len = actual_end - s
        w = np.ones(used_len, dtype=np.float64)
        if i > 0:
            ramp_len = min(chunk_ranges[i-1][1], e) - s
            if ramp_len > 0: w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if i < len(chunk_ranges) - 1:
            ramp_start = chunk_ranges[i+1][0] - s
            ramp_len = actual_end - chunk_ranges[i+1][0]
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)
        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w
    mask = weights > 0
    full[mask] /= weights[mask, None]
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

def coords_to_rows(target_id, seq, coords):
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i+1}", "resname": seq[i], "resid": i+1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0]: row[f"x_{s+1}"], row[f"y_{s+1}"], row[f"z_{s+1}"] = coords[s, i]
            else: row[f"x_{s+1}"], row[f"y_{s+1}"], row[f"z_{s+1}"] = 0, 0, 0
        rows.append(row)
    return rows

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X = coords.copy(); segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            X[s:e] = C
    return X

def adapt_template_to_query(query_seq, template_seq, template_coords):
    aln = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        new_coords[qs:qe] = template_coords[ts:te]
    idx = np.arange(len(new_coords)); mask = ~np.isnan(new_coords[:, 0])
    if not np.any(mask): return np.zeros_like(new_coords)
    for k in range(3): new_coords[:, k] = np.interp(idx, idx[mask], new_coords[mask, k])
    return new_coords

def build_segments_map(df):
    return {r["target_id"]: [(0, len(r["sequence"]))] for _, r in df.iterrows()}, {}

def split_into_chunks(seq_len, max_len, overlap):
    if seq_len <= max_len: return [(0, seq_len)]
    chunks = []; step = max_len - overlap; pos = 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len); chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    sys.path.append(code_dir); os.environ["PROTENIX_ROOT_DIR"] = root_dir
    seed_everything(SEED); test_df = pd.read_csv(test_csv)
    if LOCAL_N_SAMPLES: test_df = test_df.head(LOCAL_N_SAMPLES)
    
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV); val_seqs = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS); val_labels = pd.read_csv(DEFAULT_VAL_LBLS)
    comb_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True)
    comb_labels = pd.concat([train_labels, val_labels], ignore_index=True)
    prefixes = comb_labels["ID"].str.rsplit("_", n=1).str[0]
    train_coords = {p: g.sort_values("resid")[["x_1", "y_1", "z_1"]].values for p, g in comb_labels.groupby(prefixes)}
    seg_map, _ = build_segments_map(test_df)

    tbm_preds, ptx_queue = {}, {}
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        sims = []
        for _, tr in comb_seqs.iterrows():
            t_tid, t_seq = tr["target_id"], tr["sequence"]
            if t_tid not in train_coords: continue
            score = _aligner.score(seq, t_seq)
            norm_s = score / (2 * min(len(seq), len(t_seq)))
            if norm_s >= MIN_SIMILARITY: sims.append((t_tid, t_seq, norm_s, train_coords[t_tid]))
        sims.sort(key=lambda x: x[2], reverse=True)
        preds = []
        for i, (t_tid, t_seq, s, c) in enumerate(sims[:N_SAMPLE]):
            if s < MIN_SIMILARITY: break
            X = adapt_template_to_query(seq, t_seq, c)
            preds.append(adaptive_rna_constraints(X, tid, seg_map, confidence=s))
        tbm_preds[tid] = preds
        if len(preds) < N_SAMPLE: ptx_queue[tid] = (N_SAMPLE - len(preds), seq)

    ptx_preds = {}
    if ptx_queue and USE_PROTENIX:
        tasks = []; chunk_info = {}
        for tid, (n, full_seq) in ptx_queue.items():
            if len(full_seq) <= MAX_SEQ_LEN:
                tasks.append({"target_id": tid, "sequence": full_seq})
                chunk_info[tid] = [{"name": tid, "range": (0, len(full_seq))}]
            else:
                chunks = split_into_chunks(len(full_seq), MAX_SEQ_LEN, CHUNK_OVERLAP)
                chunk_info[tid] = []
                for ci, (cs, ce) in enumerate(chunks):
                    cname = f"{tid}_chunk{ci}"
                    tasks.append({"target_id": cname, "sequence": full_seq[cs:ce]})
                    chunk_info[tid].append({"name": cname, "range": (cs, ce)})
        
        input_json = "/kaggle/working/ptx_input.json"
        with open(input_json, "w") as f: json.dump([{"name": t["target_id"], "sequences": [{"rnaSequence": {"sequence": t["sequence"], "count": 1}}]} for t in tasks], f)
        
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from protenix.config.config import get_config
        # Note: In some environments build_configs is replaced by manual dict update
        # We use a simplified version for v180
        from configs.configs_base import configs as cb; from configs.configs_data import data_configs as cd; from configs.configs_inference import inference_configs as ci; from configs.configs_model_type import model_configs as cm
        cfg = {**cb, "data": cd, **ci}; cm_name = cm[MODEL_NAME]
        for k, v in cm_name.items():
            if isinstance(v, dict) and k in cfg: cfg[k].update(v)
            else: cfg[k] = v
        cfg["input_json_path"] = input_json; cfg["dump_dir"] = "/kaggle/working/outputs"; cfg["model_path"] = f"{root_dir}/models/{MODEL_NAME}.pt"
        
        from protenix.config.config import parse_configs
        p_cfg = parse_configs(configs=cfg, arg_str=f"--model_name {MODEL_NAME} --input_json_path {input_json} --dump_dir /kaggle/working/outputs --seeds {SEED}", fill_required_with_null=True)
        runner = InferenceRunner(p_cfg); dataset = InferenceDataset(p_cfg)
        raw_res = {}
        for i in tqdm(range(len(dataset))):
            d, _, err = dataset[i]; sname = d.get("sample_name", f"s_{i}")
            if err: raw_res[sname] = None; continue
            tid = sname.split("_chunk")[0] if "_chunk" in sname else sname
            try:
                r_cfg = update_inference_configs(p_cfg, d["N_token"].item())
                r_cfg.sample_diffusion.N_sample = ptx_queue.get(tid, (N_SAMPLE, ""))[0]
                runner.update_model_configs(r_cfg); pred = runner.predict(d)
                m = get_c1_mask(d, None).to(pred["coordinate"].device)
                raw_res[sname] = pred["coordinate"][:, m, :].detach().cpu().numpy()
            except Exception: raw_res[sname] = None
        
        for tid, (n, full_seq) in ptx_queue.items():
            chunks = chunk_info.get(tid, [])
            if not chunks: continue
            if len(chunks) == 1: ptx_preds[tid] = raw_res.get(tid)
            else:
                stitched = []
                for s in range(n):
                    cl, rl = [], []
                    for ci in chunks:
                        cr = raw_res.get(ci["name"])
                        if cr is not None: cl.append(cr[min(s, len(cr)-1)]); rl.append(ci["range"])
                    if len(cl) == len(chunks): stitched.append(stitch_chunk_coords(cl, rl, len(full_seq)))
                if stitched: ptx_preds[tid] = np.stack(stitched, axis=0)

    final_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        combined = list(tbm_preds.get(tid, []))
        ptx = ptx_preds.get(tid)
        if ptx is not None:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(ptx[j])
        while len(combined) < N_SAMPLE:
            dn = adaptive_rna_constraints(np.zeros((len(seq), 3)), tid, seg_map, confidence=0.0)
            combined.append(dn)
        final_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE], axis=0)))
    
    sub = pd.DataFrame(final_rows)
    sub.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

if __name__ == "__main__":
    main()
