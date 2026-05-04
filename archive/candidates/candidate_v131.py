import os
import sys
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ── Install dependencies ─────────────────────────────────────────────────────
def install_deps():
    deps = [
        "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
    ]
    for d in deps:
        if os.path.exists(d):
            os.system(f"pip install -q --no-index --no-deps {d}")

install_deps()

# ── Global State ─────────────────────────────────────────────────────────────
GLOBAL_SEQ_MAP = {}

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True
LOCAL_N_SAMPLES = None

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
MAX_SEQ_LEN   = 512
CHUNK_OVERLAP = 128

MIN_SIMILARITY       = 0.0
MIN_PERCENT_IDENTITY = 50.0
USE_PROTENIX         = True

USE_MSA      = "false"
USE_TEMPLATE = "false"
USE_RNA_MSA  = "true"
MODEL_N_SAMPLE = N_SAMPLE

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
            raise FileNotFoundError(f"Missing {name}: {p}")

def kabsch_align(P: np.ndarray, Q: np.ndarray):
    """Compute optimal rotation R and translation t so that R @ P + t ≈ Q."""
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

def calculate_rmsd(P, Q):
    L = min(len(P), len(Q))
    P, Q = P[:L], Q[:L]
    valid = ~(np.isnan(P).any(axis=1) | np.isnan(Q).any(axis=1))
    if valid.sum() < 3: return 99.9
    R, t = kabsch_align(P[valid], Q[valid])
    P_aligned = (P @ R.T) + t
    return np.sqrt(np.mean(np.linalg.norm(P_aligned[valid] - Q[valid], axis=1)**2))

# ─────────────── Protenix Config Helpers ─────────────────────────────────────
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
    if seq_len <= max_len: return [(0, seq_len)]
    chunks = []
    step, pos = max_len - overlap, 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len: return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:coords.shape[0]] = coords
        return out

    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]
        ov_start, ov_end = cur_start, min(prev_end, cur_end)
        ov_len = ov_end - ov_start
        if ov_len < 3:
            aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
        cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]
        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3:
            aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)

    full, weights = np.zeros((seq_len, 3)), np.zeros(seq_len)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        c_len = coords.shape[0]
        act_e = min(s + c_len, seq_len)
        used_len = act_e - s
        w = np.ones(used_len)
        if i > 0:
            ramp_len = min(chunk_ranges[i - 1][1], e) - s
            if ramp_len > 0: w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_len = act_e - next_s
            if ramp_len > 0 and next_s - s < used_len: w[next_s - s:] = np.linspace(1.0, 0.0, ramp_len)
        full[s:act_e] += coords[:used_len] * w[:, None]
        weights[s:act_e] += w
    mask = weights > 0
    full[mask] /= weights[mask, None]
    return full

# ─────────────── TBM Core Functions ──────────────────────────────────────────
_aligner = PairwiseAligner()
_aligner.mode = "global"
_aligner.match_score, _aligner.mismatch_score = 2, -1.5
_aligner.open_gap_score, _aligner.extend_gap_score = -8, -0.4

def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or str(stoich).strip() == "": return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";") for ch, cnt in [part.split(":")]]

def parse_fasta(fasta: str) -> dict:
    out, cur, parts = {}, None, []
    for line in str(fasta).splitlines():
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
        cd, order = parse_fasta(all_sq), parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = cd.get(ch)
            if not base: return [(0, len(seq))]
            for _ in range(cnt): segs.append((pos, pos + len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except: return [(0, len(seq))]

def build_segments_map(df: pd.DataFrame) -> tuple:
    seg_map = {}
    for _, r in df.iterrows(): seg_map[r["target_id"]] = get_chain_segments(r)
    return seg_map

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar_sequences_detailed(q_seq, train_df, train_coords, top_n=30):
    results = []
    for _, row in train_df.iterrows():
        tid, t_seq = row["target_id"], row["sequence"]
        if tid not in train_coords or abs(len(t_seq) - len(q_seq)) / max(len(t_seq), len(q_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(q_seq, t_seq)))
        norm_s = aln.score / (2 * min(len(q_seq), len(t_seq)))
        identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if q_seq[qp] == t_seq[tp])
        results.append((tid, t_seq, norm_s, train_coords[tid], 100 * identical / len(q_seq)))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def adapt_template_to_query(q_seq, t_seq, t_coords) -> np.ndarray:
    aln = next(iter(_aligner.align(q_seq, t_seq)))
    new = np.full((len(q_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        if te - ts == qe - qs: new[qs:qe] = t_coords[ts:te]
    for i in range(len(new)):
        if np.isnan(new[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new[j, 0])), -1)
            nv = next((j for j in range(i+1, len(new)) if not np.isnan(new[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new[i] = (1 - w) * new[pv] + w * new[nv]
            elif pv >= 0: new[i] = new[pv] + [3, 0, 0]
            elif nv >= 0: new[i] = new[nv] + [3, 0, 0]
            else: new[i] = [i * 3, 0, 0]
    return np.nan_to_num(new)

# ─────────────── Refinement Logic ────────────────────────────────────────────
def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
    X = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    
    seq = GLOBAL_SEQ_MAP.get(target_id, "")
    pairs = []
    if seq:
        n = len(seq)
        for i in range(n):
            for j in range(i + 4, n):
                if (seq[i], seq[j]) in [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]:
                    score = sum(1 for k in range(1, 4) if i+k < j-k and (seq[i+k], seq[j-k]) in [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')])
                    if score >= 2: pairs.append((i, j))

    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # i-i+1 bond (A-form trace ~5.95 A)
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.25 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            # i-i+2 soft bond (~10.2 A)
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.12 * strength)
            C[:-2] -= adj2; C[2:] += adj2

            # NEW: i-i+3 helical stabilization (~13.0 A)
            if L > 3:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                adj3 = d3 * ((13.0 - d3n) / d3n)[:, None] * (0.08 * strength)
                C[:-3] -= adj3; C[3:] += adj3

            # Local smoothing
            C[1:-1] += (0.08 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            
            # Base-pairing constraints
            for i, j in pairs:
                if s <= i < e and s <= j < e:
                    pi, pj = i-s, j-s
                    d_ij = C[pj] - C[pi]; dist = np.linalg.norm(d_ij) + 1e-6
                    adj_ij = d_ij * ((10.5 - dist) / dist) * (0.15 * strength)
                    C[pi] -= adj_ij; C[pj] += adj_ij

            # Self-avoidance
            if L >= 20:
                P = C; diff = P[:, None, :] - P[None, :, :]
                dm = np.linalg.norm(diff, axis=2) + 1e-6
                sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
                mask = (sep > 4) & (dm < 4.5)
                if np.any(mask):
                    vec = (diff * ((4.5 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C += (0.02 * strength) * vec
            X[s:e] = C
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [9.2 * np.cos(ang), 9.2 * np.sin(ang), i * 2.8]
    return coords

# ─────────────── TBM Phase ───────────────────────────────────────────────────
def tbm_phase(test_df, train_df, train_coords, segments_map):
    print(f"\nPHASE 1: TBM")
    t0, t_preds, p_queue = time.time(), {}, {}
    for _, r in test_df.iterrows():
        tid, seq = r["target_id"], r["sequence"]
        similar = find_similar_sequences_detailed(seq, train_df, train_coords, top_n=30)
        preds, used = [], set()
        for i, (tmpl_id, _, sim, tmpl_coords, pct_id) in enumerate(similar):
            if len(preds) >= N_SAMPLE or sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY: break
            if tmpl_id in used: continue
            rng = np.random.default_rng((r.name * 1000 + i) % (2**32))
            X = adapt_template_to_query(seq, "", tmpl_coords)
            preds.append(adaptive_rna_constraints(X, tid, segments_map, confidence=sim))
            used.add(tmpl_id)
        t_preds[tid] = preds
        if len(preds) < N_SAMPLE: p_queue[tid] = (N_SAMPLE - len(preds), seq)
    print(f"Phase 1 done in {time.time()-t0:.1f}s. Need Protenix for {len(p_queue)} targets.")
    return t_preds, p_queue

# ─────────────── Main ────────────────────────────────────────────────────────
def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    sys.path.append(code_dir)
    ensure_required_files(root_dir)
    seed_everything(SEED)

    test_df = pd.read_csv(test_csv)
    if not IS_KAGGLE: test_df = test_df.head(LOCAL_N_SAMPLES or 5)
    
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs = pd.read_csv(DEFAULT_VAL_CSV)
    combined_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True)
    combined_lbls = pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True)
    train_coords = process_labels(combined_lbls)
    segments_map = build_segments_map(test_df)

    GLOBAL_SEQ_MAP.update(dict(zip(test_df["target_id"], test_df["sequence"])))
    GLOBAL_SEQ_MAP.update(dict(zip(combined_seqs["target_id"], combined_seqs["sequence"])))

    # ── Local Validation ─────────────────────────────────────────────────────
    print("\nRunning Local Validation …")
    val_ids = combined_seqs["target_id"].unique()[:10]
    rmsd_list = []
    for vid in val_ids:
        if vid not in train_coords: continue
        true_c, v_seq = train_coords[vid], GLOBAL_SEQ_MAP[vid]
        pred = generate_rna_structure(v_seq, seed=42)
        pred = adaptive_rna_constraints(pred, vid, {vid: [(0, len(v_seq))]}, confidence=0.1)
        rmsd_list.append(calculate_rmsd(pred, true_c))
    if rmsd_list: print(f"LOCAL_VALIDATION_RMSD: {np.mean(rmsd_list):.5f}")

    # ── Prediction Pipeline ──────────────────────────────────────────────────
    t_preds, p_queue = tbm_phase(test_df, combined_seqs, train_coords, segments_map)
    p_preds = {}

    if p_queue and USE_PROTENIX:
        tasks = []
        for tid, (n_needed, full_seq) in p_queue.items():
            if len(full_seq) <= MAX_SEQ_LEN: tasks.append({"target_id": tid, "sequence": full_seq})
            else:
                for ci, (cs, ce) in enumerate(split_into_chunks(len(full_seq), MAX_SEQ_LEN, CHUNK_OVERLAP)):
                    tasks.append({"target_id": f"{tid}_chunk{ci}", "sequence": full_seq[cs:ce]})
        
        work_dir = Path("/kaggle/working")
        input_json = str(work_dir / "protenix_input.json")
        build_input_json(pd.DataFrame(tasks), input_json)

        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        
        configs = update_gpu_compatible_configs(build_configs(input_json, str(work_dir / "outputs"), MODEL_NAME))
        runner, dataset = InferenceRunner(configs), InferenceDataset(configs)
        raw_res = {}

        for i in tqdm(range(len(dataset)), desc="Protenix"):
            data, _, err = dataset[i]
            if err: continue
            s_name = data.get("sample_name", f"s_{i}")
            tid = s_name.split("_chunk")[0]
            n_n = p_queue.get(tid, (N_SAMPLE, ""))[0]
            try:
                new_cfg = update_inference_configs(configs, data["N_token"].item())
                new_cfg.sample_diffusion.N_sample = n_n
                runner.update_model_configs(new_cfg)
                pred = runner.predict(data)
                mask = (data["input_feature_dict"]["atom_to_tokatom_idx"] == 11) # heuristic for C1'
                raw_res[s_name] = pred["coordinate"][:, mask, :].detach().cpu().numpy()
            except: pass
            finally: gc.collect(); torch.cuda.empty_cache()

        for tid, (n_needed, full_seq) in p_queue.items():
            chunks = [raw_res.get(f"{tid}_chunk{ci}") for ci in range(len(split_into_chunks(len(full_seq), MAX_SEQ_LEN, CHUNK_OVERLAP)))]
            if all(c is not None for c in chunks):
                p_preds[tid] = np.stack([stitch_chunk_coords([c[s] for c in chunks], split_into_chunks(len(full_seq), MAX_SEQ_LEN, CHUNK_OVERLAP), len(full_seq)) for s in range(n_needed)])

    all_rows = []
    for _, r in test_df.iterrows():
        tid, seq = r["target_id"], r["sequence"]
        comb = list(t_preds.get(tid, []))
        ptx = p_preds.get(tid)
        if ptx is not None:
            for j in range(ptx.shape[0]):
                if len(comb) >= N_SAMPLE: break
                comb.append(ptx[j])
        while len(comb) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=SEED + len(comb))
            comb.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.1))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(comb[:N_SAMPLE], axis=0)))

    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, N_SAMPLE+1) for c in ["x","y","z"]]
    sub[cols].to_csv(output_csv, index=False)
    print(f"✓ Saved submission to {output_csv}")

if __name__ == "__main__":
    main()
