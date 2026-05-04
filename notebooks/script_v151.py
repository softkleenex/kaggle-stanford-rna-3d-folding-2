# v151 | CANDIDATE | N/A | STRATEGY: Implement a soft i to i+3 C1'–C1' distance constraint (13.5 ± 1.0 Å) to enforce canonical A-form helical winding and complement the existing i+2 and i+4 constraints for improved backbone geometry.

import os
import sys
import pandas as pd
import numpy as np
import torch
import gc
import json
import time
from pathlib import Path
from Bio.Align import PairwiseAligner
from tqdm import tqdm

#Consolidate pip installs
os.system('pip install -q --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl')
os.system('pip install -q --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl')
os.system('pip install -q --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl')

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = True 
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
USE_PROTENIX = True

def parse_bool(value: str, default: bool = False) -> str:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}: return "true"
    if v in {"0", "false", "f", "no", "n", "off"}: return "false"
    return "true" if default else "false"

USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))
MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

# ─────────────── Utilities ───────────────────────────────────────────────────
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

def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [
        {
            "name": row["target_id"],
            "covalent_bonds": [],
            "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}],
        }
        for _, row in df.iterrows()
    ]
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

def get_feature_c1_mask(data: dict) -> torch.Tensor:
    f = data["input_feature_dict"]
    if "centre_atom_mask" in f: return f["centre_atom_mask"].long() == 1
    return f["atom_to_tokatom_idx"].long() == 12

def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0] and i < coords.shape[1]: x, y, z = coords[s, i]
            else: x, y, z = 0.0, 0.0, 0.0
            row[f"x_{s + 1}"], row[f"y_{s + 1}"], row[f"z_{s + 1}"] = float(x), float(y), float(z)
        rows.append(row)
    return rows

def split_into_chunks(seq_len: int, max_len: int, overlap: int) -> list:
    if seq_len <= max_len: return [(0, seq_len)]
    chunks = []; step = max_len - overlap; pos = 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len); chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P: np.ndarray, Q: np.ndarray):
    centroid_P, centroid_Q = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - centroid_P, Q - centroid_Q
    H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H); d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3); 
    if d < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T; t = centroid_Q - R @ centroid_P
    return R, t

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:min(coords.shape[0], seq_len)] = coords[:min(coords.shape[0], seq_len)]
        return out
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]; cur_start, cur_end = chunk_ranges[i]
        ov_start, ov_end = cur_start, min(prev_end, cur_end); ov_len = ov_end - ov_start
        if ov_len < 3: aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
        cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]
        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3: aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full = np.zeros((seq_len, 3), dtype=np.float64); weights = np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        actual_end = min(s + coords.shape[0], seq_len); used_len = actual_end - s; w = np.ones(used_len, dtype=np.float64)
        if i > 0:
            ramp_len = min(chunk_ranges[i - 1][1], e) - s
            if ramp_len > 0: w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if i < len(chunk_ranges) - 1:
            ramp_len = actual_end - chunk_ranges[i + 1][0]
            if ramp_len > 0: w[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)
        full[s:actual_end] += coords[:used_len] * w[:, None]; weights[s:actual_end] += w
    mask = weights > 0; full[mask] /= weights[mask, None]
    return full

# ─────────────── TBM & Constraints ───────────────────────────────────────────
_aligner = PairwiseAligner(); _aligner.mode = "global"; _aligner.match_score = 2; _aligner.mismatch_score = -1.5; _aligner.open_gap_score = -8; _aligner.extend_gap_score = -0.4

def get_secondary_structure(seq: str) -> list:
    n = len(seq); pairs = []; used = [False] * n
    for stem_len in range(16, 2, -1):
        for i in range(n - stem_len):
            for j in range(n - 1, i + stem_len + 3, -1):
                match = True
                for k in range(stem_len):
                    if i+k >= n or j-k < 0 or used[i+k] or used[j-k]: match = False; break
                    if (seq[i+k]+seq[j-k]).upper() not in ("AU", "UA", "CG", "GC", "GU", "UG"): match = False; break
                if match:
                    for k in range(stem_len): pairs.append((i+k, j-k)); used[i+k] = used[j-k] = True
    return pairs

def adaptive_rna_constraints(coords, target_id, sequence, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X = coords.copy(); L = len(X); segments = segments_map.get(target_id, [(0, L)])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    pairs = get_secondary_structure(sequence); res_to_seg = {}
    for s, e in segments:
        for i in range(s, e): res_to_seg[i] = (s, e)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; seg_L = e - s
            if seg_L < 3: continue
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            C[:-1] -= d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength); C[1:] += d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            diff2 = np.maximum(np.abs(d2n - 9.2) - 0.7, 0.0)
            adj2 = d2 * (np.sign(9.2 - d2n) * diff2 / d2n)[:, None] * (0.12 * strength); C[:-2] -= adj2; C[2:] += adj2
            # NEW v151: i to i+3 constraint (13.5 ± 1.0 Å)
            if seg_L >= 4:
                d3 = C[3:] - C[:-3]; d3n = np.linalg.norm(d3, axis=1) + 1e-6
                diff3 = np.maximum(np.abs(d3n - 13.5) - 1.0, 0.0)
                adj3 = d3 * (np.sign(13.5 - d3n) * diff3 / d3n)[:, None] * (0.08 * strength); C[:-3] -= adj3; C[3:] += adj3
            if seg_L >= 5:
                d4 = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((16.8 - d4n) / d4n)[:, None] * (0.05 * strength); C[:-4] -= adj4; C[4:] += adj4
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            if seg_L >= 25:
                idx = np.linspace(0, seg_L-1, min(seg_L, 160)).astype(int); P = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm = np.linalg.norm(diff, axis=2) + 1e-6; mask = (np.abs(idx[:, None] - idx[None, :]) > 2) & (dm < 3.2)
                if np.any(mask): C[idx] += (0.015 * strength) * (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
            X[s:e] = C
        for i, j in pairs:
            if (sequence[i]+sequence[j]).upper() in ("AU", "UA", "CG", "GC") and i in res_to_seg and j in res_to_seg:
                vec = X[j] - X[i]; d = np.linalg.norm(vec) + 1e-6
                if abs(d - 10.6) > 0.6: adj = vec * ((10.6 - d) / d) * (0.25 * strength); X[i] -= adj; X[j] += adj
    return X

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}; prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes): coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict or abs(len(tseq)-len(query_seq))/max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
        results.append((tid, tseq, aln.score/(2*min(len(query_seq), len(tseq))), train_coords_dict[tid], 100*identical/len(query_seq)))
    return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln = next(iter(_aligner.align(query_seq, template_seq))); new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        if (te-ts) == (qe-qs): new_coords[qs:qe] = template_coords[ts:te]
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i+1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0: X = (i-pv)/(nv-pv); new_coords[i] = (1-X)*new_coords[pv] + X*new_coords[nv]
            elif pv >= 0: new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0: new_coords[i] = new_coords[nv] + [3, 0, 0]
            else: new_coords[i] = [i*3, 0, 0]
    return np.nan_to_num(new_coords)

def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    template_predictions, protenix_queue = {}, {}
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]; segs = segments_map.get(tid, [(0, len(seq))])
        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30); preds, used = [], set()
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id) in enumerate(similar):
            if len(preds) >= N_SAMPLE or sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY: break
            if tmpl_id in used: continue
            rng = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
            X = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            if len(preds) == 2:
                longest = max(segs, key=lambda se: se[1]-se[0]); pivot = longest[0] + int(rng.integers(10, longest[1]-longest[0]-10))
                p0 = X[pivot].copy(); ang = np.deg2rad(float(rng.uniform(-22, 22))); axis = rng.normal(size=3); axis /= np.linalg.norm(axis)+1e-12
                c, s = np.cos(ang), np.sin(ang); R = np.array([[c+axis[0]**2*(1-c), axis[0]*axis[1]*(1-c)-axis[2]*s, axis[0]*axis[2]*(1-c)+axis[1]*s],[axis[1]*axis[0]*(1-c)+axis[2]*s, c+axis[1]**2*(1-c), axis[1]*axis[2]*(1-c)-axis[0]*s],[axis[2]*axis[0]*(1-c)-axis[1]*s, axis[2]*axis[1]*(1-c)+axis[0]*s, c+axis[2]**2*(1-c)]])
                X[pivot+1:longest[1]] = (X[pivot+1:longest[1]]-p0) @ R.T + p0
            elif len(preds) >= 3: X += rng.normal(0, 0.1, X.shape)
            preds.append(adaptive_rna_constraints(X, tid, seq, segments_map, confidence=sim)); used.add(tmpl_id)
        template_predictions[tid] = preds
        if len(preds) < N_SAMPLE: protenix_queue[tid] = (N_SAMPLE - len(preds), seq)
    return template_predictions, protenix_queue

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n): ang = i*0.6; coords[i] = [10.0*np.cos(ang), 10.0*np.sin(ang), i*2.5]
    return coords

# ─────────────── Main ────────────────────────────────────────────────────────
def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths(); sys.path.append(code_dir); ensure_required_files(root_dir); seed_everything(SEED)
    test_df_full = pd.read_csv(test_csv); test_df = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE else test_df_full).reset_index(drop=True)
    train_seqs, train_labels = pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_TRAIN_LBLS)
    try: val_seqs, val_labels = pd.read_csv(DEFAULT_VAL_CSV), pd.read_csv(DEFAULT_VAL_LBLS)
    except:
        ut = train_seqs["target_id"].unique(); val_targets = ut[-5:]; train_targets = ut[:-5]
        val_seqs = train_seqs[train_seqs["target_id"].isin(val_targets)].copy(); val_labels = train_labels[train_labels["ID"].str.rsplit("_", n=1).str[0].isin(val_targets)].copy()
        train_seqs = train_seqs[train_seqs["target_id"].isin(train_targets)].copy(); train_labels = train_labels[train_labels["ID"].str.rsplit("_", n=1).str[0].isin(train_targets)].copy()
    
    combined_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True); train_coords = process_labels(pd.concat([train_labels, val_labels], ignore_index=True))
    segments_map = {r["target_id"]: [(0, len(r["sequence"]))] for _, r in test_df.iterrows()}
    
    template_preds, protenix_queue = tbm_phase(test_df, combined_seqs, train_coords, segments_map)
    protenix_preds = {}
    if protenix_queue and USE_PROTENIX:
        work_dir = Path("/kaggle/working"); work_dir.mkdir(parents=True, exist_ok=True); input_json_path = str(work_dir/"input.json")
        build_input_json(pd.DataFrame([{"target_id": tid, "sequence": s} for tid, (_, s) in protenix_queue.items()]), input_json_path)
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        configs = build_configs(input_json_path, str(work_dir/"outputs"), MODEL_NAME); configs = update_gpu_compatible_configs(configs); runner = InferenceRunner(configs); dataset = InferenceDataset(configs)
        for i in tqdm(range(len(dataset)), desc="Protenix"):
            data, atom_array, err = dataset[i]; sample_name = data.get("sample_name", f"sample_{i}"); tid = sample_name
            if err: protenix_preds[tid] = None; continue
            try:
                slen = data["N_token"].item(); n_needed = protenix_queue.get(tid, (N_SAMPLE, ""))[0]
                cfg = update_inference_configs(configs, slen); cfg.sample_diffusion.N_sample = n_needed; runner.update_model_configs(cfg); pred = runner.predict(data)
                feat = data["input_feature_dict"]; mask = (feat["centre_atom_mask"]==1) if "centre_atom_mask" in feat else (feat["atom_to_tokatom_idx"]==12); protenix_preds[tid] = pred["coordinate"][:, mask, :].detach().cpu().numpy()
            except: protenix_preds[tid] = None
            finally: gc.collect(); torch.cuda.empty_cache()
    
    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]; combined = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(ptx[j])
        while len(combined) < N_SAMPLE: combined.append(adaptive_rna_constraints(generate_rna_structure(seq, seed=len(combined)), tid, seq, segments_map, confidence=0.2))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE], axis=0)))
    
    sub = pd.DataFrame(all_rows); cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, N_SAMPLE+1) for c in ["x", "y", "z"]]
    sub[cols].to_csv(output_csv, index=False)
    print(f"✓ Saved submission to {output_csv}")

    # Local Validation Print
    print("\nEvaluating LOCAL_VALIDATION_RMSD...")
    v_seg = {r["target_id"]: [(0, len(r["sequence"]))] for _, r in val_seqs.iterrows()}
    v_preds, _ = tbm_phase(val_seqs, train_seqs, process_labels(train_labels), v_seg); v_true = process_labels(val_labels); r_list = []
    for _, r in val_seqs.iterrows():
        tid, seq = r["target_id"], r["sequence"]; p_coords = v_preds.get(tid, [adaptive_rna_constraints(generate_rna_structure(seq), tid, seq, v_seg, confidence=0.2)])[0]; t_coords = v_true.get(tid)
        if t_coords is not None and len(p_coords) == len(t_coords):
            Pc, Qc = p_coords - p_coords.mean(0), t_coords - t_coords.mean(0); H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H); R = Vt.T @ np.diag([1,1,np.linalg.det(Vt.T @ U.T)]) @ U.T
            r_list.append(np.sqrt(np.mean(np.sum((Pc @ R.T - Qc)**2, axis=1))))
    print(f"LOCAL_VALIDATION_RMSD: {np.mean(r_list) if r_list else 0.0:.5f}")

if __name__ == "__main__": main()
