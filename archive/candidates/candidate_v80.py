import gc
import hashlib
import json
import os
import sys
import time
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

def install_deps():
    deps = [
        "/kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "/kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl"
    ]
    for d in deps:
        if os.path.exists(d):
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", d], check=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# ── Biotite compatibility patch ──────────────────────────────────────────────
try:
    import biotite.structure.io.pdbx as _pdbx_early
    if not hasattr(_pdbx_early, "PDBX_COVALENT_TYPES"):
        _pdbx_early.PDBX_COVALENT_TYPES = list(getattr(
            _pdbx_early, "COVALENT_TYPES",
            ["covale", "disulf", "modres", "saltbr"]
        ))
except Exception:
    pass

# ── Constants & Paths ────────────────────────────────────────────────────────
IS_KAGGLE = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
LOCAL_N_SAMPLES = 2

DATA_BASE      = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV   = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV  = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV    = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS   = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT     = "/kaggle/working/submission.csv"

_RMSA_BASE = "/kaggle/input/datasets/zoushuxian/protenix-rmsa-repo"
_RMSA_CANDIDATES = [f"{_RMSA_BASE}/protenix_kaggle", f"{_RMSA_BASE}"]
RMSA_CODE_DIR = next(
    (p for p in _RMSA_CANDIDATES if os.path.isdir(p) and os.path.isdir(os.path.join(p, "protenix"))),
    _RMSA_CANDIDATES[0]
)
PROTENIX_ASSETS_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
DEFAULT_CODE_DIR = RMSA_CODE_DIR if os.path.isdir(RMSA_CODE_DIR) else PROTENIX_ASSETS_DIR
DEFAULT_ROOT_DIR = PROTENIX_ASSETS_DIR

FINETUNED_CKPT_PATH = "/kaggle/input/datasets/zoushuxian/protenix-finetuned-rna3db-all-1599/1599_ema_0.999.pt"
USE_FINETUNED_CKPT = os.path.exists(FINETUNED_CKPT_PATH)

MODEL_NAME = "protenix_base_20250630_v1.0.0"
N_SAMPLE = 20
SEED = 42
MAX_SEQ_LEN = 512
CHUNK_OVERLAP = 192
DIFFUSION_N_STEP = 200
DIFFUSION_N_CYCLE = 4

MIN_SIMILARITY = 0.0
MIN_PERCENT_IDENTITY_BASE = 50.0
MIN_TEMPLATES_TARGET = 5
THRESHOLD_FLOOR = 30.0

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass

def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2 ** 32)

def resolve_paths():
    return (
        os.environ.get("TEST_CSV", DEFAULT_TEST_CSV),
        os.environ.get("SUBMISSION_CSV", DEFAULT_OUTPUT),
        os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR),
        PROTENIX_ASSETS_DIR,
    )

def ensure_required_files(root_dir: str) -> None:
    root = Path(root_dir)
    for p, name in [
        (root / "checkpoint" / f"{MODEL_NAME}.pt", "checkpoint"),
        (root / "common" / "components.cif", "CCD file"),
        (root / "common" / "components.cif.rdkit_mol.pkl", "CCD cache"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [
        {"name": row["target_id"], "covalent_bonds": [], "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}]}
        for _, row in df.iterrows()
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def build_configs(input_json_path: str, dump_dir: str, model_name: str, n_sample=None, seed=None):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    ns = n_sample if n_sample is not None else N_SAMPLE
    use_seed = seed if seed is not None else SEED
    base = {**configs_base, **{"data": data_configs}, **inference_configs}
    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict): deep_update(t[k], v)
            else: t[k] = v
    deep_update(base, model_configs[model_name])
    arg_str = f"--model_name {model_name} --input_json_path {input_json_path} --dump_dir {dump_dir} --use_msa false --use_template false --use_rna_msa true --sample_diffusion.N_sample {ns} --sample_diffusion.N_step {DIFFUSION_N_STEP} --sample_diffusion.N_cycle {DIFFUSION_N_CYCLE} --seeds {use_seed}"
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

def load_runner_with_checkpoint(configs):
    from runner.inference import InferenceRunner
    runner = InferenceRunner(configs)
    if USE_FINETUNED_CKPT:
        try:
            ckpt = torch.load(FINETUNED_CKPT_PATH, map_location="cpu")
            sd = (ckpt.get("ema_model") or ckpt.get("model") or ckpt.get("state_dict") or ckpt)
            cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            runner.model.load_state_dict(cleaned, strict=False)
        except: pass
    return runner

def kabsch_align(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    if np.linalg.det(Vt.T @ U.T) < 0: Vt[2, :] *= -1
    R = Vt.T @ U.T
    return R, cQ - R @ cP

def tcsr_relax_overlap(coords, s, e, anchor_coords, weight_ramp, passes=20):
    X = coords.copy()
    L = e - s
    if L < 3: return X
    BOND_TARGET, COS_TARGET = 5.95, np.cos(102.0 * np.pi / 180.0)
    X[s:e] = (1.0 - weight_ramp[:, None]) * X[s:e] + weight_ramp[:, None] * anchor_coords
    for _ in range(passes):
        d = X[1:] - X[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
        adj = d * ((BOND_TARGET - dist) / dist)[:, None] * 0.1
        X[:-1] -= adj; X[1:] += adj
        v1, v2 = X[1:-1] - X[:-2], X[2:] - X[1:-1]
        n1, n2 = np.linalg.norm(v1, axis=1)+1e-6, np.linalg.norm(v2, axis=1)+1e-6
        cos_theta = (v1 * v2).sum(axis=1) / (n1 * n2)
        angle_adj = (cos_theta - COS_TARGET)[:, None] * 0.05
        X[1:-1] += angle_adj * (v1/n1[:,None] - v2/n2[:,None])
        X[s:e] += 0.05 * (anchor_coords - X[s:e])
    return X

def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
    if len(chunk_coords_list) == 1:
        c = chunk_coords_list[0]; out = np.zeros((seq_len, 3), dtype=c.dtype); out[:min(len(c), seq_len)] = c[:min(len(c), seq_len)]; return out
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        ps, pe, cs, ce = chunk_ranges[i-1][0], chunk_ranges[i-1][1], chunk_ranges[i][0], chunk_ranges[i][1]
        os_s, os_e = cs, min(pe, ce)
        if os_e - os_s < 3: aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(chunk_coords_list[i][os_s-cs:os_e-cs], aligned[i-1][os_s-ps:os_e-ps])
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full, weights = np.zeros((seq_len, 3)), np.zeros(seq_len)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        cl = len(coords); ae = min(s + cl, seq_len); ul = ae - s; w = np.ones(ul)
        if i > 0: rl = min(chunk_ranges[i-1][1], e) - s; w[:rl] = np.linspace(0, 1, rl)
        full[s:ae] += coords[:ul] * w[:, None]; weights[s:ae] += w
    full[weights > 0] /= weights[weights > 0, None]
    for i in range(1, len(chunk_ranges)):
        s, e = chunk_ranges[i][0], min(chunk_ranges[i-1][1], seq_len)
        if e - s >= 3: full = tcsr_relax_overlap(full, s, e, aligned[i][:e-s], np.linspace(0, 1, e-s))
    return full

def consensus_rerank(preds, seq_len):
    if len(preds) < 3: return preds
    try:
        arr = np.stack([p[:seq_len] if len(p) >= seq_len else np.pad(p, ((0, seq_len-len(p)), (0, 0))) for p in preds])
        centroid = arr.mean(axis=0); rmsds = np.sqrt(np.mean(np.sum((arr - centroid[None])**2, axis=-1), axis=-1))
        return [preds[i] for i in np.argsort(rmsds)]
    except: return preds

_aligner = PairwiseAligner(); _aligner.mode = "global"

def process_labels(df):
    coords = {}
    split = df["ID"].str.rsplit("_", n=1); pfx, pos = split.str[0], pd.to_numeric(split.str[1], errors="coerce").fillna(0).astype(int)
    for prefix, grp in pd.DataFrame({"_pfx": pfx, "_pos": pos, "x": df["x_1"], "y": df["y_1"], "z": df["z_1"]}).groupby("_pfx"):
        arr = grp.sort_values("_pos")[["x", "y", "z"]].values.astype(float)
        for i in range(len(arr)):
            if not np.isnan(arr[i, 0]): continue
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(arr[j, 0])), -1); nv = next((j for j in range(i+1, len(arr)) if not np.isnan(arr[j, 0])), -1)
            if pv >= 0 and nv >= 0: w = (i-pv)/(nv-pv); arr[i] = (1-w)*arr[pv] + w*arr[nv]
            elif pv >= 0: arr[i] = arr[pv] + [3, 0, 0]
            else: arr[i] = [i*3, 0, 0]
        coords[prefix] = np.nan_to_num(arr)
    return coords

def adapt_template_to_query(q_seq, t_seq, t_coords):
    try: aln = next(iter(_aligner.align(q_seq, t_seq)))
    except: return np.zeros((len(q_seq), 3))
    new = np.full((len(q_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        if (qe-qs) == (te-ts): new[qs:qe] = t_coords[ts:te]
    for i in range(len(new)):
        if not np.isnan(new[i, 0]): continue
        pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new[i, 0])), -1); nv = next((j for j in range(i+1, len(new)) if not np.isnan(new[i, 0])), -1)
        if pv >= 0 and nv >= 0: w = (i-pv)/(nv-pv); new[i] = (1-w)*new[pv] + w*new[nv]
        elif pv >= 0: new[i] = new[pv] + [3, 0, 0]
        else: new[i] = [i*3, 0, 0]
    return np.nan_to_num(new)

def tbm_phase(test_df, tr_seqs, tr_coords):
    print("PHASE 1: TBM"); t_preds, p_queue = {}, {}
    for _, row in test_df.iterrows():
        tid, seq, preds = row["target_id"], row["sequence"], []
        for thresh in [50, 45, 40, 35, 30]:
            results = []
            for _, tr in tr_seqs.iterrows():
                if tr["target_id"] not in tr_coords or abs(len(tr["sequence"])-len(seq))/max(len(tr["sequence"]), len(seq))>0.3: continue
                aln = next(iter(_aligner.align(seq, tr["sequence"]))); ident = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if seq[qp] == tr["sequence"][tp])
                if (100*ident/len(seq)) >= thresh: results.append((tr["target_id"], tr["sequence"], aln.score, tr_coords[tr["target_id"]]))
            if len(results) >= 5: break
        results.sort(key=lambda x: x[2], reverse=True); used = set()
        for i, (tmpl_id, tmpl_seq, score, tmpl_coords) in enumerate(results[:N_SAMPLE]):
            if tmpl_id not in used: preds.append(adapt_template_to_query(seq, tmpl_seq, tmpl_coords)); used.add(tmpl_id)
        t_preds[tid] = preds
        if len(preds) < N_SAMPLE: p_queue[tid] = (N_SAMPLE - len(preds), seq)
    return t_preds, p_queue

def calculate_rmsd(P, Q):
    R, t = kabsch_align(P, Q); P_aligned = (P @ R.T) + t
    return np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=-1)))

def run_local_validation(tr_seqs, tr_coords):
    print("Local Validation..."); holdout = tr_seqs.sample(min(15, len(tr_seqs)), random_state=SEED)
    t_rmsd, count = 0, 0
    for _, row in holdout.iterrows():
        tid, seq, true = row["target_id"], row["sequence"], tr_coords.get(row["target_id"])
        if true is None: continue
        results = []
        for _, tr in tr_seqs.iterrows():
            if tr["target_id"] == tid or tr["target_id"] not in tr_coords or abs(len(tr["sequence"])-len(seq))/max(len(tr["sequence"]), len(seq))>0.1: continue
            results.append((tr["sequence"], next(iter(_aligner.align(seq, tr["sequence"]))).score, tr_coords[tr["target_id"]]))
            if len(results) > 50: break
        results.sort(key=lambda x: x[1], reverse=True)
        best = 10.0
        if results:
            for i in range(min(3, len(results))):
                pred = adapt_template_to_query(seq, results[i][0], results[i][2])
                best = min(best, calculate_rmsd(pred, true))
        t_rmsd += best; count += 1
        print(f"  {tid}: RMSD={best:.4f}")
    print(f"LOCAL_VALIDATION_RMSD: {t_rmsd/max(1, count):.5f}")

def main():
    install_deps(); test_csv, out_csv, code_dir, root_dir = resolve_paths()
    sys.path.append(code_dir); ensure_required_files(root_dir); seed_everything(SEED)
    test_df = pd.read_csv(test_csv); test_df = (test_df.head(LOCAL_N_SAMPLES) if not IS_KAGGLE else test_df).reset_index(drop=True)
    tr_seqs = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)], ignore_index=True)
    tr_coords = process_labels(pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)], ignore_index=True))
    t_preds, p_queue = tbm_phase(test_df, tr_seqs, tr_coords); p_preds = {}
    if p_queue:
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        tasks, c_info = [], {}
        for tid, (n_needed, full_seq) in p_queue.items():
            if len(full_seq) <= MAX_SEQ_LEN: tasks.append({"target_id": tid, "sequence": full_seq}); c_info[tid] = [{"name": tid, "range": (0, len(full_seq))}]
            else:
                chunks = []; pos = 0; step = MAX_SEQ_LEN - CHUNK_OVERLAP
                while pos < len(full_seq):
                    end = min(pos + MAX_SEQ_LEN, len(full_seq)); chunks.append((pos, end))
                    if end == len(full_seq): break
                    pos += step
                c_info[tid] = []
                for ci, (cs, ce) in enumerate(chunks):
                    cname = f"{tid}_{ci}"; tasks.append({"target_id": cname, "sequence": full_seq[cs:ce]}); c_info[tid].append({"name": cname, "range": (cs, ce)})
        build_input_json(pd.DataFrame(tasks), "/kaggle/working/in.json")
        configs = update_gpu_compatible_configs(build_configs("/kaggle/working/in.json", "/kaggle/working/out", MODEL_NAME))
        runner = load_runner_with_checkpoint(configs); ds = InferenceDataset(configs); raw = {}
        for i in tqdm(range(len(ds))):
            data, _, err = ds[i]
            if err: continue
            sname = data.get("sample_name", f"s{i}"); tid_b = sname.split("_")[0]
            try:
                runner.update_model_configs(update_inference_configs(configs, data["N_token"].item()))
                pred = runner.predict(data); f = data["input_feature_dict"]; mask = (f["centre_atom_mask"] == 1) if "centre_atom_mask" in f else (f["atom_to_tokatom_idx"] == 12)
                raw[sname] = pred["coordinate"][:, mask, :].detach().cpu().numpy()
            except: pass
            finally: torch.cuda.empty_cache(); gc.collect()
        for tid, (n_needed, seq) in p_queue.items():
            chunks = c_info.get(tid, [])
            if not chunks: continue
            if len(chunks) == 1: p_preds[tid] = raw.get(tid)
            else:
                stitched = []
                for s in range(n_needed):
                    c_list = [raw[c["name"]][min(s, len(raw[c["name"]])-1)] for c in chunks if c["name"] in raw]
                    if len(c_list) == len(chunks): stitched.append(stitch_chunk_coords(c_list, [c["range"] for c in chunks], len(seq)))
                if stitched: p_preds[tid] = np.stack(stitched)
    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]; combined = list(t_preds.get(tid, []))
        if tid in p_preds and p_preds[tid] is not None:
            for j in range(p_preds[tid].shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(p_preds[tid][j])
        while len(combined) < N_SAMPLE:
            ang = np.arange(len(seq))*0.6; combined.append(np.stack([10*np.cos(ang), 10*np.sin(ang), np.arange(len(seq))*2.5], axis=1))
        final = consensus_rerank(combined[:N_SAMPLE], len(seq))
        for i in range(len(seq)):
            r = {"ID": f"{tid}_{i+1}", "resname": seq[i], "resid": i+1}
            for s in range(N_SAMPLE): r[f"x_{s+1}"], r[f"y_{s+1}"], r[f"z_{s+1}"] = final[s][i]
            all_rows.append(r)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False); run_local_validation(tr_seqs, tr_coords)

if __name__ == "__main__": main()
