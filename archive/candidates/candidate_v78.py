import os
import sys
import pandas as pd

# Fix for missing modules in Kaggle
os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

IS_KAGGLE = True
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode")
else:
    print(f"Running in LOCAL mode")

import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

DATA_BASE = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT = "/kaggle/working/submission.csv"

DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME = "protenix_base_20250630_v1.0.0"
N_SAMPLE = 5
SEED = 42
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))

MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

USE_PROTENIX = True
USE_MSA = "false"
USE_TEMPLATE = "false"
USE_RNA_MSA = "true"
MODEL_N_SAMPLE = N_SAMPLE

# ── Helical-Axis-Aligned Seam Interpolation (HAASI) ──────────────────────────

def get_helical_axis(coords: np.ndarray) -> tuple:
    """Calculates the local helical axis and center for a segment."""
    L = len(coords)
    if L < 3: return np.array([0,0,1]), coords.mean(axis=0)
    centered = coords - coords.mean(axis=0)
    # PCA to find the primary axis of the coordinates
    _, _, vh = np.linalg.svd(centered)
    axis = vh[0] # The axis of greatest elongation is usually the helical axis for a chunk
    return axis / (np.linalg.norm(axis) + 1e-12), coords.mean(axis=0)

# ── Steric-Clash-Aware Fragment Re-threading (SCAFR) ─────────────────────────

def apply_scafr(coords: np.ndarray, clash_threshold=3.0, iterations=15) -> np.ndarray:
    X = coords.copy()
    L = len(X)
    if L < 2: return X
    for _ in range(iterations):
        diffs = X[1:] - X[:-1]
        dists = np.linalg.norm(diffs, axis=1) + 1e-12
        clash_mask = dists < clash_threshold
        if not np.any(clash_mask): break
        for i in np.where(clash_mask)[0]:
            vec = X[i+1] - X[i]
            d = np.linalg.norm(vec)
            push_dir = vec / (d + 1e-12)
            noise = np.random.normal(0, 0.05, 3)
            push_dir = (push_dir + noise)
            push_dir /= np.linalg.norm(push_dir)
            X[i+1] = X[i] + push_dir * 5.95
    return X

# ── SO3 Utility (DGI) ─────────────────────────────────────────────────────────

class SO3:
    @staticmethod
    def from_matrix(R):
        tr = np.trace(R)
        cos_theta = np.clip((tr - 1) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-6: return np.zeros(3)
        sin_theta = np.sin(theta)
        omega_hat = (R - R.T) / (2 * sin_theta)
        return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]]) * theta

    @staticmethod
    def exp(omega):
        theta = np.linalg.norm(omega)
        if theta < 1e-6: return np.eye(3)
        u = omega / theta
        ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        return np.eye(3) + np.sin(theta) * ux + (1 - np.cos(theta)) * (ux @ ux)

# ── Pipeline Functions ──────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def resolve_paths():
    test_csv = os.environ.get("TEST_CSV", DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV", DEFAULT_OUTPUT)
    code_dir = os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR)
    root_dir = os.environ.get("PROTENIX_ROOT_DIR", DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir

def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [
        {"name": row["target_id"], "covalent_bonds": [], "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}]}
        for _, row in df.iterrows()
    ]
    with open(json_path, "w", encoding="utf-8") as f: json.dump(data, f)

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
    chunks, pos, step = [], 0, max_len - overlap
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P: np.ndarray, Q: np.ndarray):
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return R, cQ - R @ cP

def enforce_spring_geometry(coords: np.ndarray, iterations=60, alpha_bond=0.4, alpha_angle=0.2, tether_strength=0.03) -> np.ndarray:
    X, X_orig, L = coords.copy(), coords.copy(), len(coords)
    if L < 3: return X
    for _ in range(iterations):
        d1 = X[1:] - X[:-1]; dist1 = np.linalg.norm(d1, axis=1) + 1e-12
        err1 = (dist1 - 5.95) / dist1
        s1 = d1 * err1[:, None] * alpha_bond
        X[:-1] += s1 * 0.5; X[1:] -= s1 * 0.5
        d2 = X[2:] - X[:-2]; dist2 = np.linalg.norm(d2, axis=1) + 1e-12
        err2 = (dist2 - 10.2) / dist2
        s2 = d2 * err2[:, None] * alpha_angle
        X[:-2] += s2 * 0.5; X[2:] -= s2 * 0.5
        X += (X_orig - X) * tether_strength
    return X

def stitch_chunk_coords(chunk_coords_list: list, chunk_ranges: list, seq_len: int) -> np.ndarray:
    if len(chunk_coords_list) == 1:
        out = np.zeros((seq_len, 3), dtype=np.float64)
        out[:min(len(chunk_coords_list[0]), seq_len)] = chunk_coords_list[0][:min(len(chunk_coords_list[0]), seq_len)]
        return enforce_spring_geometry(out)

    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        pS, pE = chunk_ranges[i-1]; cS, cE = chunk_ranges[i]
        oS, oE = cS, min(pE, cE)
        if (oE - oS) < 3: aligned.append(chunk_coords_list[i].copy()); continue
        pO = aligned[i-1][oS-pS:oE-pS]; cO = chunk_coords_list[i][oS-cS:oE-cS]
        R, t = kabsch_align(cO, pO)
        aligned.append((chunk_coords_list[i] @ R.T) + t)

    full, weights = np.zeros((seq_len, 3), dtype=np.float64), np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        aL = min(len(coords), seq_len - s)
        view, w = coords[:aL], np.ones(aL, dtype=np.float64)
        if i > 0:
            overlap = chunk_ranges[i-1][1] - s
            if overlap > 0: w[:overlap] = np.linspace(0, 1, overlap)
        if i < len(chunk_ranges) - 1:
            overlap_next = e - chunk_ranges[i+1][0]
            if overlap_next > 0:
                s_next = chunk_ranges[i+1][0] - s
                w[s_next:] = np.minimum(w[s_next:], np.linspace(1, 0, overlap_next))
        full[s:s+aL] += view * w[:, None]
        weights[s:s+aL] += w

    mask = weights > 1e-12
    full[mask] /= weights[mask, None]

    # HAASI Correction: Enforce helical axis continuity across seams
    for i in range(1, len(chunk_ranges)):
        s_ov, e_ov = chunk_ranges[i][0], min(chunk_ranges[i-1][1], chunk_ranges[i][1])
        if (e_ov - s_ov) >= 6:
            ov_slice = full[s_ov:e_ov]
            axis, center = get_helical_axis(ov_slice)
            projs = (ov_slice - center) @ axis
            # Enforce 3.0A helical rise (A-form RNA)
            t_projs = projs[0] + np.arange(len(projs)) * 3.0
            full[s_ov:e_ov] += (0.35 * (t_projs - projs))[:, None] * axis # Gradual helical pitch alignment

    return enforce_spring_geometry(apply_scafr(full))

_aligner = PairwiseAligner()
_aligner.mode, _aligner.match_score, _aligner.mismatch_score = "global", 2, -1.5
_aligner.open_gap_score, _aligner.extend_gap_score = -8, -0.4

def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or str(stoich).strip() == "": return []
    return [(p.split(":")[0].strip(), int(p.split(":")[1])) for p in str(stoich).split(";")]

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
        cd, order = parse_fasta(all_sq), parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = cd.get(ch)
            if not base: return [(0, len(seq))]
            for _ in range(cnt): segs.append((pos, pos + len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except: return [(0, len(seq))]

def build_segments_map(df: pd.DataFrame) -> tuple:
    sM, tM = {}, {}
    for _, r in df.iterrows():
        tid = r["target_id"]
        sM[tid] = get_chain_segments(r)
        tM[tid] = str(r.get("stoichiometry", "")) if not pd.isna(r.get("stoichiometry", "")) else ""
    return sM, tM

def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for p, g in labels_df.groupby(prefixes):
        coords[p] = g.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict or abs(len(tseq)-len(query_seq))/max(len(tseq),1) > 0.3: continue
        try:
            aln = next(iter(_aligner.align(query_seq, tseq)))
            norm_s = aln.score / (2 * min(len(query_seq), len(tseq)))
            identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
            if (100 * identical / len(query_seq)) >= MIN_PERCENT_IDENTITY:
                results.append((tid, tseq, norm_s, train_coords_dict[tid], 100*identical/len(query_seq)))
        except: continue
    return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    try:
        aln = next(iter(_aligner.align(query_seq, template_seq)))
        nc = np.full((len(query_seq), 3), np.nan)
        for (qs, qe), (ts, te) in zip(*aln.aligned):
            if (te-ts) == (qe-qs): nc[qs:qe] = template_coords[ts:te]
        for i in range(len(nc)):
            if np.isnan(nc[i, 0]):
                pv = next((j for j in range(i-1,-1,-1) if not np.isnan(nc[j,0])), -1)
                nv = next((j for j in range(i+1,len(nc)) if not np.isnan(nc[j,0])), -1)
                if pv >= 0 and nv >= 0: nc[i] = (1-(i-pv)/(nv-pv))*nc[pv] + ((i-pv)/(nv-pv))*nc[nv]
                elif pv >= 0: nc[i] = nc[pv] + [3,0,0]
                elif nv >= 0: nc[i] = nc[nv] + [3,0,0]
                else: nc[i] = [i*3,0,0]
        return np.nan_to_num(nc)
    except: return generate_rna_structure(query_seq)

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
    X, segments = coords.copy(), segments_map.get(target_id, [(0, len(coords))])
    strength = max(0.8 * (1.0 - min(confidence, 0.98)), 0.05)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = len(C)
            if L < 3: continue
            d1 = C[1:]-C[:-1]; dist1 = np.linalg.norm(d1, axis=1)+1e-12
            C[:-1] -= d1*((5.95-dist1)/dist1)[:,None]*(0.1*strength)
            C[1:] += d1*((5.95-dist1)/dist1)[:,None]*(0.1*strength)
            X[s:e] = C
    return X

def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return enforce_spring_geometry(coords, iterations=50)

def run_pipeline(test_df, train_seqs, train_coords, seg_map, code_dir, root_dir):
    template_preds, protenix_queue = {}, {}
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        similar = find_similar_sequences_detailed(seq, train_seqs, train_coords)
        preds = []
        for t_id, t_seq, sim, t_coords, p_id in similar:
            if len(preds) >= N_SAMPLE: break
            preds.append(adaptive_rna_constraints(adapt_template_to_query(seq, t_seq, t_coords), tid, seg_map, sim))
        template_preds[tid] = preds
        if len(preds) < N_SAMPLE: protenix_queue[tid] = (N_SAMPLE - len(preds), seq)

    protenix_preds = {}
    if protenix_queue and USE_PROTENIX:
        if code_dir not in sys.path: sys.path.append(code_dir)
        try:
            from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
            from protenix.data.inference.infer_dataloader import InferenceDataset
            from configs.configs_base import configs as configs_base
            from configs.configs_data import data_configs
            from configs.configs_inference import inference_configs
            from configs.configs_model_type import model_configs
            from protenix.config.config import parse_configs

            def build_configs(input_json_path, dump_dir, model_name):
                base = {**configs_base, "data": data_configs, **inference_configs}
                def du(t, p):
                    for k, v in p.items():
                        if isinstance(v, dict) and k in t: du(t[k], v)
                        else: t[k] = v
                du(base, model_configs[model_name])
                args = f"--model_name {model_name} --input_json_path {input_json_path} --dump_dir {dump_dir} --use_msa {USE_MSA} --use_template {USE_TEMPLATE} --use_rna_msa {USE_RNA_MSA} --sample_diffusion.N_sample {MODEL_N_SAMPLE} --seeds {SEED}"
                return parse_configs(configs=base, arg_str=args, fill_required_with_null=True)

            tasks, chunk_info = [], {}
            for tid, (n_need, full_seq) in protenix_queue.items():
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

            ijp = "/kaggle/working/protenix_queue_input.json"
            build_input_json(pd.DataFrame(tasks), ijp)
            cfg = build_configs(ijp, "/kaggle/working/outputs", MODEL_NAME)
            cfg = update_gpu_compatible_configs(cfg)
            runner, dataset = InferenceRunner(cfg), InferenceDataset(cfg)

            raw_preds = {}
            for i in range(len(dataset)):
                data, atom_arr, err = dataset[i]
                if err: continue
                s_name = data.get("sample_name", f"sample_{i}")
                t_id = s_name.split("_chunk")[0]
                n_need = protenix_queue.get(t_id, (N_SAMPLE, ""))[0]
                try:
                    nc = update_inference_configs(cfg, data["N_token"].item())
                    nc.sample_diffusion.N_sample = n_need
                    runner.update_model_configs(nc)
                    pred = runner.predict(data)
                    mask = (data["input_feature_dict"]["centre_atom_mask"] == 1).to(pred["coordinate"].device)
                    raw_preds[s_name] = pred["coordinate"][:, mask, :].detach().cpu().numpy()
                except: continue

            for tid, (n_need, full_seq) in protenix_queue.items():
                chunks = chunk_info.get(tid, [])
                if not chunks: continue
                if len(chunks) == 1: protenix_preds[tid] = raw_preds.get(tid)
                else:
                    stitched_s = []
                    for s_idx in range(n_need):
                        c_list, r_list = [], []
                        for ci_ in chunks:
                            c_out = raw_preds.get(ci_["name"])
                            if c_out is not None:
                                c_list.append(c_out[s_idx if s_idx < c_out.shape[0] else -1])
                                r_list.append(ci_["range"])
                        if len(c_list) == len(chunks): stitched_s.append(stitch_chunk_coords(c_list, r_list, len(full_seq)))
                    if stitched_s: protenix_preds[tid] = np.stack(stitched_s, 0)
        except Exception as e:
            print(f"Protenix failed: {e}")
            for tid, (n_need, full_seq) in protenix_queue.items():
                protenix_preds[tid] = np.stack([generate_rna_structure(full_seq, seed=i) for i in range(n_need)])

    all_rows = []
    for _, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        combined = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(ptx[j])
        while len(combined) < N_SAMPLE: combined.append(generate_rna_structure(seq, seed=len(combined)))
        all_rows.extend(coords_to_rows(tid, seq, np.stack(combined[:N_SAMPLE], 0)))
    return all_rows

def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    seed_everything(SEED)
    train_seqs = pd.read_csv(DEFAULT_TRAIN_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS)
    train_coords = process_labels(train_labels)
    test_df = pd.read_csv(test_csv).head(LOCAL_N_SAMPLES) if not IS_KAGGLE else pd.read_csv(test_csv)
    seg_map, _ = build_segments_map(test_df)
    all_rows = run_pipeline(test_df, train_seqs, train_coords, seg_map, code_dir, root_dir)
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, N_SAMPLE+1) for c in ["x","y","z"]]
    sub[cols].to_csv(output_csv, index=False)
    print(f"✓ Saved to {output_csv}")

if __name__ == "__main__": main()
