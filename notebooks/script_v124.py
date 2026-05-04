import os
os.system("pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
os.system("pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl")

import os
import sys
import pandas as pd

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
IS_KAGGLE = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
LOCAL_N_SAMPLES = 5

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets will be processed.")

import gc
import json
import time
import numpy as np
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm
from pathlib import Path

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

def compute_rmsd(pred_coords, true_coords):
    if len(pred_coords) != len(true_coords): return np.nan
    centroid_P, centroid_Q = pred_coords.mean(axis=0), true_coords.mean(axis=0)
    Pc, Qc = pred_coords - centroid_P, true_coords - centroid_Q
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    if np.linalg.det(Vt.T @ U.T) < 0: Vt[2, :] *= -1
    R = Vt.T @ U.T
    return float(np.sqrt(np.mean(np.sum((Pc @ R.T - Qc)**2, axis=1))))

def get_c1_mask(data, atom_array):
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"): m &= atom_array.is_rna
                return torch.from_numpy(m).bool()
        except: pass
    f = data["input_feature_dict"]
    if "centre_atom_mask" in f: return (f["centre_atom_mask"] == 1).bool()
    mask11, mask12 = (f["atom_to_tokatom_idx"] == 11).bool(), (f["atom_to_tokatom_idx"] == 12).bool()
    return mask11 if abs(mask11.sum().item() - data.get("N_token", torch.tensor(0)).item()) < abs(mask12.sum().item() - data.get("N_token", torch.tensor(0)).item()) else mask12

DATA_BASE = "/kaggle/input/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV = f"{DATA_BASE}/test_sequences.csv", f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV, DEFAULT_VAL_LBLS = f"{DATA_BASE}/validation_sequences.csv", f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT = "/kaggle/working/submission.csv"
DEFAULT_CODE_DIR = "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
MODEL_NAME, N_SAMPLE, SEED = "protenix_base_20250630_v1.0.0", 5, 42
MAX_SEQ_LEN, CHUNK_OVERLAP = 512, 128
MIN_SIMILARITY, MIN_PERCENT_IDENTITY = 0.0, 50.0
USE_PROTENIX = True

def parse_bool(v, d=False):
    v = str(v).lower()
    return "true" if v in {"1", "true", "t", "y", "on"} else ("false" if v in {"0", "false", "f", "n", "off"} else ("true" if d else "false"))

USE_MSA, USE_TEMPLATE, USE_RNA_MSA = parse_bool(os.environ.get("USE_MSA", "false")), parse_bool(os.environ.get("USE_TEMPLATE", "false")), parse_bool(os.environ.get("USE_RNA_MSA", "true"))
MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

def seed_everything(s):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)

def resolve_paths(): return os.environ.get("TEST_CSV", DEFAULT_TEST_CSV), os.environ.get("SUBMISSION_CSV", DEFAULT_OUTPUT), os.environ.get("PROTENIX_CODE_DIR", DEFAULT_CODE_DIR), os.environ.get("PROTENIX_ROOT_DIR", DEFAULT_CODE_DIR)

def build_input_json(df, p):
    with open(p, "w") as f: json.dump([{"name": r["target_id"], "covalent_bonds": [], "sequences": [{"rnaSequence": {"sequence": r["sequence"], "count": 1}}]} for _, r in df.iterrows()], f)

def build_configs(p, d, m):
    from configs.configs_base import configs as cb; from configs.configs_data import data_configs as cd; from configs.configs_inference import inference_configs as ci; from configs.configs_model_type import model_configs as cm; from protenix.config.config import parse_configs
    base = {**cb, "data": cd, **ci}; base.update(cm[m])
    return parse_configs(configs=base, arg_str=f"--model_name {m} --input_json_path {p} --dump_dir {d} --use_msa {USE_MSA} --use_template {USE_TEMPLATE} --use_rna_msa {USE_RNA_MSA} --sample_diffusion.N_sample {MODEL_N_SAMPLE} --seeds {SEED}", fill_required_with_null=True)

def coords_to_rows(tid, seq, coords):
    rows = []
    for i in range(len(seq)):
        r = {"ID": f"{tid}_{i+1}", "resname": seq[i], "resid": i+1}
        for s in range(N_SAMPLE):
            x, y, z = coords[s, i] if s < coords.shape[0] and i < coords.shape[1] else (0.,0.,0.)
            r[f"x_{s+1}"], r[f"y_{s+1}"], r[f"z_{s+1}"] = float(x), float(y), float(z)
        rows.append(r)
    return rows

def split_into_chunks(l, m, o):
    if l <= m: return [(0, l)]
    chunks, pos, step = [], 0, m - o
    while pos < l:
        end = min(pos + m, l); chunks.append((pos, end))
        if end == l: break
        pos += step
    return chunks

def kabsch_align(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    if np.linalg.det(Vt.T @ U.T) < 0: Vt[2, :] *= -1
    R = Vt.T @ U.T
    return R, cQ - R @ cP

def stitch_chunk_coords(cl, rl, l):
    if len(cl) == 1:
        o = np.zeros((l, 3)); ml = min(cl[0].shape[0], l); o[:ml] = cl[0][:ml]
        return o
    aligned = [cl[0].copy()]
    for i in range(1, len(cl)):
        ps, pe = rl[i-1]; cs, ce = rl[i]; os_, oe = cs, min(pe, ce)
        if oe - os_ < 3: aligned.append(cl[i].copy()); continue
        pO, cO = aligned[i-1][os_-ps:oe-ps], cl[i][os_-cs:oe-cs]
        v = ~(np.isnan(pO).any(1) | np.isnan(cO).any(1))
        if v.sum() < 3: aligned.append(cl[i].copy()); continue
        R, t = kabsch_align(cO[v], pO[v])
        aligned.append((cl[i] @ R.T) + t)
    f, w_sum = np.zeros((l, 3)), np.zeros(l)
    for i, ((s, e), c) in enumerate(zip(rl, aligned)):
        cln = c.shape[0]; ae = min(s + cln, l); ul = ae - s; w = np.ones(ul)
        if i > 0:
            rs, re = s, min(rl[i-1][1], e); rln = re - rs
            if rln > 0: w[:rln] = np.linspace(0, 1, rln)
        if i < len(rl) - 1:
            ns = rl[i+1][0]; rst, rln = ns - s, ae - ns
            if rln > 0 and rst < ul: w[rst:ul] = np.linspace(1, 0, rln)
        f[s:ae] += c[:ul] * w[:, None]; w_sum[s:ae] += w
    m = w_sum > 0; f[m] /= w_sum[m, None]
    return f

def _make_aligner():
    a = PairwiseAligner(); a.mode = "global"; a.match_score, a.mismatch_score = 2, -1.5
    a.open_gap_score, a.extend_gap_score = -8, -0.4
    a.query_left_open_gap_score = a.query_right_open_gap_score = a.target_left_open_gap_score = a.target_right_open_gap_score = -8
    a.query_left_extend_gap_score = a.query_right_extend_gap_score = a.target_left_extend_gap_score = a.target_right_extend_gap_score = -0.4
    return a
_aligner = _make_aligner()

def get_chain_segments(row):
    s, st, sq = row["sequence"], row.get("stoichiometry", ""), row.get("all_sequences", "")
    if pd.isna(st) or pd.isna(sq) or not st.strip(): return [(0, len(s))]
    try:
        cdict, order, segs, pos = parse_fasta(sq), [p.split(":") for p in st.split(";")], [], 0
        for ch, cnt in order:
            b = cdict.get(ch.strip())
            for _ in range(int(cnt)): segs.append((pos, pos + len(b))); pos += len(b)
        return segs if pos == len(s) else [(0, len(s))]
    except: return [(0, len(s))]

def parse_fasta(f):
    o, c, p = {}, None, []
    for l in str(f).splitlines():
        l = l.strip()
        if not l: continue
        if l.startswith(">"):
            if c: o[c] = "".join(p)
            c, p = l[1:].split()[0], []
        else: p.append(l.replace(" ", ""))
    if c: o[c] = "".join(p)
    return o

def build_segments_map(df): return {r["target_id"]: get_chain_segments(r) for _, r in df.iterrows()}

def process_labels(df):
    c = {}; px = df["ID"].str.rsplit("_", n=1).str[0]
    for p, g in df.groupby(px): c[p] = g.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return c

def find_similar_sequences_detailed(q, df, cdict, top=30):
    res = []
    for _, r in df.iterrows():
        tid, tseq = r["target_id"], r["sequence"]
        if tid not in cdict or abs(len(tseq)-len(q))/max(len(tseq),len(q)) > 0.3: continue
        aln = next(iter(_aligner.align(q, tseq)))
        norm = aln.score / (2 * min(len(q), len(tseq)))
        ident = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned) for qp, tp in zip(range(qs, qe), range(ts, te)) if q[qp] == tseq[tp])
        pct = 100 * ident / len(q)
        res.append((tid, tseq, norm, cdict[tid], pct))
    res.sort(key=lambda x: x[2], reverse=True)
    return res[:top]

def adapt_template_to_query(q, t, tc):
    aln = next(iter(_aligner.align(q, t))); nc = np.full((len(q), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        if te-ts == qe-qs: nc[qs:qe] = tc[ts:te]
    for i in range(len(nc)):
        if np.isnan(nc[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(nc[j,0])), -1)
            nv = next((j for j in range(i+1, len(nc)) if not np.isnan(nc[j,0])), -1)
            nc[i] = (1-(i-pv)/(nv-pv))*nc[pv]+((i-pv)/(nv-pv))*nc[nv] if pv>=0 and nv>=0 else (nc[pv]+[3,0,0] if pv>=0 else (nc[nv]+[3,0,0] if nv>=0 else [i*3,0,0]))
    return np.nan_to_num(nc)

def adaptive_rna_constraints(X, tid, smap, confidence=1.0, passes=2):
    X = X.copy(); segs = smap.get(tid, [(0, len(X))]); str_ = max(0.75*(1-min(confidence,0.97)), 0.02)
    for _ in range(passes):
        for s, e in segs:
            C = X[s:e]; L = e - s
            if L < 3: continue
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * str_); C[:-1] -= adj; C[1:] += adj
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.1 * str_); C[:-2] -= adj2; C[2:] += adj2
            if L >= 5:
                d4 = C[4:] - C[:-4]; d4n = np.linalg.norm(d4, axis=1) + 1e-6
                adj4 = d4 * ((16.5 - d4n) / d4n)[:, None] * (0.05 * str_); C[:-4] -= adj4; C[4:] += adj4
            C[1:-1] += (0.06 * str_) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            if L >= 3: C[1:-1] -= (0.05 * str_) * (C[2:] - 2*C[1:-1] + C[:-2])
            if L >= 25:
                idx = np.linspace(0, L-1, min(L, 160)).astype(int); P = C[idx]; diff = P[:, None] - P[None, :]
                dm = np.linalg.norm(diff, 2) + 1e-6; sep = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask): C[idx] += (0.015 * str_) * (diff * ((3.2-dm)/dm)[:,:,None] * mask[:,:,None]).sum(1)
            X[s:e] = C
    return X

def _rotmat(a, ang):
    a = np.asarray(a, float); a /= np.linalg.norm(a) + 1e-12
    x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1 - c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s], [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s], [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

def apply_hinge(c, seg, rng, deg=22):
    s, e = seg; L = e - s
    if L < 30: return c
    pvt = s + int(rng.integers(10, L-10)); R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = c.copy(); p0 = X[pvt].copy(); X[pvt+1:e] = (X[pvt+1:e] - p0) @ R.T + p0
    return X

def jitter_chains(c, segs, rng, deg=12, trans=1.5):
    X = c.copy(); gc_ = X.mean(0, keepdims=True)
    for s, e in segs:
        R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
        sh = rng.normal(size=3); sh = sh / (np.linalg.norm(sh)+1e-12) * float(rng.uniform(0, trans))
        cm = X[s:e].mean(0, keepdims=True); X[s:e] = (X[s:e] - cm) @ R.T + cm + sh
    X -= X.mean(0, keepdims=True) - gc_
    return X

def smooth_wiggle(c, segs, rng, amp=0.8):
    X = c.copy()
    for s, e in segs:
        L = e - s
        if L < 20: continue
        ctrl, disp, t = np.linspace(0, L-1, 6), rng.normal(0, amp, (6, 3)), np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X

def generate_rna_structure(seq, seed=None):
    if seed: np.random.seed(seed)
    n = len(seq); c = np.zeros((n, 3))
    for i in range(n): ang = i*0.6; c[i] = [10*np.cos(ang), 10*np.sin(ang), i*2.5]
    return c

def tbm_phase(test_df, train_df, cdict, smap):
    tp, pq = {}, {}
    for _, r in test_df.iterrows():
        tid, seq, segs = r["target_id"], r["sequence"], smap.get(r["target_id"], [(0, len(r["sequence"]))])
        sim = find_similar_sequences_detailed(seq, train_df, cdict); ps, used = [], set()
        for i, (tmpl_id, tmpl_seq, s, tc, pct) in enumerate(sim):
            if len(ps) >= N_SAMPLE or s < MIN_SIMILARITY or pct < MIN_PERCENT_IDENTITY: break
            if tmpl_id in used: continue
            rng = np.random.default_rng((r.name * 10**10 + i * 10007) % (2**32))
            X = adapt_template_to_query(seq, tmpl_seq, tc); slot = len(ps)
            if slot == 1: X += rng.normal(0, max(0.01, (0.4-s)*0.06), X.shape)
            elif slot == 2: X = apply_hinge(X, max(segs, key=lambda se: se[1]-se[0]), rng)
            elif slot == 3: X = jitter_chains(X, segs, rng)
            elif slot > 3: X = smooth_wiggle(X, segs, rng)
            ps.append(adaptive_rna_constraints(X, tid, smap, confidence=s)); used.add(tmpl_id)
        tp[tid] = ps; n_n = N_SAMPLE - len(ps)
        if n_n > 0: pq[tid] = (n_n, seq)
    return tp, pq

def main():
    test_p, out_p, c_dir, r_dir = resolve_paths()
    sys.path.append(c_dir); seed_everything(SEED)
    if not IS_KAGGLE:
        t_full = pd.read_csv(DEFAULT_TRAIN_CSV); l_full = pd.read_csv(DEFAULT_TRAIN_LBLS)
        test_df = t_full.tail(LOCAL_N_SAMPLES).reset_index(drop=True)
        val_gt = process_labels(l_full[l_full["ID"].str.split("_").str[0].isin(test_df["target_id"])])
        train_df, train_l = t_full[~t_full["target_id"].isin(test_df["target_id"])], l_full[~l_full["ID"].str.split("_").str[0].isin(test_df["target_id"])]
        c_seqs, c_lbls = pd.concat([train_df, pd.read_csv(DEFAULT_VAL_CSV)]), pd.concat([train_l, pd.read_csv(DEFAULT_VAL_LBLS)])
    else:
        test_df = pd.read_csv(test_p); val_gt = None
        c_seqs, c_lbls = pd.concat([pd.read_csv(DEFAULT_TRAIN_CSV), pd.read_csv(DEFAULT_VAL_CSV)]), pd.concat([pd.read_csv(DEFAULT_TRAIN_LBLS), pd.read_csv(DEFAULT_VAL_LBLS)])
    cdict, smap = process_labels(c_lbls), build_segments_map(test_df)
    tp, pq = tbm_phase(test_df, c_seqs, cdict, smap); pp = {}
    if pq and USE_PROTENIX:
        tasks, work = [], Path("/kaggle/working")
        for tid, (n, s) in pq.items():
            if len(s) <= MAX_SEQ_LEN: tasks.append({"target_id": tid, "sequence": s})
            else:
                for ci, (cs, ce) in enumerate(split_into_chunks(len(s), MAX_SEQ_LEN, CHUNK_OVERLAP)): tasks.append({"target_id": f"{tid}_chunk{ci}", "sequence": s[cs:ce]})
        ijp = str(work / "protenix_input.json"); build_input_json(pd.DataFrame(tasks), ijp)
        try:
            from protenix.data.inference.infer_dataloader import InferenceDataset
            from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
            cfg = build_configs(ijp, str(work / "outputs"), MODEL_NAME); runner = InferenceRunner(update_gpu_compatible_configs(cfg)); ds = InferenceDataset(cfg); raw = {}
            for i in tqdm(range(len(ds))):
                d, aa, e = ds[i]; sn = d["sample_name"]; tid = sn.split("_chunk")[0] if "_chunk" in sn else sn
                n_n = pq.get(tid, (N_SAMPLE, ""))[0]; sl = d["N_token"].item()
                try:
                    nc = update_inference_configs(cfg, sl); nc.sample_diffusion.N_sample = n_n; runner.update_model_configs(nc)
                    p = runner.predict(d); rc = p["coordinate"]
                    f = d["input_feature_dict"]
                    if "centre_atom_mask" in f: m = (f["centre_atom_mask"] == 1).to(rc.device)
                    else: m11, m12 = (f["atom_to_tokatom_idx"] == 11).to(rc.device), (f["atom_to_tokatom_idx"] == 12).to(rc.device); m = m11 if abs(m11.sum()-sl) < abs(m12.sum()-sl) else m12
                    c = rc[:, m, :].detach().cpu().numpy()
                    raw[sn] = c if not (c.shape[1]>1 and np.all(np.linalg.norm(c[0,1:]-c[0,:-1], axis=-1)<1e-4)) else None
                except: raw[sn] = None
                finally: gc.collect(); torch.cuda.empty_cache()
            for tid, (n, s) in pq.items():
                if len(s) <= MAX_SEQ_LEN: pp[tid] = raw.get(tid)
                else:
                    cs = [raw.get(f"{tid}_chunk{ci}") for ci in range(len(split_into_chunks(len(s), MAX_SEQ_LEN, CHUNK_OVERLAP)))]
                    if all(c is not None for c in cs):
                        pp[tid] = np.stack([stitch_chunk_coords([c[si] for c in cs], split_into_chunks(len(s), MAX_SEQ_LEN, CHUNK_OVERLAP), len(s)) for si in range(n)], 0)
        except: pass
    all_r = []
    for _, r in test_df.iterrows():
        tid, seq = r["target_id"], r["sequence"]; comb = list(tp.get(tid, [])); ptx = pp.get(tid)
        if ptx is not None:
            for j in range(ptx.shape[0]):
                if len(comb) >= N_SAMPLE: break
                comb.append(ptx[j])
        while len(comb) < N_SAMPLE: comb.append(adaptive_rna_constraints(generate_rna_structure(seq, seed=r.name*10**6+len(comb)*1000), tid, smap, 0.2))
        all_r.extend(coords_to_rows(tid, seq, np.stack(comb[:N_SAMPLE], 0)))
    sub = pd.DataFrame(all_r); cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, N_SAMPLE+1) for c in ["x","y","z"]]
    sub[[c for c in cols if c.startswith(("x_","y_","z_"))]] = sub[[c for c in cols if c.startswith(("x_","y_","z_"))]].clip(-999.999, 9999.999)
    sub[cols].to_csv(out_p, index=False)
    if val_gt:
        rms = []
        for tid in test_df["target_id"]:
            tr = val_gt.get(tid); pr = sub[sub["ID"].str.startswith(tid+"_")].sort_values("resid")[["x_1","y_1","z_1"]].values
            if tr is not None and len(pr) == len(tr): r = compute_rmsd(pr, tr); rms.append(r) if not np.isnan(r) else None
        if rms: print(f"LOCAL_VALIDATION_RMSD: {np.mean(rms):.5f}")

if __name__ == "__main__": main()
