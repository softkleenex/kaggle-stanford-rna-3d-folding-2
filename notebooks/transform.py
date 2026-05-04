import re

with open('notebooks/top1/script.py', 'r') as f:
    code = f.read()

# 1. Update stitch_chunk_coords
stitch_old = """
        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)
"""

stitch_new = """
        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, ramp_len))

        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = 0.5 + 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, ramp_len))
"""
code = code.replace(stitch_old, stitch_new)

# 2. Update adaptive_rna_constraints
const_old = """
def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X
"""

const_new = """
def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=5) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.4 Å (typical C1'-C1' distance)
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.4 - dist) / dist)[:, None] * (0.25 * strength)
            C[:-1] -= adj; C[1:] += adj
            # soft i–i+2  ~9.6 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((9.6 - d2n) / d2n)[:, None] * (0.12 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # Laplacian smoothing
            C[1:-1] += (0.05 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            # self-avoidance (full array, C1'-C1' clash threshold ~3.8 Å)
            if L >= 10:
                P    = C; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
                mask = (sep > 2) & (dm < 3.8)
                if np.any(mask):
                    vec = (diff * ((3.8 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C += (0.02 * strength) * vec
            X[s:e] = C
    return X
"""
code = code.replace(const_old.strip(), const_new.strip())

# 3. Apply constraints to stitched protenix predictions
ptx_old = """
        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                combined.append(ptx[j])  # (seq_len, 3)
"""

ptx_new = """
        # Append Protenix predictions to fill remaining slots
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE:
                    break
                ptx_pred = ptx[j]
                if len(seq) > MAX_SEQ_LEN:
                    # Refine chunk boundaries for long sequences
                    ptx_pred = adaptive_rna_constraints(ptx_pred, tid, segments_map, confidence=0.9, passes=2)
                combined.append(ptx_pred)  # (seq_len, 3)
"""
code = code.replace(ptx_old.strip(), ptx_new.strip())

with open('notebooks/script_v36.py', 'w') as f:
    f.write(code)

print("Done.")
