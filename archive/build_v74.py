import re

with open('notebooks/script_v73.py', 'r') as f:
    code = f.read()

# Remove the pbd_relaxation function
code = re.sub(
    r"def pbd_relaxation\(.*?return C\n",
    "",
    code,
    flags=re.DOTALL
)

# We need to replace stitch_chunk_coords with the new version
# I'll just find the start of stitch_chunk_coords and the end of it
start_idx = code.find("def stitch_chunk_coords")
end_idx = code.find("# ─────────────── TBM Core Functions ──────────────────────────────────────────", start_idx)

new_stitch_code = """
def distance_matrix_relaxation(coords_1: np.ndarray, coords_2: np.ndarray, 
                               blended_coords: np.ndarray, 
                               ramp_weights: np.ndarray, 
                               num_iters: int = 150) -> np.ndarray:
    \"\"\"
    Distance Matrix Blending (DMB) with PyTorch MDS gradient relaxation.
    Optimizes the interpolated coordinates to match the linearly blended 
    distance matrices of the two overlapping chunks, eliminating shrinkage.
    \"\"\"
    L = len(blended_coords)
    if L < 3: return blended_coords
    
    # 1. Compute pairwise distance matrices for the overlap
    def calc_D(c):
        return np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    
    D1 = calc_D(coords_1)
    D2 = calc_D(coords_2)
    
    # 2. Blend distance matrices based on the ramp weights
    # We use the average of the weights for the pairwise interaction
    W = 0.5 * (ramp_weights[:, None] + ramp_weights[None, :])
    D_target = W * D1 + (1.0 - W) * D2
    
    # 3. PyTorch MDS optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(blended_coords, dtype=torch.float32, device=device, requires_grad=True)
    D_t = torch.tensor(D_target, dtype=torch.float32, device=device)
    
    # Pin the boundary regions to ensure seamless attachment
    pin_mask = torch.zeros(L, dtype=torch.bool, device=device)
    pin_mask[:3] = True
    pin_mask[-3:] = True
    X_orig = torch.tensor(blended_coords, dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam([X], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        D_pred = torch.norm(diff, dim=2) + 1e-6
        
        # Loss 1: Match target distance matrix
        loss_D = torch.nn.functional.mse_loss(D_pred, D_t)
        
        # Loss 2: Pin boundaries
        loss_pin = torch.nn.functional.mse_loss(X[pin_mask], X_orig[pin_mask])
        
        # Loss 3: Prior on bond length (5.95) and angle distance (10.2)
        bond_lens = torch.norm(X[1:] - X[:-1], dim=1)
        loss_bond = torch.nn.functional.mse_loss(bond_lens, torch.full_like(bond_lens, 5.95))
        
        if L > 2:
            angle_dists = torch.norm(X[2:] - X[:-2], dim=1)
            loss_angle = torch.nn.functional.mse_loss(angle_dists, torch.full_like(angle_dists, 10.2))
        else:
            loss_angle = torch.tensor(0.0, device=device)
        
        loss = loss_D + 10.0 * loss_pin + 2.0 * loss_bond + 1.0 * loss_angle
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return X.detach().cpu().numpy()

def stitch_chunk_coords(chunk_coords_list: list,
                        chunk_ranges: list,
                        seq_len: int) -> np.ndarray:
    \"\"\"
    Merge overlapping chunk coordinates into a full sequence geometry.
    Applies Kabsch alignment on overlapping residues, and uses Distance
    Matrix Blending (DMB) with PyTorch MDS relaxation to strictly preserve
    physical volume and bond geometries without shrinkage.
    \"\"\"
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len:
            return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:coords.shape[0]] = coords
        return out

    # Start with the first chunk aligned to itself (identity)
    aligned = [chunk_coords_list[0].copy()]

    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]

        ov_start = cur_start
        ov_end = min(prev_end, cur_end)
        ov_len = ov_end - ov_start

        if ov_len < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue

        prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
        cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]

        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue

        # Align current chunk to previous chunk using only the overlap region
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        transformed = (chunk_coords_list[i] @ R.T) + t
        aligned.append(transformed)

    # Blend them together
    full = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len, dtype=np.float64)
    
    # Store overlap records to apply DMB-MDS after linear blending
    overlaps = []

    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        chunk_len = coords.shape[0]
        actual_end = min(s + chunk_len, seq_len)
        used_len = actual_end - s

        w = np.ones(used_len, dtype=np.float64)

        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
                
                # Record overlap information
                overlaps.append({
                    'start': ov_start,
                    'end': ov_end,
                    'ramp_weights': w[:ramp_len].copy(),
                    'coords_prev': aligned[i - 1][ov_start - chunk_ranges[i - 1][0] : ov_end - chunk_ranges[i - 1][0]].copy(),
                    'coords_cur': coords[:ramp_len].copy()
                })

        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)

        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w

    mask = weights > 0
    full[mask] /= weights[mask, None]
    full = np.nan_to_num(full)

    # Apply Distance Matrix Blending MDS Relaxation around the seams
    for ov in overlaps:
        ov_s, ov_e = ov['start'], ov['end']
        
        # We expand the relaxation window slightly to ensure continuity
        margin = 10
        relax_s = max(0, ov_s - margin)
        relax_e = min(seq_len, ov_e + margin)
        
        if relax_e - relax_s > 3:
            # Re-extract coordinates with margins if possible, but the blended region itself is the core.
            # For exact intrinsic blending, we apply the relaxation directly on the overlapping portion.
            blended_overlap = full[ov_s:ov_e].copy()
            c1 = ov['coords_prev']
            c2 = ov['coords_cur']
            
            # Relax the overlap
            relaxed_overlap = distance_matrix_relaxation(
                c1, c2, blended_overlap, ov['ramp_weights'], num_iters=150
            )
            
            # Insert back
            full[ov_s:ov_e] = relaxed_overlap
            
            # Do a tiny laplacian smoothing pass on the boundaries to ensure C0 continuity
            for _ in range(5):
                for b_idx in [ov_s, ov_e - 1]:
                    if b_idx > 0 and b_idx < seq_len - 1:
                        full[b_idx] = 0.5 * (full[b_idx - 1] + full[b_idx + 1])

    return full

"""

new_code = code[:start_idx] + new_stitch_code + "\n" + code[end_idx:]

with open('notebooks/script_v74.py', 'w') as f:
    f.write(new_code)
print("script_v74.py written successfully.")
