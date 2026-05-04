import sys

with open("notebooks/script_v345.py", "r") as f:
    content = f.read()

# Replace the STRATEGY string
old_strategy = "STRATEGY: Implement a 10.5 Å distance constraint between C1' atoms of base-paired residues to stabilize the canonical A-form helix width and refine the inter-strand pairing geometry."
new_strategy = "STRATEGY: Implement a 3.8 Å distance constraint between O2' atoms of adjacent residues i and i+1 to stabilize the A-form ribose stacking and reinforce the characteristic C3'-endo pucker."
content = content.replace(old_strategy, new_strategy)

# Insert the O2'-O2' distance constraint
o2_constraint = """        # ── O2'(i) - O2'(i+1) Distance Constraint ───────────────────────────
        # Implement a 3.8 Å distance constraint between O2' atoms of adjacent 
        # residues i and i+1 to stabilize the A-form ribose stacking and 
        # reinforce the characteristic C3'-endo pucker.
        o2_strength = 0.25 * strength
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            in_stem_o2 = np.zeros(L, dtype=bool)
            if stems:
                for stem in stems:
                    for i_s, j_s in stem:
                        if s <= i_s < e: in_stem_o2[i_s - s] = True
                        if s <= j_s < e: in_stem_o2[j_s - s] = True
            
            # Estimate O2' position using local normal
            v1 = C[2:] - C[1:-1]; v2 = C[:-2] - C[1:-1]
            n_vec = np.cross(v1, v2); dn = np.linalg.norm(n_vec, axis=1, keepdims=True) + 1e-6
            o2_pos = C[1:-1] + (n_vec / dn) * 2.5
            
            o2_mask = in_stem_o2[1:-2] & in_stem_o2[2:-1]
            if np.any(o2_mask):
                d_o2 = o2_pos[1:] - o2_pos[:-1]; dist_o2 = np.linalg.norm(d_o2, axis=1) + 1e-6
                adj_o2 = d_o2[o2_mask] * ((3.8 - dist_o2[o2_mask]) / dist_o2[o2_mask])[:, None] * o2_strength
                C[1:-2][o2_mask] -= adj_o2 * 0.5; C[2:-1][o2_mask] += adj_o2 * 0.5
            X[s:e] = C

        # ── O2'(i) ... O4'(i+1) Backbone Stabilization (OBS) Constraint"""

old_obs = "        # ── O2'(i) ... O4'(i+1) Backbone Stabilization (OBS) Constraint"
content = content.replace(old_obs, o2_constraint)

with open("notebooks/script_v346.py", "w") as f:
    f.write(content)

print("Modification complete.")
