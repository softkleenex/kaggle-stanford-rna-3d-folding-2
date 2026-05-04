import pandas as pd
import numpy as np

# Load train labels
df = pd.read_csv('/kaggle/input/stanford-rna-3d-folding-2/train_labels.csv')
prefixes = df["ID"].str.rsplit("_", n=1).str[0]
avg_dists = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

for prefix, grp in df.groupby(prefixes):
    coords = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    for k in range(1, 11):
        if len(coords) > k:
            d = np.linalg.norm(coords[k:] - coords[:-k], axis=1)
            avg_dists[k].extend(d)

for k in range(1, 11):
    print(f"i to i+{k}: mean={np.nanmean(avg_dists[k]):.2f}, median={np.nanmedian(avg_dists[k]):.2f}, std={np.nanstd(avg_dists[k]):.2f}")
