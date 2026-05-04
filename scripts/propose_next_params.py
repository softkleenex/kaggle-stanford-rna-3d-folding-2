import os
import random

HISTORY_FILE = "history.log"

def get_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return f.readlines()

def propose():
    history = get_history()
    
    # Defaults
    n_sample = 5
    min_similarity = 0.0
    chunk_overlap = 128
    
    tried_combos = []
    for line in history:
        if "Params:" in line:
            try:
                # Format: ... | Params: N_SAMPLE=5, MIN_SIMILARITY=0.0, CHUNK_OVERLAP=128
                params_part = line.split("Params:")[1].strip()
                p_dict = dict(item.split("=") for item in params_part.split(", "))
                tried_combos.append((int(p_dict["N_SAMPLE"]), float(p_dict["MIN_SIMILARITY"]), int(p_dict["CHUNK_OVERLAP"])))
            except:
                continue

    # Simple logic: explore higher N_SAMPLE or different MIN_SIMILARITY
    options_n = [5, 8, 10, 12]
    options_sim = [0.0, 0.1, 0.2, 0.3]
    options_overlap = [128, 192, 256]
    
    while True:
        n = random.choice(options_n)
        s = random.choice(options_sim)
        o = random.choice(options_overlap)
        if (n, s, o) not in tried_combos:
            return n, s, o

if __name__ == "__main__":
    n, s, o = propose()
    print(f"N_SAMPLE={n}")
    print(f"MIN_SIMILARITY={s}")
    print(f"CHUNK_OVERLAP={o}")
