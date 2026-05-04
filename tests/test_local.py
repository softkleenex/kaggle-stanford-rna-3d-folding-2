import os
import sys
import pandas as pd
os.environ["KAGGLE_IS_COMPETITION_RERUN"] = ""
sys.path.insert(0, "notebooks")
try:
    import script_v349
    script_v349.IS_KAGGLE = False
    script_v349.LOCAL_N_SAMPLES = 2
    
    # Need to override find_data_base logic so it finds local files
    script_v349.DEFAULT_VAL_CSV = "data/raw/validation_sequences.csv"
    script_v349.DEFAULT_TRAIN_CSV = "data/raw/train_sequences.csv"
    script_v349.DEFAULT_TRAIN_LBLS = "data/raw/train_labels.csv"
    script_v349.DEFAULT_VAL_LBLS = "data/raw/validation_labels.csv"
    
    script_v349.main()
except Exception as e:
    print(e)
