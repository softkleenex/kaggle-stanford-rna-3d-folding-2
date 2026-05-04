import sys
import re

with open(sys.argv[1], 'r') as f:
    code = f.read()

# Replace hardcoded DATA_BASE logic with robust dynamic finding
code = re.sub(
    r'def setup_environment\(\).*?return TEST_CSV, OUTPUT_CSV, CODE_DIR, ROOT_DIR',
    '''def setup_environment():
    import os
    from pathlib import Path
    
    # Dynamically find the test_sequences.csv file to locate DATA_BASE
    test_csv_path = next(Path('/kaggle/input').rglob('test_sequences.csv'), None)
    if test_csv_path:
        DATA_BASE = str(test_csv_path.parent)
        TEST_CSV = str(test_csv_path)
    elif os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        DATA_BASE = "/kaggle/input/stanford-rna-3d-folding-2"
        TEST_CSV = f"{DATA_BASE}/test_sequences.csv"
    else:
        DATA_BASE = "./data"
        TEST_CSV = f"{DATA_BASE}/test_sequences.csv"

    # Safely find Protenix Code Dir
    protenix_base = next(Path('/kaggle/input').rglob('protenix*'), None)
    if protenix_base and os.path.exists(protenix_base):
        CODE_DIR = str(protenix_base)
    elif os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        CODE_DIR = "/kaggle/input/protenix-v1-adjusted" # Fallback
    else:
        CODE_DIR = "./protenix"

    ROOT_DIR = CODE_DIR if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else "."
    OUTPUT_CSV = "/kaggle/working/submission.csv" if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else "./submission.csv"
    
    return TEST_CSV, OUTPUT_CSV, CODE_DIR, ROOT_DIR''',
    code, flags=re.DOTALL
)

# Replace the DEFAULT_* hardcoded paths with a dynamic helper function
code = re.sub(
    r'DEFAULT_TRAIN_CSV\s*=\s*"/kaggle.*?\nDEFAULT_TRAIN_LBLS.*?\nDEFAULT_VAL_CSV.*?\nDEFAULT_VAL_LBLS.*?\n',
    '''def find_dataset_file(filename):
    from pathlib import Path
    match = next(Path('/kaggle/input').rglob(filename), None)
    if match: return str(match)
    return f"/kaggle/input/stanford-rna-3d-folding-2/{filename}"

DEFAULT_TRAIN_CSV  = find_dataset_file("train_sequences.csv")
DEFAULT_TRAIN_LBLS = find_dataset_file("train_labels.csv")
DEFAULT_VAL_CSV    = find_dataset_file("validation_sequences.csv")
DEFAULT_VAL_LBLS   = find_dataset_file("validation_labels.csv")
''',
    code, flags=re.DOTALL
)

with open(sys.argv[1], 'w') as f:
    f.write(code)
