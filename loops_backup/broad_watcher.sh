#!/bin/bash
# Autonomous Loop Controller for Broad Exploration

echo "Starting Broad Autonomous Loop Watcher" > broad_loop.log

function wait_and_submit() {
    local kernel=$1
    echo "Monitoring $kernel..." >> broad_loop.log
    while true; do
        STATUS=$(kaggle kernels status $kernel)
        if [[ $STATUS == *"COMPLETE"* ]]; then
            echo "$kernel is COMPLETE. Submitting..." >> broad_loop.log
            kaggle kernels output $kernel -p submissions/$(basename $kernel) --force
            kaggle competitions submit -c stanford-rna-3d-folding-2 -k $kernel -f submission.csv -m "Auto Submit: $kernel" -v 1
            break
        elif [[ $STATUS == *"ERROR"* ]]; then
            echo "$kernel FAILED. Aborting submission." >> broad_loop.log
            break
        fi
        sleep 60
    done
}

# 1. Wait for v24 (Top 1 Solution with Kabsch Alignment)
wait_and_submit "softkleenex/rna-folding-v24-ensemble"

# 2. Automatically launch v25 (RibonanzaNet 3D)
echo "Building v25 RibonanzaNet..." >> broad_loop.log
python3 build_v25.py

cat > notebooks/kernel-metadata.json <<'EOF'
{
  "id": "softkleenex/rna-folding-v25-ribonanza",
  "title": "RNA Folding v25 Ribonanza",
  "code_file": "rna_folding_v25_ribonanza.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "shujun717/ribonanzanet-pretrained-checkpoint",
    "shujun717/ribonanzanet2d-final",
    "shujun717/ribonanzanet-3d-finetune"
  ],
  "competition_sources": ["stanford-rna-3d-folding-2"]
}
EOF

kaggle kernels push -p notebooks
wait_and_submit "softkleenex/rna-folding-v25-ribonanza"

echo "Broad Loop Finished. Waiting for AI next instructions." >> broad_loop.log
