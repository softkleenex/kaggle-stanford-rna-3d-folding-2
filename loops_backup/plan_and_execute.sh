#!/bin/bash
# Autonomous Execution Controller

echo "=== Autonomous Loop Triggered ==="
echo "Writing new plan to current_plan.md..."

cat << 'PLAN' > current_plan.md
# Current Plan

## KERNEL_ID
KERNEL_ID=softkleenex/rna-folding-v24-ensemble

## Status
- **Goal**: Introduce OpenRNAFold to the pipeline and ensemble it with Protenix to break the 0.44 score barrier.
- **Hypothesis**: The OpenRNAFold model captures different structural motifs than Protenix. Blending their output coordinates (or falling back to one when the other fails) will result in a more robust 3D structure and a significantly higher score.
- **Action**: Modify the notebook generation script to include OpenRNAFold weights, build the v24 notebook, and push.
PLAN

echo "Executing build step for v24-ensemble..."
# We will use python to build the next notebook, then trigger run_competition.sh
python3 build_v24.py

# Push metadata update
cat > notebooks/kernel-metadata.json <<'META'
{
  "id": "softkleenex/rna-folding-v24-ensemble",
  "title": "RNA Folding v24 Ensemble",
  "code_file": "rna_folding_v24.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "qiweiyin/protenix-v1-adjusted",
    "yuriygreben/stanford-rna-protenix-wheels-3-12"
  ],
  "competition_sources": ["stanford-rna-3d-folding-2"]
}
META

echo "Triggering run_competition.sh in the background..."
./run_competition.sh > run_competition.out 2>&1 &
echo "Automation cycle successfully handed over to background process (PID: $!)."
