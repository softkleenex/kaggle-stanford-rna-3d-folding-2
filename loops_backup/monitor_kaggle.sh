#!/bin/bash

# Configuration
KERNEL_REF="softkleenex/rna-folding-v2-final"
METADATA_DIR="notebooks"

echo "🚀 Pushing kernel: $KERNEL_REF"
kaggle kernels push -p $METADATA_DIR

# Wait for 30s to start
sleep 30

while true; do
    STATUS_JSON=$(kaggle kernels status $KERNEL_REF)
    echo "Current Status: $STATUS_JSON"
    
    if [[ $STATUS_JSON == *"KernelWorkerStatus.COMPLETE"* ]]; then
        echo "✅ SUCCESS: Kernel finished successfully!"
        break
    elif [[ $STATUS_JSON == *"KernelWorkerStatus.ERROR"* ]]; then
        echo "❌ ERROR: Kernel failed. Downloading logs..."
        kaggle kernels output $KERNEL_REF -p notebooks/logs --force
        tail -n 100 notebooks/logs/rna-folding-baseline-protenix-tbm.log
        echo "Wait 2 minutes and retry..."
        sleep 120
        echo "🚀 Retrying push..."
        kaggle kernels push -p $METADATA_DIR
        sleep 60
    elif [[ $STATUS_JSON == *"KernelWorkerStatus.RUNNING"* ]]; then
        echo "⏳ Running... check again in 2 minutes."
        sleep 120
    elif [[ $STATUS_JSON == *"KernelWorkerStatus.QUEUED"* ]]; then
        echo "⏳ Queued... check again in 1 minute."
        sleep 60
    else
        echo "❓ Unknown status. Check manually. $STATUS_JSON"
        sleep 60
    fi
done
