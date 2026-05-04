#!/bin/bash

# Configuration
KERNEL_ID=$1
COMPETITION="stanford-rna-3d-folding-2"
METADATA_DIR="notebooks"

if [ -z "$KERNEL_ID" ]; then
    echo "Usage: ./auto_pipeline.sh <softkleenex/kernel-name>"
    exit 1
fi

echo "======================================================="
echo "🚀 [STAGE 1] PUSHING KERNEL: $KERNEL_ID"
echo "======================================================="
kaggle kernels push -p $METADATA_DIR

echo "⏳ Waiting 30s for kernel to initialize..."
sleep 30

echo "======================================================="
echo "👀 [STAGE 2] MONITORING EXECUTION"
echo "======================================================="
while true; do
    STATUS=$(kaggle kernels status $KERNEL_ID)
    TIMESTAMP=$(date +"[%H:%M:%S]")
    echo "$TIMESTAMP $STATUS"
    
    if [[ $STATUS == *"COMPLETE"* ]]; then
        echo "✅ Kernel Execution COMPLETE!"
        break
    elif [[ $STATUS == *"ERROR"* ]]; then
        echo "❌ Kernel Execution FAILED."
        echo "Downloading logs for AI analysis..."
        LOG_DIR="notebooks/logs_$(basename $KERNEL_ID)"
        mkdir -p $LOG_DIR
        kaggle kernels output $KERNEL_ID -p $LOG_DIR --force
        tail -n 50 $LOG_DIR/*.log
        echo "⚠️ OVER TO AI: Please analyze the logs, fix the code, and rerun the pipeline."
        exit 1
    elif [[ $STATUS == *"RUNNING"* ]]; then
        sleep 60
    else
        sleep 30
    fi
done

echo "======================================================="
echo "📤 [STAGE 3] SUBMITTING TO LEADERBOARD"
echo "======================================================="
kaggle competitions submit -c $COMPETITION -k $KERNEL_ID -f submission.csv -m "Auto Submit: $KERNEL_ID"

echo "⏳ Waiting 30s for submission to register..."
sleep 30

echo "======================================================="
echo "🏆 [STAGE 4] WAITING FOR SCORE"
echo "======================================================="
while true; do
    SUB_STATUS=$(kaggle competitions submissions -c $COMPETITION --csv | head -n 2 | tail -n 1)
    TIMESTAMP=$(date +"[%H:%M:%S]")
    
    # Check if the submission is still pending/scoring
    if [[ $SUB_STATUS == *"pending"* ]] || [[ $SUB_STATUS == *"scoring"* ]]; then
        echo "$TIMESTAMP Still Scoring... $SUB_STATUS"
        sleep 60
    elif [[ $SUB_STATUS == *"error"* ]] || [[ $SUB_STATUS == *"Error"* ]]; then
        echo "❌ $TIMESTAMP Submission Scoring FAILED!"
        echo "$SUB_STATUS"
        echo "⚠️ OVER TO AI: Check submission format rules."
        exit 1
    else
        # It's complete! Extract the score
        SCORE=$(echo "$SUB_STATUS" | awk -F',' '{print $5}')
        echo "✅ $TIMESTAMP Scoring COMPLETE! 🎯 YOUR SCORE: $SCORE"
        break
    fi
done

echo "======================================================="
echo "🎉 PIPELINE FINISHED. OVER TO AI FOR NEXT ITERATION."
echo "======================================================="
