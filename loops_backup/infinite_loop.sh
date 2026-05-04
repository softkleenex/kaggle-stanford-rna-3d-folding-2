#!/bin/bash

COMPETITION="stanford-rna-3d-folding-2"
METADATA_DIR="notebooks"
LOG_FILE="loop_output.log"
HISTORY_FILE="history.log"

function log() {
    echo "$(date +"[%Y-%m-%d %H:%M:%S]") $1" | tee -a $LOG_FILE
}

# Start version
VERSION=26

log "=== INITIALIZING INFINITE AUTO-LOOP V2 ==="

while true; do
    log "--- VERSION v$VERSION START ---"
    
    # 1. Propose Params
    log "Proposing next parameters..."
    PARAMS=$(python3 scripts/propose_next_params.py)
    log "Proposed: $PARAMS"
    
    # Extract values
    N_SAMPLE=$(echo "$PARAMS" | grep N_SAMPLE | cut -d'=' -f2)
    MIN_SIM=$(echo "$PARAMS" | grep MIN_SIMILARITY | cut -d'=' -f2)
    OVERLAP=$(echo "$PARAMS" | grep CHUNK_OVERLAP | cut -d'=' -f2)
    
    # 2. Build Notebook
    log "Building notebook v$VERSION..."
    KERNEL_ID=$(python3 scripts/build_dynamic.py $N_SAMPLE $MIN_SIM $OVERLAP $VERSION)
    log "Kernel ID: $KERNEL_ID"
    
    # 3. Push Kernel
    log "Pushing kernel to Kaggle..."
    kaggle kernels push -p $METADATA_DIR
    sleep 30
    
    # 4. Monitor
    log "Monitoring execution..."
    while true; do
        STATUS=$(kaggle kernels status $KERNEL_ID)
        log "Current Status: $STATUS"
        
        if [[ $STATUS == *"COMPLETE"* ]]; then
            log "✅ Execution Complete!"
            break
        elif [[ $STATUS == *"ERROR"* ]]; then
            log "❌ Execution Failed. Check logs manually. Skipping to next iteration."
            echo "$(date) | v$VERSION | FAILED | Params: $PARAMS" >> $HISTORY_FILE
            VERSION=$((VERSION + 1))
            continue 2
        fi
        sleep 120
    done
    
    # 5. Submit
    log "Submitting to leaderboard..."
    # Try kernel-based submission first
    SUB_OUT=$(kaggle competitions submit -c $COMPETITION -k $KERNEL_ID -f submission.csv -m "Auto v$VERSION: $PARAMS" -v 1 2>&1)
    
    if [[ $SUB_OUT == *"400"* ]] || [[ $SUB_OUT == *"Error"* ]]; then
        log "⚠️ Kernel submission failed. Trying fallback (Download & Upload)..."
        mkdir -p temp_sub
        kaggle kernels output $KERNEL_ID -p temp_sub --force
        kaggle competitions submit -c $COMPETITION -f temp_sub/submission.csv -m "Fallback v$VERSION: $PARAMS"
    fi
    
    # 6. Wait for score
    log "Waiting for score..."
    while true; do
        LATEST_SUB=$(kaggle competitions submissions -c $COMPETITION --csv | head -n 2 | tail -n 1)
        if [[ $LATEST_SUB == *"pending"* ]] || [[ $LATEST_SUB == *"scoring"* ]]; then
            sleep 60
        else
            SCORE=$(echo "$LATEST_SUB" | awk -F',' '{print $5}')
            log "🎯 FINAL SCORE for v$VERSION: $SCORE"
            echo "$(date) | v$VERSION | COMPLETE | Score: $SCORE | Params: $PARAMS" >> $HISTORY_FILE
            break
        fi
    done
    
    log "--- VERSION v$VERSION FINISHED ---"
    VERSION=$((VERSION + 1))
    
    # Brief cooldown
    sleep 60
done
