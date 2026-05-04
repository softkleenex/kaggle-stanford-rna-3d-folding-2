#!/bin/bash
# BROAD LOOP V2: Robust Monitoring & One-time Submission

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="broad_loop_v2.log"
HISTORY_FILE="history.log"

log() {
    echo "$(date +"[%Y-%m-%d %H:%M:%S]") | $1" | tee -a $LOG_FILE
}

while true; do
    if [ -f next_kernel.txt ]; then
        KERNEL_ID=$(cat next_kernel.txt)
        log "🚀 NEW TASK: $KERNEL_ID"
        rm next_kernel.txt
        
        # 1. PUSH
        log "Pushing kernel..."
        kaggle kernels push -p notebooks
        sleep 30
        
        # 2. MONITOR
        log "Monitoring execution..."
        while true; do
            STATUS=$(kaggle kernels status $KERNEL_ID 2>/dev/null)
            if [[ $STATUS == *"COMPLETE"* ]]; then
                log "✅ COMPLETE!"
                break
            elif [[ $STATUS == *"ERROR"* ]]; then
                log "❌ ERROR! Waiting for AI to fix."
                echo "$(date) | $KERNEL_ID | FAILED" >> $HISTORY_FILE
                # Wait for AI to provide a NEW next_kernel.txt
                continue 2 
            fi
            sleep 120
        done
        
        # 3. SUBMIT ONCE
        log "Submitting to leaderboard..."
        kaggle competitions submit -c $COMPETITION -k $KERNEL_ID -f submission.csv -m "Auto v2 Loop: $KERNEL_ID" -v 1
        
        # 4. WAIT FOR SCORE
        log "Waiting for scoring..."
        while true; do
            LATEST=$(kaggle competitions submissions -c $COMPETITION --csv | head -n 2 | tail -n 1)
            if [[ $LATEST == *"pending"* ]] || [[ $LATEST == *"scoring"* ]]; then
                sleep 60
            else
                SCORE=$(echo "$LATEST" | awk -F',' '{print $5}')
                log "🎯 SCORE ACHIEVED: $SCORE"
                echo "$(date) | $KERNEL_ID | COMPLETE | Score: $SCORE" >> $HISTORY_FILE
                break
            fi
        done
        
        log "Cycle finished. Ready for next AI injection."
    else
        # Just heartbeat
        sleep 30
    fi
done
