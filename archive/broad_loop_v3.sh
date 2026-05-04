#!/bin/bash
# BROAD LOOP V3: ULTIMATE ROBUST VERSION
# This script ensures ONLY ONE submission per kernel version and handles autonomous architectural shifts.

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="broad_loop_v3.log"
HISTORY_FILE="history.log"
LOCK_FILE="loop.lock"

# Prevent multiple instances
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if ps -p "$PID" > /dev/null; then
        echo "Loop already running with PID $PID. Exiting."
        exit 1
    fi
fi
echo $$ > "$LOCK_FILE"

log() {
    echo "$(date +"[%Y-%m-%d %H:%M:%S]") | $1" | tee -a "$LOG_FILE"
}

check_submission_exists() {
    local kernel_desc=$1
    # Check if a submission with this description already exists in the last 10 entries
    kaggle competitions submissions -c "$COMPETITION" | head -n 15 | grep -q "$kernel_desc"
    return $?
}

while true; do
    if [ -f next_kernel.txt ]; then
        KERNEL_ID=$(cat next_kernel.txt)
        VERSION=$(echo "$KERNEL_ID" | grep -oE "v[0-9]+" | head -n 1)
        DESC="Auto v3: $KERNEL_ID"
        
        log "🚀 NEW TASK DETECTED: $KERNEL_ID ($VERSION)"
        rm next_kernel.txt
        
        # 1. PUSH (if not already running)
        STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null)
        if [[ "$STATUS" == *"RUNNING"* ]] || [[ "$STATUS" == *"COMPLETE"* ]]; then
            log "Kernel already pushed/running. Skipping push."
        else
            log "Pushing kernel..."
            kaggle kernels push -p notebooks
            sleep 30
        fi
        
        # 2. MONITOR
        log "Monitoring execution..."
        while true; do
            STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null)
            if [[ "$STATUS" == *"COMPLETE"* ]]; then
                log "✅ KERNEL COMPLETE!"
                break
            elif [[ "$STATUS" == *"ERROR"* ]]; then
                log "❌ KERNEL FAILED! Logging and waiting for AI manual intervention."
                echo "$(date) | $KERNEL_ID | FAILED" >> "$HISTORY_FILE"
                continue 2 
            fi
            sleep 120
        done
        
        # 3. SUBMIT (with duplicate check)
        if check_submission_exists "$DESC"; then
            log "⚠️ Submission for $DESC already exists. Skipping redundant submit."
        else
            log "Submitting to leaderboard..."
            kaggle competitions submit -c "$COMPETITION" -k "$KERNEL_ID" -f submission.csv -m "$DESC" -v 1
        fi
        
        # 4. WAIT FOR SCORE
        log "Waiting for score..."
        while true; do
            LATEST=$(kaggle competitions submissions -c "$COMPETITION" --csv | head -n 2 | tail -n 1)
            if [[ "$LATEST" == *"pending"* ]] || [[ "$LATEST" == *"scoring"* ]]; then
                sleep 60
            else
                SCORE=$(echo "$LATEST" | awk -F',' '{print $5}')
                log "🎯 SCORE ACHIEVED: $SCORE"
                echo "$(date) | $KERNEL_ID | COMPLETE | Score: $SCORE" >> "$HISTORY_FILE"
                
                # OPTIONAL: Here we can trigger the AI to design the NEXT kernel
                # echo "softkleenex/rna-folding-v$((VERSION_NUM + 1))" > next_kernel.txt
                break
            fi
        done
        
        log "Cycle finished for $KERNEL_ID. Awaiting next instructions or next_kernel.txt."
    else
        # Periodic check for current running kernels just in case
        sleep 60
    fi
done
