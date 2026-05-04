#!/bin/bash

# Configuration
COMPETITION="stanford-rna-3d-folding-2"
METADATA_DIR="notebooks"
LOG_FILE="state.log"
HISTORY_FILE="history.log"
PLAN_FILE="current_plan.md"

function log() {
    echo "$(date +"[%Y-%m-%d %H:%M:%S]") $1" | tee -a $LOG_FILE
}

function update_history() {
    local kernel=$1
    local status=$2
    local score=$3
    echo "$(date +"[%Y-%m-%d %H:%M:%S]") | Kernel: $kernel | Status: $status | Score: $score" >> $HISTORY_FILE
}

log "Starting Competition Loop for $COMPETITION"

while true; do
    # Read the current plan
    if [ ! -f $PLAN_FILE ]; then
        log "No $PLAN_FILE found. Please provide initial plan."
        exit 1
    fi
    
    # macOS compatible grep/awk for extracting KERNEL_ID
    KERNEL_ID=$(awk -F'=' '/^KERNEL_ID=/ {print $2}' $PLAN_FILE)
    
    if [ -z "$KERNEL_ID" ]; then
        log "KERNEL_ID not defined in $PLAN_FILE. Exiting."
        exit 1
    fi

    log "-------------------------------------------------------"
    log "🚀 [STAGE 1] EXECUTING KERNEL: $KERNEL_ID"
    log "-------------------------------------------------------"
    STATUS=$(kaggle kernels status $KERNEL_ID 2>/dev/null)
    if [[ "$STATUS" == *"RUNNING"* ]] || [[ "$STATUS" == *"QUEUED"* ]] || [[ "$STATUS" == *"COMPLETE"* ]]; then
        log "Kernel is already pushed. Current status: $STATUS"
    else
        log "Pushing kernel..."
        kaggle kernels push -p $METADATA_DIR
        sleep 30
    fi
    
    log "👀 [STAGE 2] MONITORING EXECUTION"
    while true; do
        STATUS=$(kaggle kernels status $KERNEL_ID)
        log "Status: $STATUS"
        
        if [[ $STATUS == *"COMPLETE"* ]]; then
            log "✅ Kernel Execution COMPLETE!"
            EXEC_STATUS="SUCCESS"
            break
        elif [[ $STATUS == *"ERROR"* ]]; then
            log "❌ Kernel Execution FAILED."
            LOG_DIR="notebooks/logs_$(basename $KERNEL_ID)"
            mkdir -p $LOG_DIR
            kaggle kernels output $KERNEL_ID -p $LOG_DIR --force
            tail -n 50 $LOG_DIR/*.log > latest_error.txt
            EXEC_STATUS="FAILED"
            break
        elif [[ $STATUS == *"RUNNING"* ]]; then
            sleep 60
        else
            sleep 30
        fi
    done

    if [ "$EXEC_STATUS" == "FAILED" ]; then
        log "⚠️ Execution failed. Updating history and exiting loop for AI analysis."
        update_history "$KERNEL_ID" "FAILED" "N/A"
        exit 1
    fi

    log "-------------------------------------------------------"
    log "📤 [STAGE 3] SUBMITTING TO LEADERBOARD"
    log "-------------------------------------------------------"
    mkdir -p submissions/$(basename $KERNEL_ID)
    kaggle kernels output $KERNEL_ID -p submissions/$(basename $KERNEL_ID) --force
    
    log "Submitting to competition via Code Kernel..."
    # FIX: Added -v 1 to specify version for code competition
    kaggle competitions submit -c $COMPETITION -k $KERNEL_ID -f submission.csv -m "Auto Submit: $KERNEL_ID" -v 1
    
    sleep 30

    log "-------------------------------------------------------"
    log "🏆 [STAGE 4] WAITING FOR SCORE"
    log "-------------------------------------------------------"
    SCORE="N/A"
    while true; do
        SUB_STATUS=$(kaggle competitions submissions -c $COMPETITION --csv | head -n 2 | tail -n 1)
        
        if [[ $SUB_STATUS == *"pending"* ]] || [[ $SUB_STATUS == *"scoring"* ]]; then
            log "Still Scoring... $SUB_STATUS"
            sleep 60
        elif [[ $SUB_STATUS == *"error"* ]] || [[ $SUB_STATUS == *"Error"* ]]; then
            log "❌ Submission Scoring FAILED! $SUB_STATUS"
            SCORE="ERROR"
            break
        else
            SCORE=$(echo "$SUB_STATUS" | awk -F',' '{print $5}')
            log "✅ Scoring COMPLETE! 🎯 YOUR SCORE: $SCORE"
            break
        fi
    done

    log "📝 Updating history..."
    update_history "$KERNEL_ID" "COMPLETE" "$SCORE"
    
    log "-------------------------------------------------------"
    log "🔄 LOOP FINISHED. OVER TO AI FOR EVALUATION AND NEW PLAN."
    log "-------------------------------------------------------"
    
    exit 0
done