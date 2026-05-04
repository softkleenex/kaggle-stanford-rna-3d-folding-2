#!/bin/bash
# INFINITE Autonomous Broad Exploration Loop

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="infinite_broad.log"

echo "=== STARTING INFINITE BROAD LOOP ===" > $LOG_FILE

# We have already submitted v24, and v26 is currently running via broad_watcher_v2.sh.
# This loop will take over the infinite aspect.

function wait_and_submit() {
    local kernel=$1
    echo "$(date) | Monitoring $kernel..." >> $LOG_FILE
    while true; do
        STATUS=$(kaggle kernels status $kernel 2>/dev/null)
        if [[ $STATUS == *"COMPLETE"* ]]; then
            echo "$(date) | $kernel is COMPLETE. Submitting..." >> $LOG_FILE
            mkdir -p submissions/$(basename $kernel)
            kaggle kernels output $kernel -p submissions/$(basename $kernel) --force
            kaggle competitions submit -c $COMPETITION -k $kernel -f submission.csv -m "Auto Submit: $kernel" -v 1
            return 0
        elif [[ $STATUS == *"ERROR"* ]]; then
            echo "$(date) | $kernel FAILED. Will require AI analysis." >> $LOG_FILE
            return 1
        fi
        sleep 60
    done
}

function wait_for_score() {
    echo "$(date) | Waiting for score..." >> $LOG_FILE
    while true; do
        LATEST_SUB=$(kaggle competitions submissions -c $COMPETITION --csv | head -n 2 | tail -n 1)
        if [[ $LATEST_SUB == *"pending"* ]] || [[ $LATEST_SUB == *"scoring"* ]]; then
            sleep 60
        else
            SCORE=$(echo "$LATEST_SUB" | awk -F',' '{print $5}')
            echo "$(date) | 🎯 NEW SCORE: $SCORE" >> $LOG_FILE
            # Stop the loop and return control to AI for the next BIG idea
            echo "TRIGGER_AI_NEXT_PHASE" >> $LOG_FILE
            return 0
        fi
    done
}

# The true "Infinite" part:
while true; do
    echo "$(date) | Loop iteration started." >> $LOG_FILE
    
    # We check if there's a file called 'next_kernel.txt' which the AI creates
    # when it wants to deploy a new architecture.
    if [ -f next_kernel.txt ]; then
        NEXT_KERNEL=$(cat next_kernel.txt)
        echo "$(date) | Picked up new kernel: $NEXT_KERNEL" >> $LOG_FILE
        
        kaggle kernels push -p notebooks
        sleep 30
        
        wait_and_submit "$NEXT_KERNEL"
        if [ $? -eq 0 ]; then
            wait_for_score
        fi
        
        # Remove the file so we don't re-run it
        rm next_kernel.txt
    else
        echo "$(date) | Waiting for AI to design the next architecture (next_kernel.txt not found)..." >> $LOG_FILE
        sleep 60
    fi
done
