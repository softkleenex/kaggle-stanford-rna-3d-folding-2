#!/bin/bash
echo "Starting Broad Autonomous Loop Watcher V2" > broad_loop_v2.log

function wait_and_submit() {
    local kernel=$1
    echo "Monitoring $kernel..." >> broad_loop_v2.log
    while true; do
        STATUS=$(kaggle kernels status $kernel)
        if [[ $STATUS == *"COMPLETE"* ]]; then
            echo "$kernel is COMPLETE. Submitting..." >> broad_loop_v2.log
            kaggle kernels output $kernel -p submissions/$(basename $kernel) --force
            kaggle competitions submit -c stanford-rna-3d-folding-2 -k $kernel -f submission.csv -m "Auto Submit: $kernel" -v 1
            break
        elif [[ $STATUS == *"ERROR"* ]]; then
            echo "$kernel FAILED." >> broad_loop_v2.log
            break
        fi
        sleep 60
    done
}

wait_and_submit "softkleenex/rna-folding-v26-ribonanza-fixed"

echo "Broad Loop V2 Finished." >> broad_loop_v2.log
