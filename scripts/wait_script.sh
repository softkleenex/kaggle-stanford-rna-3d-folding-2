#!/bin/bash
echo "Monitoring state.log for completion..."
END_TIME=$((SECONDS + 270))
while [ $SECONDS -lt $END_TIME ]; do
    if grep -q "LOOP FINISHED" state.log || grep -q "Execution failed" state.log || grep -q "Scoring FAILED" state.log; then
        echo "Pipeline finished or failed!"
        tail -n 15 state.log
        exit 0
    fi
    sleep 30
done
echo "Timeout reached. Still running."
tail -n 5 state.log
exit 0
