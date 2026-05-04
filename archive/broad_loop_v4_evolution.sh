#!/bin/bash
# BROAD LOOP V4: THE EVOLUTION (Fully Autonomous)
# This script manages [Execution -> Submission -> Analysis -> Evolution -> Repeat] forever.

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="broad_loop_v4.log"
HISTORY_FILE="history.log"
VERSION_FILE="current_version.txt"

# Initialize version if not exists
if [ ! -f "$VERSION_FILE" ]; then echo "28" > "$VERSION_FILE"; fi

log() {
    echo "$(date +"[%Y-%m-%d %H:%M:%S]") | $1" | tee -a "$LOG_FILE"
}

while true; do
    VER=$(cat "$VERSION_FILE")
    KERNEL_ID="softkleenex/rna-folding-v${VER}-auto"
    
    log "--- STARTING CYCLE FOR VERSION v$VER ---"
    
    # 1. ARCHITECTURE EVOLUTION (Call Gemini to write the code)
    log "🧠 AI Research & Code Generation for v$VER..."
    # We pass the last 20 lines of history so Gemini knows the context
    CONTEXT=$(tail -n 20 "$HISTORY_FILE")
    
    gemini -y -p "You are an autonomous ML researcher for the Kaggle RNA 3D Folding 2 competition. 
    History so far: $CONTEXT.
    Goal: Exceed score 0.44. 
    Current Plan: Research the 'RNAPro' architecture or 'OpenMM' geometry relaxation on Kaggle discussions. 
    Action: Write a complete, improved Python script for v$VER. 
    Rules: 
    1. Include the robust offline wheel installer at the top.
    2. Use 'resolve_kaggle_path' for data paths.
    3. Save the code to 'notebooks/script_v${VER}.py' using write_file.
    4. DO NOT use placeholders. DO NOT output an .ipynb, just raw .py."
    
    if [ ! -f "notebooks/script_v${VER}.py" ]; then
        log "❌ AI failed to produce script. Retrying in 5 min..."
        sleep 300
        continue
    fi

    # 2. PACKAGE & PUSH
    log "📦 Packaging v$VER into Notebook..."
    python3 -c "
import json
with open('notebooks/script_v${VER}.py', 'r') as f: code = f.read()
nb = {'cells': [{'cell_type': 'code', 'metadata': {'trusted': True}, 'outputs': [], 'source': [code]}], 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('notebooks/rna_folding_v${VER}.ipynb', 'w') as f: json.dump(nb, f, indent=1)
"
    cat > notebooks/kernel-metadata.json <<EOF
{
  "id": "$KERNEL_ID",
  "title": "RNA Folding v${VER} Auto",
  "code_file": "rna_folding_v${VER}.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": ["qiweiyin/protenix-v1-adjusted", "yuriygreben/stanford-rna-protenix-wheels-3-12"],
  "competition_sources": ["$COMPETITION"]
}
EOF
    log "🚀 Pushing kernel $KERNEL_ID..."
    kaggle kernels push -p notebooks
    
    # 3. MONITOR
    log "⏳ Monitoring execution..."
    while true; do
        STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null)
        if [[ "$STATUS" == *"COMPLETE"* ]]; then
            log "✅ Execution Complete!"
            break
        elif [[ "$STATUS" == *"ERROR"* ]]; then
            log "❌ Kernel Failed. Downloading logs for next AI cycle..."
            kaggle kernels output "$KERNEL_ID" -p "notebooks/logs_v$VER" --force
            echo "$(date) | v$VER | FAILED | Params: N/A" >> "$HISTORY_FILE"
            echo $((VER + 1)) > "$VERSION_FILE"
            continue 2
        fi
        sleep 180
    done
    
    # 4. SUBMIT (Handle limit and failures)
    log "📤 Submitting to Leaderboard..."
    SUB_OUT=$(kaggle competitions submit -c "$COMPETITION" -k "$KERNEL_ID" -f submission.csv -m "Auto v$VER" -v 1 2>&1)
    
    if [[ "$SUB_OUT" == *"403"* ]]; then
        log "⚠️ Submission Limit Reached. Waiting 1 hour..."
        sleep 3600
        # Re-attempt submission
        continue
    elif [[ "$SUB_OUT" == *"Error"* ]]; then
        log "⚠️ Submission Error. Skipping score check for this version."
    else
        # 5. WAIT FOR SCORE
        log "🏆 Waiting for Score..."
        while true; do
            LATEST=$(kaggle competitions submissions -c "$COMPETITION" --csv | head -n 2 | tail -n 1)
            if [[ "$LATEST" == *"pending"* ]] || [[ "$LATEST" == *"scoring"* ]]; then
                sleep 60
            else
                SCORE=$(echo "$LATEST" | awk -F',' '{print $5}')
                log "🎯 FINAL SCORE for v$VER: $SCORE"
                echo "$(date) | v$VER | COMPLETE | Score: $SCORE" >> "$HISTORY_FILE"
                break
            fi
        done
    fi
    
    # 6. ITERATE
    echo $((VER + 1)) > "$VERSION_FILE"
    log "🔄 CYCLE v$VER FINISHED. Moving to next..."
    sleep 60
done
